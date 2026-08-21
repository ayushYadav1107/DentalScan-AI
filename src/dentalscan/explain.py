"""Visual explanation for YOLO detectors, with faithfulness checks.

Pretty heatmaps are easy and prove nothing. This module produces saliency maps
*and* two quantitative checks that the maps actually describe the model:

**Pointing-game energy** - the fraction of saliency mass that falls inside the
boxes the model predicted. A map that puts most of its energy outside every
predicted box is not explaining the prediction.

**Deletion AUC** - progressively blank the highest-saliency pixels and watch the
target class score fall. A faithful map causes a fast drop (low AUC); an
unfaithful one barely moves the score. Reported against a random-order baseline
so the number is interpretable.

Two saliency methods are provided:

* :class:`EigenCAM` - the first principal component of a chosen feature map.
  Gradient-free, so it works unchanged across YOLOv8/v10/v12 heads and never
  suffers from the vanishing-gradient artefacts that plague Grad-CAM on
  detection heads. This is the default.
* :class:`GradCAM` - gradient-weighted activations with a detection-specific
  scalar target (the summed class logits for one class across all anchors),
  which makes the map class-conditional: you can ask "where does the model see
  *caries*", not just "where does it look".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .constants import CLASS_NAMES


# --------------------------------------------------------------------------- #
# Layer selection
# --------------------------------------------------------------------------- #

def list_candidate_layers(model) -> list[tuple[int, str]]:
    """Enumerate indexable modules of an Ultralytics model, for `--layer`."""
    core = model.model.model if hasattr(model.model, "model") else model.model
    return [(i, type(m).__name__) for i, m in enumerate(core)]


def default_layer_index(model) -> int:
    """Last layer before the detection head.

    That is the deepest purely-convolutional feature map: semantically rich
    enough to be class-discriminative, still spatial enough to localise. Taking
    a layer inside the head instead gives maps dominated by anchor grid
    structure rather than image content.
    """
    core = model.model.model if hasattr(model.model, "model") else model.model
    head_names = {"Detect", "v10Detect", "Segment", "Pose", "OBB"}
    for i in range(len(core) - 1, -1, -1):
        if type(core[i]).__name__ not in head_names:
            return i
    return len(core) - 2


# --------------------------------------------------------------------------- #
# CAM implementations
# --------------------------------------------------------------------------- #

class _ActivationHook:
    """Capture the forward activation (and optionally the gradient) of a layer."""

    def __init__(self, module, retain_grad: bool = False) -> None:
        self.activation = None
        self.gradient = None
        self._retain_grad = retain_grad
        self._handles = [module.register_forward_hook(self._forward)]

    def _forward(self, _module, _inputs, output):
        tensor = output[0] if isinstance(output, (tuple, list)) else output
        self.activation = tensor
        if self._retain_grad and tensor.requires_grad:
            tensor.retain_grad()
            self._handles.append(tensor.register_hook(self._save_grad))

    def _save_grad(self, grad):
        self.gradient = grad

    def close(self) -> None:
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass


def _normalise(cam: np.ndarray) -> np.ndarray:
    cam = np.nan_to_num(cam, nan=0.0, posinf=0.0, neginf=0.0)
    cam = cam - cam.min()
    peak = cam.max()
    return cam / peak if peak > 0 else cam


def _resize(cam: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    import cv2
    return cv2.resize(cam.astype(np.float32), (size[1], size[0]), interpolation=cv2.INTER_LINEAR)


class EigenCAM:
    """Gradient-free saliency: first principal component of the activations."""

    name = "eigencam"

    def __init__(self, model, layer_index: int | None = None) -> None:
        self.model = model
        core = model.model.model if hasattr(model.model, "model") else model.model
        self.layer_index = default_layer_index(model) if layer_index is None else layer_index
        self.module = core[self.layer_index]

    def __call__(self, image: np.ndarray, target_class: int | None = None,
                 imgsz: int = 640) -> np.ndarray:
        import torch

        hook = _ActivationHook(self.module, retain_grad=False)
        try:
            with torch.no_grad():
                self.model.predict(image, imgsz=imgsz, verbose=False)
            activation = hook.activation
            if activation is None:
                raise RuntimeError(
                    f"Layer {self.layer_index} produced no activation. "
                    "Use list_candidate_layers() to pick another."
                )
            feat = activation[0].detach().float().cpu().numpy()  # (C, H, W)
        finally:
            hook.close()

        channels, height, width = feat.shape
        flat = feat.reshape(channels, height * width)
        flat = flat - flat.mean(axis=1, keepdims=True)
        # Top singular vector of the (H*W, C) matrix == first PC of the spatial map.
        try:
            _u, _s, vt = np.linalg.svd(flat.T, full_matrices=False)
            cam = (flat.T @ vt[0]).reshape(height, width)
        except np.linalg.LinAlgError:
            cam = feat.mean(axis=0)
        # Sign of a principal component is arbitrary; orient it towards the
        # high-activation region so the heatmap is comparable across images.
        if np.corrcoef(cam.ravel(), feat.mean(axis=0).ravel())[0, 1] < 0:
            cam = -cam
        cam = np.maximum(cam, 0)
        return _normalise(_resize(cam, image.shape[:2]))


class GradCAM:
    """Class-conditional saliency using gradients of the detection head."""

    name = "gradcam"

    def __init__(self, model, layer_index: int | None = None) -> None:
        self.model = model
        core = model.model.model if hasattr(model.model, "model") else model.model
        self.layer_index = default_layer_index(model) if layer_index is None else layer_index
        self.module = core[self.layer_index]

    @staticmethod
    def _class_score(raw, target_class: int, top_k: int = 20):
        """Scalar target: summed logits of ``target_class`` over top-k anchors.

        Ultralytics detection heads emit ``(B, 4 + num_classes, num_anchors)``.
        Summing only the strongest anchors keeps the gradient focused on the
        locations that drive the actual detections rather than on the thousands
        of near-zero background anchors.
        """
        import torch

        tensor = raw[0] if isinstance(raw, (tuple, list)) else raw
        if tensor.dim() != 3:
            raise RuntimeError(f"Unexpected head output shape {tuple(tensor.shape)}")
        scores = tensor[0, 4:, :]                       # (num_classes, num_anchors)
        if target_class >= scores.shape[0]:
            raise IndexError(
                f"target_class={target_class} but head predicts {scores.shape[0]} classes"
            )
        row = scores[target_class]
        k = min(top_k, row.numel())
        return torch.topk(row, k).values.sum()

    def __call__(self, image: np.ndarray, target_class: int = 0,
                 imgsz: int = 640) -> np.ndarray:
        import torch

        core_model = self.model.model
        core_model.eval()
        for param in core_model.parameters():
            param.requires_grad_(True)

        device = next(core_model.parameters()).device
        tensor = self._preprocess(image, imgsz).to(device)

        hook = _ActivationHook(self.module, retain_grad=True)
        try:
            core_model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                raw = core_model(tensor)
                score = self._class_score(raw, target_class)
                score.backward()

            activation = hook.activation
            gradient = hook.gradient
            if activation is None or gradient is None:
                raise RuntimeError(
                    "No gradient reached the chosen layer. Try a shallower layer, "
                    "or use EigenCAM which needs no gradients."
                )
            act = activation[0].detach().float().cpu().numpy()
            grad = gradient[0].detach().float().cpu().numpy()
        finally:
            hook.close()
            core_model.zero_grad(set_to_none=True)

        weights = grad.mean(axis=(1, 2))               # global-average-pooled gradients
        cam = np.maximum((weights[:, None, None] * act).sum(axis=0), 0)
        return _normalise(_resize(cam, image.shape[:2]))

    @staticmethod
    def _preprocess(image: np.ndarray, imgsz: int):
        import cv2
        import torch

        resized = cv2.resize(image, (imgsz, imgsz))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)


CAM_METHODS = {"eigencam": EigenCAM, "gradcam": GradCAM}


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def overlay_cam(image: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a normalised saliency map over a BGR image."""
    import cv2

    heat = cv2.applyColorMap((255 * np.clip(cam, 0, 1)).astype(np.uint8), cv2.COLORMAP_JET)
    if heat.shape[:2] != image.shape[:2]:
        heat = cv2.resize(heat, (image.shape[1], image.shape[0]))
    base = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(heat, alpha, base, 1 - alpha, 0)


# --------------------------------------------------------------------------- #
# Faithfulness metrics
# --------------------------------------------------------------------------- #

@dataclass
class FaithfulnessScores:
    pointing_energy: float          # saliency mass inside predicted boxes, 0..1
    box_area_fraction: float        # what that would be by chance
    energy_lift: float              # pointing_energy / box_area_fraction
    deletion_auc: float             # lower is better
    deletion_auc_random: float      # baseline with random pixel order
    deletion_gain: float            # random - cam;  > 0 means the map is informative


def pointing_energy(cam: np.ndarray, boxes: np.ndarray) -> tuple[float, float]:
    """Fraction of saliency inside predicted boxes, and the chance rate.

    Comparing the two is what makes the number meaningful: a map that is uniform
    everywhere scores exactly the box area fraction, so only the ratio above 1
    is evidence of localisation.
    """
    if cam.size == 0:
        return float("nan"), float("nan")
    mask = np.zeros(cam.shape, dtype=bool)
    for x1, y1, x2, y2 in np.asarray(boxes, dtype=int).reshape(-1, 4):
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, cam.shape[1]), min(y2, cam.shape[0])
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
    total = cam.sum()
    inside = cam[mask].sum() if mask.any() else 0.0
    return (
        float(inside / total) if total > 0 else float("nan"),
        float(mask.mean()),
    )


def deletion_curve(
    model,
    image: np.ndarray,
    cam: np.ndarray,
    target_class: int,
    steps: int = 12,
    order: str = "cam",
    seed: int = 0,
    imgsz: int = 640,
) -> np.ndarray:
    """Target-class score as high-saliency pixels are progressively removed."""
    rng = np.random.default_rng(seed)
    flat = cam.ravel()
    ranking = (
        np.argsort(-flat) if order == "cam"
        else rng.permutation(flat.size)
    )

    scores: list[float] = []
    n_pixels = flat.size
    for step in range(steps + 1):
        occluded = image.copy().reshape(-1, image.shape[-1]) if image.ndim == 3 else image.copy().ravel()
        k = int(n_pixels * step / steps)
        if k:
            occluded[ranking[:k]] = 0
        occluded = occluded.reshape(image.shape)

        result = model.predict(occluded, conf=0.001, imgsz=imgsz, verbose=False)[0]
        if result.boxes is not None and len(result.boxes):
            cls = result.boxes.cls.cpu().numpy().astype(int)
            conf = result.boxes.conf.cpu().numpy()
            match = conf[cls == target_class]
            scores.append(float(match.max()) if match.size else 0.0)
        else:
            scores.append(0.0)
    return np.asarray(scores)


def faithfulness(
    model,
    image: np.ndarray,
    cam: np.ndarray,
    boxes: np.ndarray,
    target_class: int,
    steps: int = 12,
    imgsz: int = 640,
) -> FaithfulnessScores:
    """Full faithfulness report for one image."""
    energy, chance = pointing_energy(cam, boxes)
    cam_curve = deletion_curve(model, image, cam, target_class, steps=steps,
                               order="cam", imgsz=imgsz)
    rand_curve = deletion_curve(model, image, cam, target_class, steps=steps,
                                order="random", imgsz=imgsz)
    auc_cam = float(np.trapezoid(cam_curve, dx=1.0 / steps))
    auc_rand = float(np.trapezoid(rand_curve, dx=1.0 / steps))
    return FaithfulnessScores(
        pointing_energy=energy,
        box_area_fraction=chance,
        energy_lift=float(energy / chance) if chance and chance > 0 else float("nan"),
        deletion_auc=auc_cam,
        deletion_auc_random=auc_rand,
        deletion_gain=auc_rand - auc_cam,
    )
