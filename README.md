# DentalScan AI

Six-class dental condition detection on panoramic radiographs (OPG), with a
research harness for measuring it honestly.

**[Live demo](https://dentalscan-ai.streamlit.app/)** ·
**[Technical report (PDF)](report/main.pdf)** ·
**[Model card](MODEL_CARD.md)**

---

## What is here

Two things share this repository:

1. **A working detector and a Streamlit interface.** Upload an OPG, get
   colour-coded boxes with confidence scores and a saliency overlay showing
   where the model looked.
2. **A measurement harness.** Per-class metrics with bootstrap confidence
   intervals, grouped cross-validation, a declarative multi-seed ablation grid,
   TIDE-style error decomposition, and saliency maps scored for faithfulness
   rather than shown as illustrations.

The second exists because the first was originally reported as a single number,
`mAP@0.5 = 0.923`, and that number turns out not to be a measurement. The
[technical report](report/main.pdf) explains why in four pages; the short version
is below.

---

## Results

### Per-class AP@0.5, three architectures, identical recipe

| Class     | N (val) | YOLOv8n | YOLOv10s  | YOLOv12m  |
|-----------|--------:|--------:|----------:|----------:|
| Caries    | 14      | 0.781   | 0.849     | **0.875** |
| Healthy   | 67      | 0.734   | 0.837     | **0.850** |
| Impacted  | 18      | 0.978   | **0.995** | **0.995** |
| Infection | 4 \*    | 0.750   | 0.870     | 0.788     |
| Fractured | 9 \*    | 0.973   | 0.995     | 0.968     |
| BDC/BDR   | 3 \*    | 0.671   | 0.995     | 0.995     |
| **mAP@0.5 — all six classes**   | | 0.815 | **0.923** | 0.912 |
| **mAP@0.5 — the three with N ≥ 10** | | 0.831 | 0.894 | **0.907** |

\* fewer than 10 validation instances. These are arithmetic, not evidence.

Two things the aggregate number hid. The classes driving the high mean are
exactly the three with the least support. And the ranking of the two leading
architectures **inverts** depending on whether those classes are counted.

### Cost against accuracy

| Model    | Best epoch | mAP@0.5   | mAP@0.5:0.95 | Training time |
|----------|-----------:|----------:|-------------:|--------------:|
| YOLOv8n  | 198        | 0.795     | 0.480        | 30 min        |
| YOLOv10s | 214        | **0.928** | 0.611        | **72 min**    |
| YOLOv12m | 240        | 0.909     | **0.625**    | 332 min       |

YOLOv12m costs 4.6× YOLOv10s and does not beat it at IoU 0.5. Its edge at
mAP@0.5:0.95 suggests the extra capacity buys tighter boxes on objects it
already finds, not more detections — which for a triage tool is the less
valuable of the two.

Single runs on one GPU. "Best" is a maximum over epochs on the validation split
and is optimistically biased; it is not a held-out estimate. That is the next
thing the harness fixes.

### Where the model actually fails

Inter-class confusion accounts for roughly **1%** of predictions. Essentially
every error is a missed detection or a background false positive:

| Class     | Recall | Support | 95% interval  |
|-----------|-------:|--------:|---------------|
| Impacted  | 1.00   | 18/18   | [0.82, 1.00]  |
| Fractured | 0.89   | 8/9     | [0.56, 0.98]  |
| Caries    | 0.79   | 11/14   | [0.52, 0.92]  |
| Healthy   | 0.64   | 43/67   | [0.52, 0.75]  |
| BDC/BDR   | 1.00   | 3/3     | [0.44, 1.00]  |
| Infection | 0.50   | 2/4     | [0.15, 0.85]  |

Half of all periapical infections are never proposed at all. 82% of background
false positives are labelled `Healthy`. The classification head is not the
bottleneck — given a proposal, the model names it correctly. Proposal recall is.

---

## The measurement problem

The dataset is 231 annotated radiographs, offline-augmented 3× on the training
side only:

| Split | Images | Source radiographs | Boxes | Caries | Infection | Impacted | BDC/BDR | Fractured | Healthy |
|-------|-------:|-------------------:|------:|-------:|----------:|---------:|--------:|----------:|--------:|
| train | 558    | 186                | 5642  | 853    | 172       | 729      | 69      | 526       | 3293    |
| val   | 23     | 23                 | 115   | 14     | 4         | 18       | 3       | 9         | 67      |
| test  | 23     | 23                 | 102   | 26     | 5         | 12       | **0**   | 5         | 54      |

- **No leakage.** Grouping every file by the source radiograph in its filename,
  the intersection between every pair of splits is empty. Verified, not assumed.
- **Class imbalance is 48:1** in training (Healthy 3293 vs BDC/BDR 69).
- **`BDC/BDR` has zero instances in test.** The held-out split cannot score one
  of the six classes at all.
- **Three classes have fewer than 10 validation instances.** An AP of 0.995 on
  three boxes carries a 95% interval on recall of [0.44, 1.00].

The fix is not a better backbone. It is a protocol the data can support: pooled
**grouped, rarity-aware 5-fold cross-validation** over all 231 radiographs.
Folds are grouped by source image so augmented copies never straddle a boundary,
and assigned rarest-class-first so scarce classes stay present everywhere. Result:

| | Current split | 5-fold CV |
|---|---:|---:|
| Radiographs used for evaluation | 23 | 231 |
| `BDC/BDR` evaluation instances | 3 | **72** (10–20 per fold) |
| Classes scorable on the held-out set | 5 of 6 | **6 of 6** |
| Trainings required | 1 | 5 |

---

## Reproducing everything

```bash
pip install -r requirements-research.txt

# ── dataset.py ── find it, audit it, split it
python dataset.py prepare --search-root . --also-single-copy
python dataset.py audit   --dataset-root <dataset root>
python dataset.py folds   --image-dirs <train/images> <valid/images> <test/images> --folds 5

# ── model.py ── train, ablate, aggregate
python model.py train     --name baseline --model yolov10s.pt --seeds 0 1 2 3 4
python model.py train     --name baseline_cv --model yolov10s.pt --cv-dir configs/cv
python model.py ablate    --dry-run          # what it would run, and what it costs
python model.py ablate    --seeds 0 1 2
python model.py aggregate --manifest results/ablation_manifest.json

# ── evaluate.py ── per-class metrics with intervals, and head-to-head
python evaluate.py per-class --weights runs/.../best.pt --images <test/images> --out results/v10s
python evaluate.py compare   --runs runs/detect/Yolo_8n_250 runs/detect/Yolo_10s_train \
                             runs/detect/Yolo_12m_250epochs \
                             --labels YOLOv8n YOLOv10s YOLOv12m

# ── evaluate_new.py ── why it is wrong, and where it looks
python evaluate_new.py errors  --cache results/v10s/predictions.npz --out results/v10s
python evaluate_new.py sweep   --cache results/v10s/predictions.npz
python evaluate_new.py explain --weights runs/.../best.pt --images <test/images> --faithfulness
```

Every command has `--help`. `dataset.py audit` is the one to run first on any new
data.

`make all` runs the audit, the folds, the comparison and the test suite. `make report` rebuilds the PDF.

---

## Design notes

A few choices that are not obvious:

**Inference runs once; every metric reads a cache.** A bootstrap confidence
interval needs the metric recomputed on hundreds of resamples. Re-running the
network each time is wasteful, so `predict.py` caches per-image predictions and
ground truth to an `.npz`, and everything downstream — per-class AP, intervals,
error decomposition, threshold sweeps, model-vs-model comparison — is NumPy over
that cache. One artefact, fully traceable, and a threshold sweep costs nothing.

**AP is implemented here, not read off the framework**, for the same reason. It
follows the COCO convention and is pinned by unit tests against Ultralytics'
own `ap_per_class` to within 0.02 AP on randomised inputs
(`tests/test_metrics.py`, 17 tests).

**Precision and recall are reported at the deployed threshold (0.25)**, not as
curve summaries. AP says nothing about behaviour at the operating point a
clinician actually sees.

**The bootstrap resamples images, not boxes.** Boxes within one radiograph are
correlated; a box-level bootstrap would understate the interval.

**Classes with no ground truth return NaN, never 0.0.** A silent zero in a macro
average is the easiest way to publish a wrong number.

**Ablations are deltas from one baseline**, declared in
[`configs/ablation.yaml`](configs/ablation.yaml) with a stated hypothesis each,
so any difference in the result is attributable to the delta. Every variant
reports mean ± std over seeds with Welch's *t* and Cohen's *d*, so "moved the
needle" and "within noise" are verdicts rather than adjectives.

**Saliency maps are scored, not just shown.** Pointing-game energy (fraction of
saliency mass inside the predicted boxes, against the box-area chance rate) and
deletion AUC against a random-order baseline. A map that scores at chance is
decoration.

---

## Layout

```
dataset.py             prepare | audit | folds
model.py               train | ablate | aggregate
evaluate.py            per-class | compare
evaluate_new.py        errors | sweep | explain
app.py                 Streamlit viewer
.streamlit/config.toml theme and upload limits

src/dentalscan/        the library those four CLIs share
  constants.py         classes, validated palette, canonical paths
  data.py              dataset audit, leakage check, grouped CV splits
  metrics.py           AP, per-class metrics, bootstrap and paired bootstrap
  predict.py           inference cache: predict once, analyse many times
  error_analysis.py    TIDE-style decomposition, size strata, threshold sweep
  explain.py           Eigen-CAM, Grad-CAM, pointing energy, deletion AUC
  train.py             seeded training with recorded provenance
  aggregate.py         mean ± std, Welch's t, Cohen's d, verdicts
  report.py            Markdown and LaTeX table rendering
  viz.py               figures

configs/
  data.yaml            generated by `dataset.py prepare`
  ablation.yaml        the ablation grid, as deltas from the baseline
  cv/                  generated fold configs and image lists
tests/                 known-answer tests, cross-checked against Ultralytics
report/                LaTeX source and the compiled PDF
results/               generated metrics, figures, audit
```

---

## Classes

| # | Class     | Description |
|---|-----------|-------------|
| 0 | Caries    | Carious lesion — radiolucency in enamel or dentine |
| 1 | Infection | Periapical or periodontal infection |
| 2 | Impacted  | Tooth that failed to erupt into normal occlusion |
| 3 | BDC/BDR   | Bone defect, coronal or root region |
| 4 | Fractured | Crown or root fracture |
| 5 | Healthy   | Normal tooth region, no visible pathology |

`Healthy` is a background label rather than a condition, which makes the
six-class mean a mixture of two different tasks. See the
[model card](MODEL_CARD.md).

---

## The viewer

```bash
pip install -r requirements.txt
streamlit run app.py
```

One centred column, no rail, no panels inside panels: the radiograph *is* the
page and everything else is a quiet line of controls around it.

**The image is the hero.** Scroll to zoom at the cursor, drag to pan,
double-click to fit, `1:1` for actual pixels. The frame sizes itself to the
study's aspect ratio rather than to a fixed number. The floating toolbar fades
in on hover and stays out of the way otherwise; its `Overlay` slider crossfades
between annotated and original, and `Compare` blanks the overlay while held —
the fastest way to check a box sits on a real feature. A 30-pixel carious lesion
on a 1935-pixel panoramic is unreadable at fit-to-width, so this is not
decoration.

**Progressive disclosure.** Condition filters are pills under the title; the
confidence threshold is a single slider directly beneath the image, where the
effect of moving it is visible without looking anywhere else. Everything
else — box labels, interior wash, overlay strength, NMS IoU, inference size —
lives behind `Settings`. Export lives behind `Export`. Saliency is a toggle that
swaps the viewport contents in place rather than a second tab.

**One forward pass.** Inference runs once per image at confidence 0.01. The
threshold slider and the condition pills filter that cached result, so they
respond instantly and every displayed number comes from a single pass. Changing
NMS IoU or inference size does re-run the model, and the UI says which is which.

**`Healthy` is grey and off by default.** It is a background label, not a
finding, and 58% of all boxes in the training data; drawn like pathology it
buries the pathology.

**Measured recall travels with every detection.** Each finding carries its
class's validation recall and support, badged ⚠ when the class has fewer than
ten instances and is effectively unmeasured, ↓ when recall is below 0.70. A
model that misses half of all infections should not present an infection the
same way it presents an impacted tooth.

**Export.** Annotated PNG at full resolution, findings CSV, and a JSON record
that names the weights and the image by SHA-256 and carries every threshold in
force — so a finding can be traced back to exactly what produced it.

The app looks for `runs/detect/Yolo_10s_train/weights/best.pt`. Without it, it
loads the base COCO model and says so in a red banner — those detections are
meaningless for dental use.

Box colours are the five-hue pathology palette in `constants.py`, chosen by
search inside the OKLCH lightness band for a dark surface and verified against a
six-check palette validator (lightness band, chroma floor, all-pairs
colour-vision-deficient separation, all-pairs normal-vision separation, contrast
against the surface). Class identity is never carried by colour alone: every box
is labelled, labels stack across rows with leader lines rather than overwriting
each other, and every findings row names the class in text.

---

## Data

[Dental OPG X-Ray Dataset](https://data.mendeley.com/datasets/c4hhrkxytw/4),
Mendeley Data. 231 annotated panoramic radiographs, six classes.

## Not a medical device

Research artefact and demonstration only. Not validated for clinical use, not
calibrated, trained on a single-source dataset with one annotator and no
inter-rater agreement statistics. Do not use it to make decisions about anyone's
teeth.
