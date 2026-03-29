# evaluate.py

from ultralytics import YOLO
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import os
from PIL import Image
import cv2
from torchvision import models

# Paths
YOLO_MODEL_PATH = "./runs/detect/10s/weights/best.pt"
CNN_MODEL_PATH = "./cnn_best.pth"
DATA_YAML_PATH = "data.yaml"
TEST_IMAGE_DIR = "C:/Dev/Python/Yolo/dataset/images/test"

CLASS_NAMES = ["Caries", "Infection", "Impacted", "BDC/BDR", "Fractured", "Healthy"]

# Load CNN model (ResNet18)
def load_cnn_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 6)
    model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

# Preprocess for CNN
def transform_image(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(img).unsqueeze(0)

# YOLO Evaluation
def evaluate_yolo(model_path, data_yaml):
    model = YOLO(model_path)
    results = model.val(data=data_yaml, save_json=True)
    print("\n📊 YOLO EVALUATION METRICS")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall: {results.box.mr:.4f}")
    print(f"mAP@0.5: {results.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {results.box.map:.4f}")
    return results

# CNN Evaluation using YOLO detections
def evaluate_cnn_on_yolo_detections(yolo_model, cnn_model):
    correct = 0
    total = 0

    for fname in os.listdir(TEST_IMAGE_DIR):
        if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        path = os.path.join(TEST_IMAGE_DIR, fname)
        image = cv2.imread(path)
        results = yolo_model(image)

        for i, box in enumerate(results[0].boxes.xyxy):
            cls_yolo = int(results[0].boxes.cls[i].item())
            x1, y1, x2, y2 = map(int, box.tolist())
            cropped = image[y1:y2, x1:x2]

            if cropped.size == 0:
                continue

            pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            input_tensor = transform_image(pil_crop)

            with torch.no_grad():
                output = cnn_model(input_tensor)
                cls_cnn = torch.argmax(output, dim=1).item()

            if cls_cnn == cls_yolo:
                correct += 1
            total += 1

    acc = (correct / total) * 100 if total else 0
    print(f"\n🧠 CNN ACCURACY ON YOLO DETECTIONS: {acc:.2f}% ({correct}/{total} correct)")
    return acc

# Run full pipeline
if __name__ == "__main__":
    yolo_model = YOLO(YOLO_MODEL_PATH)
    cnn_model = load_cnn_model()

    evaluate_yolo(YOLO_MODEL_PATH, DATA_YAML_PATH)
    evaluate_cnn_on_yolo_detections(yolo_model, cnn_model)
