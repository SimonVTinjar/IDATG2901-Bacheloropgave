import json
from pathlib import Path
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
from torchvision import transforms, models

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR.parent / "models" / "classifiers" / "box_classifier_filled_missing_v1.pth"
JSON_PATH = BASE_DIR / "image.json"
TEMPLATE_PATH = BASE_DIR / "template.png"
OUTPUT_PATH = BASE_DIR / "checked_template.png"

IMAGE_SIZE = (128, 128)


def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    classes = checkpoint["classes"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, classes


def classify_crop(model, classes, crop):
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])

    x = transform(crop).unsqueeze(0)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = probs.argmax().item()

    return classes[pred_idx], probs[pred_idx].item()


def main():
    model, classes = load_model()

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    img = Image.open(TEMPLATE_PATH).convert("RGB")
    draw = ImageDraw.Draw(img)

    print("\nResultat:\n")

    for box in data["boxes"]:
        label = box["label"]
        x1, y1, x2, y2 = box["bbox"]

        crop = img.crop((x1, y1, x2, y2))
        pred, confidence = classify_crop(model, classes, crop)

        color = "red" if pred == "missing" else "green"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1, max(0, y1 - 12)), f"{label}: {pred}", fill=color)

        print(f"{label}: {pred} ({confidence:.2%})")

    img.save(OUTPUT_PATH)
    print("\nLagret:", OUTPUT_PATH)


if __name__ == "__main__":
    main()