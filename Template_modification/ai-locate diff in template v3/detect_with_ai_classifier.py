from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models

BASE_DIR = Path(__file__).parent

ORIGINAL_PATH = "image.png"
TEMPLATE_PATH = "Prob Done v2.png"
MODEL_PATH = BASE_DIR / "box_classifier.pth"

OUTPUT_PATH = BASE_DIR / "ai_detected_missing.png"

THRESHOLD = 2
MIN_AREA = 100
IMAGE_SIZE = (128, 128)


def load_classifier():
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
        output = model(x)
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = probs.argmax().item()

    return classes[pred_idx], probs[pred_idx].item()


def main():
    model, classes = load_classifier()

    original = Image.open(ORIGINAL_PATH).convert("RGB")
    template = Image.open(TEMPLATE_PATH).convert("RGB")
    template = template.resize(original.size)

    orig_np = np.array(original)
    temp_np = np.array(template)

    orig_gray = cv2.cvtColor(orig_np, cv2.COLOR_RGB2GRAY)
    temp_gray = cv2.cvtColor(temp_np, cv2.COLOR_RGB2GRAY)

    orig_gray = cv2.GaussianBlur(orig_gray, (7, 7), 0)
    temp_gray = cv2.GaussianBlur(temp_gray, (7, 7), 0)

    diff = cv2.absdiff(orig_gray, temp_gray)

    _, mask = cv2.threshold(diff, THRESHOLD, 255, cv2.THRESH_BINARY)

    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    vis = temp_np.copy()

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        if area < MIN_AREA:
            continue

        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)

        crop = template.crop((x1, y1, x2, y2))
        pred, conf = classify_crop(model, classes, crop)

        print(f"Boks {i}: {pred} ({conf:.2%}) bbox={[x1, y1, x2, y2]}")

        if pred == "missing":
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.putText(
                vis,
                f"missing {conf:.0%}",
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

    Image.fromarray(vis).save(OUTPUT_PATH)
    print("Lagret:", OUTPUT_PATH)


if __name__ == "__main__":
    main()