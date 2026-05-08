from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
import random
import string

BASE_DIR = Path(__file__).parent

ORIGINAL_PATH = "image.png"
TEMPLATE_PATH = "Prob Done v2.png"
MODEL_PATH = BASE_DIR.parent / "models" / "classifiers" / "box_type_classifier_v3.pth"
OUTPUT_PATH = BASE_DIR / "final_outputv2-1.png"
FONT_PATH = BASE_DIR / "arial.ttf"

THRESHOLD = 1
MIN_AREA = 10
IMAGE_SIZE = (500, 500)

HORIZONTAL_JOIN = 80
VERTICAL_JOIN = 8


def generate_by_type(box_type):
    if box_type == "serial":
        return "".join(random.choices(string.ascii_uppercase, k=2)) + "".join(random.choices(string.digits, k=8))

    if box_type == "serial_2":
        return "".join(random.choices(string.ascii_uppercase, k=1)) + "".join(random.choices(string.digits, k=6))

    if box_type == "code":
        return "".join(random.choices(string.digits, k=2))

    if box_type == "top_left_code":
        return "".join(random.choices(string.ascii_uppercase, k=1)) + "".join(random.choices(string.digits, k=3))

    if box_type in ["top_right_code", "top_rigth_code"]:
        return "".join(random.choices(string.digits, k=1))

    if box_type == "series":
        return random.choice(["A", "B", "C", "D"])

    if box_type == "signature_1":
        return "O. Nordmann"

    if box_type == "signature_2":
        return "K. Hansen"

    return ""


def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    classes = checkpoint["classes"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print("Modellklasser:", classes)
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


def load_font(size):
    try:
        return ImageFont.truetype(str(FONT_PATH), size)
    except Exception:
        return ImageFont.load_default()


def fit_font(draw, text, box, max_size=120):
    x1, y1, x2, y2 = box
    max_w = max(1, x2 - x1)
    max_h = max(1, y2 - y1)

    for size in range(max_size, 6, -1):
        font = load_font(size)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        if tw <= max_w and th <= max_h:
            return font

    return load_font(8)


def draw_text(draw, text, box, fill="black"):
    font = fit_font(draw, text, box)

    x1, y1, x2, y2 = box
    bbox = draw.textbbox((0, 0), text, font=font)

    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    x = x1 + (x2 - x1 - tw) // 2
    y = y1 + (y2 - y1 - th) // 2

    draw.text((x, y), text, fill=fill, font=font)


def main():
    model, classes = load_model()

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

    small_kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small_kernel)

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (HORIZONTAL_JOIN, VERTICAL_JOIN)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, horizontal_kernel)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    cv2.imwrite(str(BASE_DIR / "debug_grouped_mask.png"), mask)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    candidate_boxes = []

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        if area < MIN_AREA:
            continue

        if w < 10 or h < 5:
            continue

        candidate_boxes.append([int(x), int(y), int(x + w), int(y + h)])

    print(f"Fant {len(candidate_boxes)} grupperte kandidatbokser.")

    draw_img = template.copy()
    draw = ImageDraw.Draw(draw_img)

    used_boxes = 0

    for i, box in enumerate(candidate_boxes, start=1):
        x1, y1, x2, y2 = box

        crop = template.crop((x1, y1, x2, y2))
        box_type, conf = classify_crop(model, classes, crop)

        print(f"Boks {i}: {box_type} ({conf:.2%}) bbox={box}")

        if box_type == "noise":
            continue

        text = generate_by_type(box_type)

        if not text:
            continue

        used_boxes += 1
        print(f"Generert for {box_type}: {text}")

        draw.rectangle(box, outline="red", width=2)
        draw_text(draw, text, box)

    draw_img.save(OUTPUT_PATH)

    print(f"\nFylte inn {used_boxes} bokser.")
    print("Lagret:", OUTPUT_PATH)
    print("Debug mask:", BASE_DIR / "debug_grouped_mask.png")


if __name__ == "__main__":
    main()