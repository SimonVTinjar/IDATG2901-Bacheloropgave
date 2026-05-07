from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
import random
import string
import json


# -------------------------------------------------
# KONFIGURASJON
# -------------------------------------------------

BASE_DIR = Path(__file__).parent

ORIGINAL_PATH = BASE_DIR / "image.png"
TEMPLATE_PATH = BASE_DIR / "Prob Done v2.png"

MODEL_PATH = BASE_DIR / "box_type_position_missing_classifier.pth"

TEMPLATE_JSON_OUTPUT = BASE_DIR / "template_boxes.json"
TEMPLATE_MASK_OUTPUT = BASE_DIR / "template_mask.png"
DEBUG_GROUPED_MASK_OUTPUT = BASE_DIR / "debug_grouped_mask.png"
DEBUG_BOXES_OUTPUT = BASE_DIR / "debug_template_boxes.png"

# Viktig:
# IMAGE_SIZE brukes bare for crop transform inn i ResNet.
# Selve bilde-størrelsen hentes fra original.size.
IMAGE_SIZE = (593, 1411)

THRESHOLD = 1
MIN_AREA = 10000

HORIZONTAL_JOIN = 80
VERTICAL_JOIN = 10

MASK_PADDING = 6


# -------------------------------------------------
# MODELL FOR BOKS-KLASSIFISERING
# -------------------------------------------------

class ImagePositionClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        backbone = models.resnet18(weights=None)
        self.image_features = nn.Sequential(*list(backbone.children())[:-1])
        image_dim = backbone.fc.in_features

        self.position_net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(image_dim + 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, image, position):
        img_feat = self.image_features(image)
        img_feat = torch.flatten(img_feat, 1)

        pos_feat = self.position_net(position)

        combined = torch.cat([img_feat, pos_feat], dim=1)
        return self.classifier(combined)


# -------------------------------------------------
# GENERER TESTVERDIER
# -------------------------------------------------

def generate_by_type(box_type):
    if box_type == "serial":
        return (
            "".join(random.choices(string.ascii_uppercase, k=2))
            + "".join(random.choices(string.digits, k=8))
        )

    if box_type == "serial_2":
        return (
            "".join(random.choices(string.ascii_uppercase, k=1))
            + "".join(random.choices(string.digits, k=6))
        )

    if box_type == "code":
        return "".join(random.choices(string.digits, k=2))

    if box_type == "top_left_code":
        return (
            "".join(random.choices(string.ascii_uppercase, k=1))
            + "".join(random.choices(string.digits, k=3))
        )

    if box_type in ["top_right_code", "top_rigth_code"]:
        return "".join(random.choices(string.digits, k=1))

    if box_type == "series":
        return random.choice(["A", "B", "C", "D"])

    if box_type == "signature_1":
        return "O. Nordmann"

    if box_type == "signature_2":
        return "K. Hansen"

    return ""


# -------------------------------------------------
# HJELPEFUNKSJONER
# -------------------------------------------------

def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    classes = checkpoint["classes"]

    model = ImagePositionClassifier(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print("Modellklasser:", classes)
    return model, classes


def make_position_tensor(box, image_size):
    x1, y1, x2, y2 = box
    img_w, img_h = image_size

    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h

    return torch.tensor([[cx, cy, bw, bh]], dtype=torch.float32)


def classify_crop_with_position(model, classes, crop, box, image_size):
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])

    x = transform(crop).unsqueeze(0)
    pos = make_position_tensor(box, image_size)

    with torch.no_grad():
        output = model(x, pos)
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = probs.argmax().item()

    return classes[pred_idx], probs[pred_idx].item()


def make_mask_from_regions(image_size, regions, padding=6):
    """
    Lager mask:
    svart = behold bildet
    hvit = AI får redigere dette området
    """
    w, h = image_size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    for region in regions:
        x1, y1, x2, y2 = region["bbox"]

        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        draw.rectangle([x1, y1, x2, y2], fill=255)

    return mask


def find_candidate_boxes(original, template):
    """
    Finner forskjeller mellom original og template.
    Returnerer bbox-listen som [x1, y1, x2, y2].
    """
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

    cv2.imwrite(str(DEBUG_GROUPED_MASK_OUTPUT), mask)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    candidate_boxes = []

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        if area < MIN_AREA:
            continue

        if w < 10 or h < 5:
            continue

        candidate_boxes.append([int(x), int(y), int(x + w), int(y + h)])

    return candidate_boxes


# -------------------------------------------------
# HOVEDPROGRAM
# -------------------------------------------------

def main():
    if not ORIGINAL_PATH.exists():
        raise FileNotFoundError(f"Fant ikke originalbildet: {ORIGINAL_PATH}")

    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Fant ikke templatebildet: {TEMPLATE_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Fant ikke modellen: {MODEL_PATH}")

    model, classes = load_model()

    original = Image.open(ORIGINAL_PATH).convert("RGB")
    template = Image.open(TEMPLATE_PATH).convert("RGB")

    # Template må ha samme størrelse som original
    if template.size != original.size:
        print(f"Resizer template fra {template.size} til {original.size}")
        template = template.resize(original.size)

    image_size = original.size

    candidate_boxes = find_candidate_boxes(original, template)

    print(f"Fant {len(candidate_boxes)} kandidatbokser fra diff.")

    template_json = {
        "image": str(TEMPLATE_PATH.name),
        "image_size": {
            "width": image_size[0],
            "height": image_size[1]
        },
        "boxes": []
    }

    debug_img = template.copy()
    debug_draw = ImageDraw.Draw(debug_img)

    used_boxes = 0

    for i, box in enumerate(candidate_boxes, start=1):
        x1, y1, x2, y2 = box

        crop = template.crop((x1, y1, x2, y2))

        box_type, conf = classify_crop_with_position(
            model=model,
            classes=classes,
            crop=crop,
            box=box,
            image_size=image_size
        )

        print(f"Boks {i}: {box_type} ({conf:.2%}) bbox={box}")

        if box_type == "noise":
            continue

        value = generate_by_type(box_type)

        if not value:
            continue

        used_boxes += 1

        template_json["boxes"].append({
            "label": box_type,
            "bbox": box,
            "value": value,
            "confidence": float(conf)
        })

        debug_draw.rectangle(box, outline="red", width=2)
        debug_draw.text((x1, max(0, y1 - 12)), box_type, fill="red")

        print(f"Skal redigere {box_type} til: {value}")

    with open(TEMPLATE_JSON_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(template_json, f, indent=2, ensure_ascii=False)

    edit_mask = make_mask_from_regions(
        image_size=image_size,
        regions=template_json["boxes"],
        padding=MASK_PADDING
    )

    edit_mask.save(TEMPLATE_MASK_OUTPUT)
    debug_img.save(DEBUG_BOXES_OUTPUT)

    print()
    print("Ferdig.")
    print("Brukbare bokser:", used_boxes)
    print("Lagret JSON:", TEMPLATE_JSON_OUTPUT)
    print("Lagret mask:", TEMPLATE_MASK_OUTPUT)
    print("Lagret debug-bilde:", DEBUG_BOXES_OUTPUT)
    print("Lagret grouped mask:", DEBUG_GROUPED_MASK_OUTPUT)


if __name__ == "__main__":
    main()
