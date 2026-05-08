from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import cv2
import json
import numpy as np


# -------------------------------------------------
# KONFIGURASJON
# -------------------------------------------------

BASE_DIR = Path(__file__).parent

ORIGINAL_PATH = BASE_DIR / "image.png"
TEMPLATE_PATH = BASE_DIR / "Prob Done v2.png"

OUTPUT_JSON = BASE_DIR / "template_named_boxes.json"
OUTPUT_PREVIEW = BASE_DIR / "template_numbered_preview.png"
OUTPUT_DEBUG_MASK = BASE_DIR / "debug_grouped_mask.png"

THRESHOLD = 1
MIN_AREA = 10000

HORIZONTAL_JOIN = 80
VERTICAL_JOIN = 10

DISPLAY_MAX_WIDTH = 1400
DISPLAY_MAX_HEIGHT = 800

LABELS = [
    "serial",
    "serial_2",
    "code",
    "top_left_code",
    "top_right_code",
    "series",
    "signature_1",
    "signature_2",
    "noise"
]


# -------------------------------------------------
# FINN BOKSER MED DIFF
# -------------------------------------------------

def detect_boxes(original, template):
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

    cv2.imwrite(str(OUTPUT_DEBUG_MASK), mask)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    boxes = []

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        if area < MIN_AREA:
            continue

        if w < 10 or h < 5:
            continue

        boxes.append([int(x), int(y), int(x + w), int(y + h)])

    # Sorter ovenfra og ned, venstre til høyre
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

    return boxes


# -------------------------------------------------
# VISNING
# -------------------------------------------------

def scale_for_display(image):
    h, w = image.shape[:2]

    scale_w = DISPLAY_MAX_WIDTH / w
    scale_h = DISPLAY_MAX_HEIGHT / h
    scale = min(scale_w, scale_h, 1.0)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h))
    return resized, scale


def show_current_box(template, boxes, current_index):
    img = np.array(template.copy())
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box

        if idx == current_index:
            color = (0, 255, 0)   # grønn
            thickness = 6
        else:
            color = (0, 0, 255)   # rød
            thickness = 3

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            img,
            str(idx + 1),
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            3
        )

    display, scale = scale_for_display(img)

    cv2.imshow("Template - grønn boks er aktiv", display)
    cv2.waitKey(1)


def show_crop(template, box, padding=80):
    w, h = template.size
    x1, y1, x2, y2 = box

    x1p = max(0, x1 - padding)
    y1p = max(0, y1 - padding)
    x2p = min(w, x2 + padding)
    y2p = min(h, y2 + padding)

    crop = template.crop((x1p, y1p, x2p, y2p))
    crop_np = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)

    display, scale = scale_for_display(crop_np)

    cv2.imshow("Crop av aktiv boks", display)
    cv2.waitKey(1)


def save_numbered_preview(template, boxes):
    preview = template.copy()
    draw = ImageDraw.Draw(preview)

    try:
        font = ImageFont.truetype("arial.ttf", 60)
    except Exception:
        font = ImageFont.load_default()

    for idx, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = box

        draw.rectangle([x1, y1, x2, y2], outline="red", width=8)

        text = str(idx)
        tb = draw.textbbox((0, 0), text, font=font)

        tw = tb[2] - tb[0]
        th = tb[3] - tb[1]

        draw.rectangle(
            [x1, max(0, y1 - th - 20), x1 + tw + 30, y1],
            fill="red"
        )
        draw.text(
            (x1 + 15, max(0, y1 - th - 15)),
            text,
            fill="white",
            font=font
        )

    preview.save(OUTPUT_PREVIEW)


# -------------------------------------------------
# MANUELL LABELING
# -------------------------------------------------

def print_menu():
    print("\nVelg label:")
    for i, label in enumerate(LABELS, start=1):
        print(f"{i}. {label}")

    print("\nAndre valg:")
    print("s = skip denne boksen")
    print("q = avslutt og lagre")
    print()


def ask_label():
    while True:
        print_menu()
        choice = input("Skriv nummer eller label: ").strip()

        if choice.lower() == "q":
            return "QUIT"

        if choice.lower() == "s":
            return None

        if choice.isdigit():
            idx = int(choice)

            if 1 <= idx <= len(LABELS):
                label = LABELS[idx - 1]
            else:
                print("Ugyldig nummer.")
                continue
        else:
            label = choice

        if label not in LABELS:
            print("Ukjent label.")
            continue

        if label == "noise":
            return None

        return label


def manual_label(template, boxes):
    named_boxes = []

    for index, box in enumerate(boxes):
        print("\n" + "=" * 50)
        print(f"Boks {index + 1} av {len(boxes)}")
        print("bbox:", box)

        show_current_box(template, boxes, index)
        show_crop(template, box)

        label = ask_label()

        if label == "QUIT":
            break

        if label is None:
            print("Hoppet over.")
            continue

        value = input("Skriv value/tekst for boksen, eller Enter for tom: ").strip()

        item = {
            "label": label,
            "bbox": box
        }

        if value:
            item["value"] = value

        named_boxes.append(item)

        print("Lagret:", item)

    return named_boxes


# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main():
    if not ORIGINAL_PATH.exists():
        raise FileNotFoundError(f"Fant ikke originalbildet: {ORIGINAL_PATH}")

    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Fant ikke templatebildet: {TEMPLATE_PATH}")

    original = Image.open(ORIGINAL_PATH).convert("RGB")
    template = Image.open(TEMPLATE_PATH).convert("RGB")

    if template.size != original.size:
        print(f"Resizer template fra {template.size} til {original.size}")
        template = template.resize(original.size)

    print("Finner bokser...")
    boxes = detect_boxes(original, template)

    print(f"Fant {len(boxes)} kandidatbokser.")

    if not boxes:
        print("Ingen bokser funnet.")
        return

    save_numbered_preview(template, boxes)

    print("Lagret preview:", OUTPUT_PREVIEW)
    print("Et vindu åpnes nå. Den grønne boksen er boksen du navngir.")

    named_boxes = manual_label(template, boxes)

    output = {
        "image": TEMPLATE_PATH.name,
        "image_path": str(TEMPLATE_PATH),
        "image_size": {
            "width": template.size[0],
            "height": template.size[1]
        },
        "boxes": named_boxes
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    cv2.destroyAllWindows()

    print("\nFerdig.")
    print("Lagret JSON:", OUTPUT_JSON)
    print("Antall bokser lagret:", len(named_boxes))


if __name__ == "__main__":
    main()