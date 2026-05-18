from pathlib import Path
from PIL import Image
import json
import numpy as np
import cv2

BASE_DIR = Path(__file__).parent

ORIGINAL_PATH = BASE_DIR.parent / "image.png"
FIELDS_PATH = BASE_DIR / "fields.json"
OUTPUT_STYLE_PATH = BASE_DIR / "style_reference.json"


def rgb_to_hex(rgb):
    r, g, b = [int(v) for v in rgb]
    return f"#{r:02x}{g:02x}{b:02x}"


def estimate_text_color(crop):
    """
    Finner ca tekstfarge ved å se på de mørkeste pikslene.
    """
    arr = np.array(crop.convert("RGB"))

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # Velg mørke piksler
    mask = gray < 120

    if mask.sum() < 5:
        return "#000000"

    colors = arr[mask]

    avg = colors.mean(axis=0)
    return rgb_to_hex(avg)


def estimate_text_height(crop):
    """
    Estimerer teksthøyde basert på mørke piksler.
    """
    arr = np.array(crop.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    mask = gray < 140

    ys = np.where(mask)[0]

    if len(ys) == 0:
        return 12

    text_height = int(ys.max() - ys.min())

    return max(8, text_height)


def estimate_alignment(crop):
    """
    Prøver å se om tekst er venstre, center eller høyre.
    """
    arr = np.array(crop.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    mask = gray < 140
    xs = np.where(mask)[1]

    if len(xs) == 0:
        return "center"

    text_center = (xs.min() + xs.max()) / 2
    crop_center = crop.size[0] / 2

    diff = text_center - crop_center

    if diff < -crop.size[0] * 0.15:
        return "left"
    elif diff > crop.size[0] * 0.15:
        return "right"
    else:
        return "center"


def main():
    original = Image.open(ORIGINAL_PATH).convert("RGB")

    with open(FIELDS_PATH, "r", encoding="utf-8") as f:
        fields_data = json.load(f)

    styles = {
        "image": str(ORIGINAL_PATH),
        "fields": []
    }

    for field in fields_data["fields"]:
        label = field["label"]
        box = field["bbox"]

        x1, y1, x2, y2 = box
        crop = original.crop((x1, y1, x2, y2))

        color = estimate_text_color(crop)
        text_height = estimate_text_height(crop)
        align = estimate_alignment(crop)

        style = {
            "label": label,
            "bbox": box,
            "color": color,
            "font_size": int(text_height * 1.4),
            "align": align
        }

        styles["fields"].append(style)

        print(label, "→", style)

    with open(OUTPUT_STYLE_PATH, "w", encoding="utf-8") as f:
        json.dump(styles, f, indent=2, ensure_ascii=False)

    print("\nLagret:", OUTPUT_STYLE_PATH)


if __name__ == "__main__":
    main()