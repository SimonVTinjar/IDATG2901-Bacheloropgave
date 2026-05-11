import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import random

BASE_DIR = Path(__file__).parent

JSON_PATH = BASE_DIR / "image.json"
IMAGE_PATH = BASE_DIR / "image.png"

OUT_FILLED = BASE_DIR / "dataset" / "filled"
OUT_MISSING = BASE_DIR / "dataset" / "missing"

OUT_FILLED.mkdir(parents=True, exist_ok=True)
OUT_MISSING.mkdir(parents=True, exist_ok=True)


def average_color(img):
    small = img.resize((1, 1))
    return small.getpixel((0, 0))


def expand_box(box, pad, width, height):
    x1, y1, x2, y2 = box
    return [
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(width, x2 + pad),
        min(height, y2 + pad)
    ]


def make_missing_version(img, box):
    """
    Lager en kopi av bildet der området i boksen er fjernet.
    """
    result = img.copy()
    draw = ImageDraw.Draw(result)

    w, h = img.size
    x1, y1, x2, y2 = box

    # Ta litt område rundt boksen for å finne ca bakgrunnsfarge
    bg_box = expand_box(box, 12, w, h)
    bg_crop = img.crop(bg_box).filter(ImageFilter.GaussianBlur(8))
    color = average_color(bg_crop)

    # Fyll boksen med bakgrunnsfarge
    draw.rectangle([x1, y1, x2, y2], fill=color)

    return result


def main():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    img = Image.open(IMAGE_PATH).convert("RGB")

    for i, item in enumerate(data["boxes"]):
        label = item["label"].replace(" ", "_")
        box = item["bbox"]

        x1, y1, x2, y2 = box

        # 1. Crop original boks = filled
        filled_crop = img.crop((x1, y1, x2, y2))
        filled_crop.save(OUT_FILLED / f"{i}_{label}_filled.png")

        # 2. Lag bilde der boksen er fjernet
        missing_img = make_missing_version(img, box)

        # 3. Crop samme boks fra fjernet-versjonen = missing
        missing_crop = missing_img.crop((x1, y1, x2, y2))
        missing_crop.save(OUT_MISSING / f"{i}_{label}_missing.png")

        print("Laget:", label)

    print("\nFerdig.")
    print("Filled:", OUT_FILLED)
    print("Missing:", OUT_MISSING)


if __name__ == "__main__":
    main()