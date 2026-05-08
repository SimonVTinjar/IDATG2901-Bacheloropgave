from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json
import random
import string

BASE_DIR = Path(__file__).parent

FIELDS_PATH = BASE_DIR / "fields.json"
TEMPLATE_PATH = "Prob Done v2.png"
OUTPUT_PATH = BASE_DIR / "final_output.png"
FONT_PATH = BASE_DIR / "arial.ttf"


def generate_by_type(label):
    if label == "serial":
        return "".join(random.choices(string.ascii_uppercase, k=2)) + "".join(random.choices(string.digits, k=8))

    if label == "serial_2":
        return "".join(random.choices(string.ascii_uppercase, k=1)) + "".join(random.choices(string.digits, k=6))

    if label == "code":
        return "".join(random.choices(string.digits, k=2))

    if label == "top_left_code":
        return "".join(random.choices(string.ascii_uppercase, k=1)) + "".join(random.choices(string.digits, k=3))

    if label == "top_right_code":
        return "".join(random.choices(string.digits, k=1))

    if label == "series":
        return random.choice(["A", "B", "C", "D"])

    if label == "signature_1":
        return "O. Nordmann"

    if label == "signature_2":
        return "K. Hansen"

    return ""


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

        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        if text_w <= max_w and text_h <= max_h:
            return font

    return load_font(8)


def draw_text_in_box(draw, text, box, fill="black"):
    font = fit_font(draw, text, box)

    x1, y1, x2, y2 = box
    bbox = draw.textbbox((0, 0), text, font=font)

    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = x1 + (x2 - x1 - text_w) // 2
    y = y1 + (y2 - y1 - text_h) // 2

    draw.text((x, y), text, fill=fill, font=font)


def scale_box(box, src_w, src_h, dst_w, dst_h):
    x1, y1, x2, y2 = box
    return [
        int(x1 * dst_w / src_w),
        int(y1 * dst_h / src_h),
        int(x2 * dst_w / src_w),
        int(y2 * dst_h / src_h)
    ]


def main():
    with open(FIELDS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    src_w = data["image_size"]["width"]
    src_h = data["image_size"]["height"]

    img = Image.open(TEMPLATE_PATH).convert("RGB")
    draw = ImageDraw.Draw(img)
    dst_w, dst_h = img.size

    for field in data["fields"]:
        label = field["label"]
        box = scale_box(field["bbox"], src_w, src_h, dst_w, dst_h)

        text = generate_by_type(label)

        if not text:
            continue

        print(label, "→", text)

        draw.rectangle(box, outline="red", width=2)
        draw_text_in_box(draw, text, box)

    img.save(OUTPUT_PATH)
    print("Lagret:", OUTPUT_PATH)


if __name__ == "__main__":
    main()