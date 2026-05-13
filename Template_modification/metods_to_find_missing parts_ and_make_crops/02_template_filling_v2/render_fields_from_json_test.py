import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).parent

JSON_PATH = BASE_DIR / "s-l12004_9.json"
TARGET_IMAGE_PATH = BASE_DIR / "template.png" 
OUTPUT_PATH = BASE_DIR / "output.png"
FONT_PATH = "arial.ttf"


def load_font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()


def scale_box(box, src_w, src_h, dst_w, dst_h):
    x1, y1, x2, y2 = box

    scale_x = dst_w / src_w
    scale_y = dst_h / src_h

    return [
        int(x1 * scale_x),
        int(y1 * scale_y),
        int(x2 * scale_x),
        int(y2 * scale_y),
    ]


def fit_font(draw, text, box, max_size=60):
    x1, y1, x2, y2 = box
    max_w = max(1, x2 - x1)
    max_h = max(1, y2 - y1)

    for size in range(max_size, 10, -1):
        font = load_font(size)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        if tw <= max_w and th <= max_h:
            return font

    return load_font(10)


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
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    src_w = data["image_size"]["width"]
    src_h = data["image_size"]["height"]
    boxes = data["boxes"]

    img = Image.open(TARGET_IMAGE_PATH).convert("RGB")
    draw = ImageDraw.Draw(img)

    dst_w, dst_h = img.size

    print(f"Original størrelse fra JSON: {src_w}x{src_h}")
    print(f"Ny template størrelse: {dst_w}x{dst_h}")

    print("\nFant disse boksene:\n")
    for i, box in enumerate(boxes, start=1):
        scaled_bbox = scale_box(box["bbox"], src_w, src_h, dst_w, dst_h)
        print(f"{i}. {box['label']} original={box['bbox']} skalert={scaled_bbox}")

    print("\n---\n")

    for i, box in enumerate(boxes, start=1):
        label = box["label"]
        scaled_bbox = scale_box(box["bbox"], src_w, src_h, dst_w, dst_h)

        text = input(f"Hva vil du ha i '{label}' (boks {i})? ").strip()
        if text:
            draw_text(draw, text, scaled_bbox, fill="black")

    img.save(OUTPUT_PATH)
    print(f"\nLagret: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()