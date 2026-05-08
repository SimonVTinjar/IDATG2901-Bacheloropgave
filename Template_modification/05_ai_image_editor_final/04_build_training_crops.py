from pathlib import Path
from PIL import Image, ImageDraw
import json


# -------------------------------------------------
# KONFIGURASJON
# -------------------------------------------------

BASE_DIR = Path(__file__).parent


DATA_DIR = BASE_DIR / "data"
ORIGINAL_DIR = DATA_DIR / "images"
TEMPLATE_DIR = DATA_DIR / "templates"
ANNOTATION_DIR = DATA_DIR / "labels"

OUTPUT_DIR = BASE_DIR / "training_data"
INPUT_DIR = OUTPUT_DIR / "input"
TARGET_DIR = OUTPUT_DIR / "target"
MASK_DIR = OUTPUT_DIR / "mask"
TEXT_GUIDE_DIR = OUTPUT_DIR / "text_guide"

METADATA_PATH = OUTPUT_DIR / "metadata.jsonl"

CROP_MARGIN = 40
MASK_PADDING = 4



def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_image(folder, stem):
    for ext in [".png", ".jpg", ".jpeg"]:
        path = folder / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def clamp(v, low, high):
    return max(low, min(high, v))


def expand_bbox(bbox, image_size, margin):
    x1, y1, x2, y2 = bbox
    w, h = image_size

    return [
        clamp(x1 - margin, 0, w),
        clamp(y1 - margin, 0, h),
        clamp(x2 + margin, 0, w),
        clamp(y2 + margin, 0, h),
    ]


def make_crop_mask(crop_size, bbox, crop_box, padding=4):
    crop_x1, crop_y1, _, _ = crop_box
    x1, y1, x2, y2 = bbox

    lx1 = x1 - crop_x1
    ly1 = y1 - crop_y1
    lx2 = x2 - crop_x1
    ly2 = y2 - crop_y1

    cw, ch = crop_size
    mask = Image.new("L", (cw, ch), 0)
    draw = ImageDraw.Draw(mask)

    lx1 = clamp(lx1 - padding, 0, cw)
    ly1 = clamp(ly1 - padding, 0, ch)
    lx2 = clamp(lx2 + padding, 0, cw)
    ly2 = clamp(ly2 + padding, 0, ch)

    draw.rectangle([lx1, ly1, lx2, ly2], fill=255)
    return mask


def create_text_guide(crop_size, bbox, crop_box, text):
    """
    Lager et ekstra bilde som viser ønsket tekst grovt inni boksen.
    Dette er IKKE final output. Det er bare et kontrollsignal til treningsmodellen.

    Modellen lærer:
      input crop + mask + text guide -> target crop
    """
    crop_x1, crop_y1, _, _ = crop_box
    x1, y1, x2, y2 = bbox

    lx1 = x1 - crop_x1
    ly1 = y1 - crop_y1
    lx2 = x2 - crop_x1
    ly2 = y2 - crop_y1

    guide = Image.new("L", crop_size, 0)
    draw = ImageDraw.Draw(guide)

    if not text:
        return guide

    # Bruk default-font for guide.
    # Poenget er bare å gi modellen tegnplassering, ikke riktig stil.
    font = ImageDraw.ImageDraw(guide).getfont()

    # Enkel sentrering med default font.
    bbox_text = draw.textbbox((0, 0), text, font=font)
    tw = bbox_text[2] - bbox_text[0]
    th = bbox_text[3] - bbox_text[1]

    bw = max(1, lx2 - lx1)
    bh = max(1, ly2 - ly1)

    tx = lx1 + max(0, (bw - tw) // 2)
    ty = ly1 + max(0, (bh - th) // 2)

    draw.text((tx, ty), text, fill=255, font=font)
    return guide


def default_value_for_label(label):
    # Brukes bare hvis JSON ikke har "value".
    examples = {
        "serial": "AB12345678",
        "serial_2": "A123456",
        "code": "12",
        "top_left_code": "A123",
        "top_right_code": "1",
        "top_rigth_code": "1",
        "series": "A",
        "signature_1": "O. Nordmann",
        "signature_2": "K. Hansen",
    }
    return examples.get(label, "")


# -------------------------------------------------
# HOVEDPROGRAM
# -------------------------------------------------

def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_GUIDE_DIR.mkdir(parents=True, exist_ok=True)

    json_files = sorted(ANNOTATION_DIR.glob("*.json"))

    if not json_files:
        print("Fant ingen JSON-filer i:", ANNOTATION_DIR)
        return

    metadata = []

    for json_path in json_files:
        stem = json_path.stem

        original_path = find_image(ORIGINAL_DIR, stem)
        template_path = find_image(TEMPLATE_DIR, stem)

        if original_path is None:
            print(f"[HOPPER OVER] Mangler original for {stem}")
            continue

        if template_path is None:
            print(f"[HOPPER OVER] Mangler template for {stem}")
            continue

        ann = load_json(json_path)

        w = ann["image_size"]["width"]
        h = ann["image_size"]["height"]
        image_size = (w, h)

        original = Image.open(original_path).convert("RGB")
        template = Image.open(template_path).convert("RGB")

        if original.size != image_size:
            original = original.resize(image_size)

        if template.size != image_size:
            template = template.resize(image_size)

        boxes = ann.get("boxes", [])

        for i, box in enumerate(boxes):
            label = box["label"]
            bbox = box["bbox"]
            value = box.get("value", default_value_for_label(label))

            crop_box = expand_bbox(bbox, image_size, CROP_MARGIN)

            input_crop = template.crop(crop_box)
            target_crop = original.crop(crop_box)
            mask_crop = make_crop_mask(input_crop.size, bbox, crop_box, MASK_PADDING)
            text_guide = create_text_guide(input_crop.size, bbox, crop_box, value)

            sample_id = f"{stem}_{i:03d}_{label}"

            input_path = INPUT_DIR / f"{sample_id}_input.png"
            target_path = TARGET_DIR / f"{sample_id}_target.png"
            mask_path = MASK_DIR / f"{sample_id}_mask.png"
            guide_path = TEXT_GUIDE_DIR / f"{sample_id}_textguide.png"

            input_crop.save(input_path)
            target_crop.save(target_path)
            mask_crop.save(mask_path)
            text_guide.save(guide_path)

            metadata.append({
                "sample_id": sample_id,
                "label": label,
                "value": value,
                "bbox": bbox,
                "crop_box": crop_box,
                "input": str(input_path),
                "target": str(target_path),
                "mask": str(mask_path),
                "text_guide": str(guide_path),
            })

        print(f"[OK] {stem}: {len(boxes)} bokser")

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        for item in metadata:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print()
    print("Ferdig.")
    print("Antall samples:", len(metadata))
    print("Metadata:", METADATA_PATH)


if __name__ == "__main__":
    main()
