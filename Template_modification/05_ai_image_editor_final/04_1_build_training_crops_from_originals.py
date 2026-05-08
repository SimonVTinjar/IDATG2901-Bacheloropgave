from pathlib import Path
from PIL import Image, ImageDraw
import json
import statistics


BASE_DIR = Path(__file__).parent

DATASET_DIR = BASE_DIR / "data"

IMAGE_DIR_CANDIDATES = [
    DATASET_DIR / "images",
    DATASET_DIR / "originals",
    DATASET_DIR
]

ANNOTATION_DIR_CANDIDATES = [
    DATASET_DIR / "labels",
    DATASET_DIR / "json",
    DATASET_DIR
]

OUTPUT_DIR = BASE_DIR / "training_data_from_originals"

INPUT_DIR = OUTPUT_DIR / "input"
TARGET_DIR = OUTPUT_DIR / "target"
MASK_DIR = OUTPUT_DIR / "mask"
TEXT_GUIDE_DIR = OUTPUT_DIR / "text_guide"
DEBUG_DIR = OUTPUT_DIR / "debug"

METADATA_PATH = OUTPUT_DIR / "metadata.jsonl"

VALID_EXTENSIONS = [".png", ".jpg", ".jpeg"]

CROP_MARGIN = 40
MASK_PADDING = 4

# "median" bruker farge fra området rundt boksen.
# "white" fyller boksen med hvitt.
BLANK_METHOD = "median"


def find_existing_folder(candidates, name):
    for folder in candidates:
        if folder.exists() and folder.is_dir():
            return folder

    raise FileNotFoundError(
        f"Fant ikke {name}. Sjekk at 'datta set' ligger ved siden av scriptet."
    )


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_image_by_stem(folder, stem):
    for ext in VALID_EXTENSIONS:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def find_image_from_annotation(annotation, image_folder, json_path):
    image_name = annotation.get("image")
    if image_name:
        p = image_folder / image_name
        if p.exists():
            return p

    p = find_image_by_stem(image_folder, json_path.stem)
    if p is not None:
        return p

    image_path = annotation.get("image_path")
    if image_path:
        p = Path(image_path)
        if p.exists():
            return p

    return None


def normalize_label(label):
    if label == "top_rigth_code":
        return "top_right_code"
    return label


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


def sample_background_color(img, bbox, pad=8):
    x1, y1, x2, y2 = bbox
    w, h = img.size

    pixels = []

    # over
    y_top1 = max(0, y1 - pad)
    y_top2 = y1
    if y_top2 > y_top1:
        region = img.crop((x1, y_top1, x2, y_top2))
        pixels.extend(list(region.getdata()))

    # under
    y_bot1 = y2
    y_bot2 = min(h, y2 + pad)
    if y_bot2 > y_bot1:
        region = img.crop((x1, y_bot1, x2, y_bot2))
        pixels.extend(list(region.getdata()))

    # venstre
    x_left1 = max(0, x1 - pad)
    x_left2 = x1
    if x_left2 > x_left1:
        region = img.crop((x_left1, y1, x_left2, y2))
        pixels.extend(list(region.getdata()))

    # høyre
    x_right1 = x2
    x_right2 = min(w, x2 + pad)
    if x_right2 > x_right1:
        region = img.crop((x_right1, y1, x_right2, y2))
        pixels.extend(list(region.getdata()))

    if not pixels:
        return (255, 255, 255)

    rs = [p[0] for p in pixels]
    gs = [p[1] for p in pixels]
    bs = [p[2] for p in pixels]

    return (
        int(statistics.median(rs)),
        int(statistics.median(gs)),
        int(statistics.median(bs)),
    )


def blank_box_in_image(img, bbox, method="median", padding=2):
    result = img.copy()
    draw = ImageDraw.Draw(result)

    x1, y1, x2, y2 = bbox
    w, h = img.size

    x1 = clamp(x1 - padding, 0, w)
    y1 = clamp(y1 - padding, 0, h)
    x2 = clamp(x2 + padding, 0, w)
    y2 = clamp(y2 + padding, 0, h)

    if method == "white":
        fill = (255, 255, 255)
    else:
        fill = sample_background_color(img, [x1, y1, x2, y2], pad=12)

    draw.rectangle([x1, y1, x2, y2], fill=fill)
    return result


def create_text_guide(crop_size, bbox, crop_box, text):
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

    font = ImageDraw.ImageDraw(guide).getfont()

    tb = draw.textbbox((0, 0), text, font=font)
    tw = tb[2] - tb[0]
    th = tb[3] - tb[1]

    bw = max(1, lx2 - lx1)
    bh = max(1, ly2 - ly1)

    tx = lx1 + max(0, (bw - tw) // 2)
    ty = ly1 + max(0, (bh - th) // 2)

    draw.text((tx, ty), text, fill=255, font=font)
    return guide


def default_value_for_label(label):
    examples = {
        "serial": "AB12345678",
        "serial_2": "Ab1234568",
        "code": "12",
        "top_left_code": "A123",
        "top_right_code": "1",
        "series": "20026",
        "signature_1": "O. Nordmann",
        "signature_2": "K. Hansen",
    }
    return examples.get(label, "")


def make_debug_crop(input_crop, target_crop, mask_crop):
    w, h = input_crop.size
    debug = Image.new("RGB", (w * 3, h), "white")
    debug.paste(input_crop, (0, 0))
    debug.paste(target_crop, (w, 0))
    debug.paste(mask_crop.convert("RGB"), (w * 2, 0))
    return debug


def main():
    image_dir = find_existing_folder(IMAGE_DIR_CANDIDATES, "bildemappe")
    annotation_dir = find_existing_folder(ANNOTATION_DIR_CANDIDATES, "JSON-mappe")

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_GUIDE_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    json_files = sorted(annotation_dir.glob("*.json"))

    if not json_files:
        print("Fant ingen JSON-filer i:", annotation_dir)
        return

    print("Bruker bildemappe:", image_dir)
    print("Bruker JSON-mappe:", annotation_dir)
    print("Fant JSON-filer:", len(json_files))

    metadata = []
    total_boxes = 0
    skipped = 0

    for json_path in json_files:
        ann = load_json(json_path)
        image_path = find_image_from_annotation(ann, image_dir, json_path)

        if image_path is None:
            print(f"[HOPPER OVER] Fant ikke bilde for {json_path.name}")
            skipped += 1
            continue

        original = Image.open(image_path).convert("RGB")

        image_size_json = ann.get("image_size", {})
        w = image_size_json.get("width")
        h = image_size_json.get("height")

        if w and h:
            image_size = (w, h)

            if original.size != image_size:
                print(f"[INFO] Resizer {image_path.name} fra {original.size} til {image_size}")
                original = original.resize(image_size)
        else:
            image_size = original.size

        boxes = ann.get("boxes", [])

        if not boxes:
            print(f"[HOPPER OVER] Ingen bokser i {json_path.name}")
            skipped += 1
            continue

        print(f"[{json_path.stem}] bokser: {len(boxes)}")

        for i, box in enumerate(boxes):
            label = normalize_label(box.get("label", ""))
            bbox = box.get("bbox")

            if not label or not bbox or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox

            if x2 <= x1 or y2 <= y1:
                continue

            value = box.get("value", default_value_for_label(label))

            crop_box = expand_bbox(bbox, image_size, CROP_MARGIN)

            target_crop = original.crop(crop_box)

            blanked_full = blank_box_in_image(
                original,
                bbox=bbox,
                method=BLANK_METHOD,
                padding=MASK_PADDING
            )
            input_crop = blanked_full.crop(crop_box)

            mask_crop = make_crop_mask(
                crop_size=input_crop.size,
                bbox=bbox,
                crop_box=crop_box,
                padding=MASK_PADDING
            )

            text_guide = create_text_guide(
                crop_size=input_crop.size,
                bbox=bbox,
                crop_box=crop_box,
                text=value
            )

            sample_id = f"{json_path.stem}_{i:03d}_{label}"

            input_path = INPUT_DIR / f"{sample_id}_input.png"
            target_path = TARGET_DIR / f"{sample_id}_target.png"
            mask_path = MASK_DIR / f"{sample_id}_mask.png"
            guide_path = TEXT_GUIDE_DIR / f"{sample_id}_textguide.png"
            debug_path = DEBUG_DIR / f"{sample_id}_debug.png"

            input_crop.save(input_path)
            target_crop.save(target_path)
            mask_crop.save(mask_path)
            text_guide.save(guide_path)

            debug_crop = make_debug_crop(input_crop, target_crop, mask_crop)
            debug_crop.save(debug_path)

            metadata.append({
                "sample_id": sample_id,
                "source_image": str(image_path),
                "source_json": str(json_path),
                "label": label,
                "value": value,
                "bbox": bbox,
                "crop_box": crop_box,
                "input": str(input_path),
                "target": str(target_path),
                "mask": str(mask_path),
                "text_guide": str(guide_path),
                "debug": str(debug_path),
            })

            total_boxes += 1

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        for item in metadata:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print()
    print("Ferdig.")
    print("Antall treningssamples:", total_boxes)
    print("Hoppet over:", skipped)
    print("Metadata:", METADATA_PATH)
    print("Input crops:", INPUT_DIR)
    print("Target crops:", TARGET_DIR)
    print("Mask crops:", MASK_DIR)
    print("Debug crops:", DEBUG_DIR)


if __name__ == "__main__":
    main()
