from pathlib import Path
from PIL import Image, ImageDraw
import json


# -------------------------------------------------
# INNSTILLINGER
# -------------------------------------------------

BASE_DIR = Path(__file__).parent

ANNOTATION_DIR = BASE_DIR / "labels"
IMAGE_DIR = BASE_DIR / "images"

OUTPUT_DIR = BASE_DIR / "output"
MASK_DIR = OUTPUT_DIR / "masks"
DEBUG_DIR = OUTPUT_DIR / "debug"

CROP_DIR = OUTPUT_DIR / "crops"
CROP_INPUT_DIR = CROP_DIR / "input"
CROP_MASK_DIR = CROP_DIR / "mask"

METADATA_PATH = OUTPUT_DIR / "metadata.jsonl"

MASK_PADDING = 4
CROP_MARGIN = 20


# -------------------------------------------------
# HJELPEFUNKSJONER
# -------------------------------------------------

def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_image(annotation, json_path):
    """
    Prøver å finne bildet som hører til JSON-filen.

    Først bruker den annotation["image"].
    Hvis den ikke finner bildet i images-mappen,
    prøver den annotation["image_path"].
    """
    image_name = annotation.get("image")

    if image_name:
        local_image_path = IMAGE_DIR / image_name
        if local_image_path.exists():
            return local_image_path

    image_path = annotation.get("image_path")
    if image_path:
        image_path = Path(image_path)
        if image_path.exists():
            return image_path

    # Siste fallback:
    # prøv samme navn som json-filen, men med vanlige bildeendelser
    for ext in [".jpg", ".jpeg", ".png"]:
        fallback = IMAGE_DIR / f"{json_path.stem}{ext}"
        if fallback.exists():
            return fallback

    return None


def make_full_mask(image_size, boxes, padding=4):
    """
    Lager full mask for hele bildet.
    Svart = ikke endre.
    Hvit = bokser som kan endres.
    """
    width, height = image_size

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    for box in boxes:
        x1, y1, x2, y2 = box["bbox"]

        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)

        draw.rectangle([x1, y1, x2, y2], fill=255)

    return mask


def make_debug_image(image, boxes):
    """
    Lager debug-bilde med røde bokser og labels.
    """
    debug = image.copy()
    draw = ImageDraw.Draw(debug)

    for box in boxes:
        label = box["label"]
        x1, y1, x2, y2 = box["bbox"]

        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, max(0, y1 - 12)), label, fill="red")

    return debug


def expand_box(bbox, image_size, margin=20):
    """
    Lager et større crop rundt boksen.
    """
    x1, y1, x2, y2 = bbox
    width, height = image_size

    crop_x1 = max(0, x1 - margin)
    crop_y1 = max(0, y1 - margin)
    crop_x2 = min(width, x2 + margin)
    crop_y2 = min(height, y2 + margin)

    return [crop_x1, crop_y1, crop_x2, crop_y2]


def make_crop_mask(crop_size, bbox, crop_box, padding=2):
    """
    Lager mask for én crop.
    """
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
    x1, y1, x2, y2 = bbox

    local_x1 = x1 - crop_x1
    local_y1 = y1 - crop_y1
    local_x2 = x2 - crop_x1
    local_y2 = y2 - crop_y1

    width, height = crop_size

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    local_x1 = max(0, local_x1 - padding)
    local_y1 = max(0, local_y1 - padding)
    local_x2 = min(width, local_x2 + padding)
    local_y2 = min(height, local_y2 + padding)

    draw.rectangle([local_x1, local_y1, local_x2, local_y2], fill=255)

    return mask


def safe_name(text):
    """
    Gjør filnavn tryggere.
    """
    return (
        text.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


# -------------------------------------------------
# HOVEDPIPELINE
# -------------------------------------------------

def process_one_json(json_path, metadata_lines):
    annotation = load_json(json_path)

    image_path = find_image(annotation, json_path)

    if image_path is None:
        print(f"[ADVARSEL] Fant ikke bilde for: {json_path.name}")
        return

    image = Image.open(image_path).convert("RGB")

    width = annotation["image_size"]["width"]
    height = annotation["image_size"]["height"]
    expected_size = (width, height)

    if image.size != expected_size:
        print(
            f"[INFO] Resizer {image_path.name} "
            f"fra {image.size} til {expected_size}"
        )
        image = image.resize(expected_size)

    boxes = annotation.get("boxes", [])

    if not boxes:
        print(f"[ADVARSEL] Ingen bokser i: {json_path.name}")
        return

    image_id = json_path.stem

    # -----------------------------
    # 1. Full mask
    # -----------------------------
    full_mask = make_full_mask(expected_size, boxes, padding=MASK_PADDING)
    full_mask_path = MASK_DIR / f"{image_id}_mask.png"
    full_mask.save(full_mask_path)

    # -----------------------------
    # 2. Debug-bilde
    # -----------------------------
    debug_image = make_debug_image(image, boxes)
    debug_path = DEBUG_DIR / f"{image_id}_debug.png"
    debug_image.save(debug_path)

    # -----------------------------
    # 3. Crops per boks
    # -----------------------------
    for i, box in enumerate(boxes):
        label = box["label"]
        bbox = box["bbox"]

        crop_box = expand_box(
            bbox=bbox,
            image_size=expected_size,
            margin=CROP_MARGIN
        )

        crop = image.crop(crop_box)
        crop_mask = make_crop_mask(
            crop_size=crop.size,
            bbox=bbox,
            crop_box=crop_box,
            padding=2
        )

        sample_id = f"{image_id}_{i:03d}_{safe_name(label)}"

        input_crop_name = f"{sample_id}_input.png"
        mask_crop_name = f"{sample_id}_mask.png"

        input_crop_path = CROP_INPUT_DIR / input_crop_name
        mask_crop_path = CROP_MASK_DIR / mask_crop_name

        crop.save(input_crop_path)
        crop_mask.save(mask_crop_path)

        metadata = {
            "sample_id": sample_id,
            "source_json": str(json_path),
            "source_image": str(image_path),
            "label": label,
            "bbox": bbox,
            "crop_box": crop_box,
            "full_mask": str(full_mask_path),
            "input_crop": str(input_crop_path),
            "mask_crop": str(mask_crop_path)
        }

        # Hvis du senere legger til "value" i JSON,
        # blir den automatisk tatt med her.
        if "value" in box:
            metadata["value"] = box["value"]

        metadata_lines.append(metadata)

    print(f"[OK] {json_path.name}: {len(boxes)} bokser")


def main():
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    CROP_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    CROP_MASK_DIR.mkdir(parents=True, exist_ok=True)

    json_files = sorted(ANNOTATION_DIR.glob("*.json"))

    if not json_files:
        print("Fant ingen JSON-filer i:", ANNOTATION_DIR)
        return

    metadata_lines = []

    print(f"Fant {len(json_files)} JSON-filer.")

    for json_path in json_files:
        process_one_json(json_path, metadata_lines)

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        for item in metadata_lines:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\nFerdig.")
    print("Antall crop-samples:", len(metadata_lines))
    print("Metadata lagret:", METADATA_PATH)
    print("Masker lagret:", MASK_DIR)
    print("Debug-bilder lagret:", DEBUG_DIR)
    print("Crops lagret:", CROP_DIR)


if __name__ == "__main__":
    main()