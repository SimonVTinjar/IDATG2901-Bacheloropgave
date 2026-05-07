from pathlib import Path
from PIL import Image
import json


# -------------------------------------------------
# KONFIG
# -------------------------------------------------

BASE_DIR = Path(__file__).parent

# Datasett-mappen din
DATASET_DIR = BASE_DIR / "data"

# Scriptet prøver disse mappene for bilder
IMAGE_DIR_CANDIDATES = [
    DATASET_DIR / "images",
    DATASET_DIR / "originals",
    DATASET_DIR
]

# Scriptet prøver disse mappene for JSON
ANNOTATION_DIR_CANDIDATES = [
    DATASET_DIR / "labels",
    DATASET_DIR / "json",
    DATASET_DIR
]

# Output
OUTPUT_DIR = BASE_DIR / "reference_crops"

VALID_EXTENSIONS = [".png", ".jpg", ".jpeg"]

# Hvor mye ekstra luft rundt cropen.
# 0 = bare boksen.
# 2-5 = litt kontekst.
CROP_PADDING = 0

# Slett gammel reference_crops før ny kjøring?
CLEAR_OUTPUT_FIRST = True


# -------------------------------------------------
# HJELPEFUNKSJONER
# -------------------------------------------------

def find_existing_folder(candidates, name):
    for folder in candidates:
        if folder.exists() and folder.is_dir():
            return folder

    raise FileNotFoundError(
        f"Fant ikke {name}. Sjekk at mappen 'datta set' ligger ved siden av scriptet."
    )


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_label(label):
    if label == "top_rigth_code":
        return "top_right_code"
    return label


def find_image_by_stem(folder, stem):
    for ext in VALID_EXTENSIONS:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def find_image_from_annotation(annotation, image_folder, json_path):
    """
    Prøver å finne bildet på flere måter:

    1. Bruker annotation["image"]
    2. Bruker samme filnavn som JSON
    3. Bruker annotation["image_path"] hvis den finnes
    """

    # 1. image-feltet i JSON
    image_name = annotation.get("image")
    if image_name:
        p = image_folder / image_name
        if p.exists():
            return p

    # 2. samme stem som JSON
    p = find_image_by_stem(image_folder, json_path.stem)
    if p is not None:
        return p

    # 3. image_path fra JSON
    image_path = annotation.get("image_path")
    if image_path:
        p = Path(image_path)
        if p.exists():
            return p

    return None


def clamp(v, low, high):
    return max(low, min(high, v))


def crop_with_padding(img, bbox, padding=0):
    x1, y1, x2, y2 = bbox
    w, h = img.size

    x1 = clamp(x1 - padding, 0, w)
    y1 = clamp(y1 - padding, 0, h)
    x2 = clamp(x2 + padding, 0, w)
    y2 = clamp(y2 + padding, 0, h)

    return img.crop((x1, y1, x2, y2))


def clear_output_dir(output_dir):
    if not output_dir.exists():
        return

    for path in output_dir.rglob("*"):
        if path.is_file():
            path.unlink()

    # fjern tomme undermapper
    for path in sorted(output_dir.rglob("*"), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass


# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main():
    image_dir = find_existing_folder(IMAGE_DIR_CANDIDATES, "bildemappe")
    annotation_dir = find_existing_folder(ANNOTATION_DIR_CANDIDATES, "JSON-mappe")

    if CLEAR_OUTPUT_FIRST:
        clear_output_dir(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_files = sorted(annotation_dir.glob("*.json"))

    if not json_files:
        print("Fant ingen JSON-filer i:", annotation_dir)
        return

    print("Bruker bildemappe:", image_dir)
    print("Bruker JSON-mappe:", annotation_dir)
    print("Fant JSON-filer:", len(json_files))
    print()

    total_crops = 0
    skipped_images = 0
    skipped_boxes = 0
    count_by_label = {}

    for json_path in json_files:
        try:
            annotation = load_json(json_path)
        except Exception as e:
            print(f"[HOPPER OVER] Klarte ikke lese JSON {json_path.name}: {e}")
            skipped_images += 1
            continue

        image_path = find_image_from_annotation(annotation, image_dir, json_path)

        if image_path is None:
            print(f"[HOPPER OVER] Fant ikke bilde for {json_path.name}")
            skipped_images += 1
            continue

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[HOPPER OVER] Klarte ikke åpne bilde {image_path.name}: {e}")
            skipped_images += 1
            continue

        # Resize hvis JSON sier en annen størrelse
        image_size = annotation.get("image_size", {})
        width = image_size.get("width")
        height = image_size.get("height")

        if width and height and img.size != (width, height):
            print(f"[INFO] Resizer {image_path.name} fra {img.size} til {(width, height)}")
            img = img.resize((width, height))

        boxes = annotation.get("boxes", [])

        if not boxes:
            print(f"[HOPPER OVER] Ingen bokser i {json_path.name}")
            skipped_images += 1
            continue

        print(f"[{json_path.stem}] bokser: {len(boxes)}")

        for i, box_info in enumerate(boxes):
            label = box_info.get("label")
            bbox = box_info.get("bbox")

            if not label or not bbox or len(bbox) != 4:
                skipped_boxes += 1
                continue

            label = normalize_label(label)

            x1, y1, x2, y2 = bbox

            if x2 <= x1 or y2 <= y1:
                skipped_boxes += 1
                continue

            crop = crop_with_padding(img, bbox, padding=CROP_PADDING)

            label_dir = OUTPUT_DIR / label
            label_dir.mkdir(parents=True, exist_ok=True)

            output_name = f"{json_path.stem}_{i:03d}.png"
            output_path = label_dir / output_name

            crop.save(output_path)

            total_crops += 1
            count_by_label[label] = count_by_label.get(label, 0) + 1

    print()
    print("Ferdig.")
    print("Lagret reference crops i:", OUTPUT_DIR)
    print("Antall crops:", total_crops)
    print("Hoppet over bilder/JSON:", skipped_images)
    print("Hoppet over bokser:", skipped_boxes)
    print()
    print("Antall per label:")

    for label, count in sorted(count_by_label.items()):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()