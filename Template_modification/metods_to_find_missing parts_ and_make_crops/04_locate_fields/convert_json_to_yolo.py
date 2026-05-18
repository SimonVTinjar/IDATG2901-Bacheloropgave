import json
from pathlib import Path

BASE_DIR = Path(__file__).parent

IMAGES_DIR = BASE_DIR / "YOLO_training" / "images"
LABELS_JSON_DIR = BASE_DIR / "YOLO_training" / "labels"
LABELS_YOLO_DIR = BASE_DIR / "YOLO_training" / "labels_yolo"

LABELS_YOLO_DIR.mkdir(parents=True, exist_ok=True)

CLASS_MAP = {
    "serial": 0,
    "serial_2": 1,
    "code": 2,
    "top_left_code": 3,
    "top_right_code": 4,
    "series": 5,
    "signature_1": 6,
    "signature_2": 7,
    "noise": 8
}


def convert():
    for json_file in LABELS_JSON_DIR.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_w = data["image_size"]["width"]
        img_h = data["image_size"]["height"]

        lines = []

        for box in data["boxes"]:
            label = box["label"]

            if label not in CLASS_MAP:
                continue

            class_id = CLASS_MAP[label]

            x1, y1, x2, y2 = box["bbox"]

            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h

            lines.append(f"{class_id} {cx} {cy} {w} {h}")

        out_file = LABELS_YOLO_DIR / (json_file.stem + ".txt")

        with open(out_file, "w") as f:
            f.write("\n".join(lines))

        print("Lagret:", out_file)


if __name__ == "__main__":
    convert()