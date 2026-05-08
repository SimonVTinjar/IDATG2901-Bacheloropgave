from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import json

BASE_DIR = Path(__file__).parent

ORIGINAL_PATH = "image.png"
TEMPLATE_PATH = "Prob Done v2.png"

OUT_FILLED = BASE_DIR / "dataset" / "filled"
OUT_MISSING = BASE_DIR / "dataset" / "missing"
OUT_JSON = BASE_DIR / "diff_boxes.json"
OUT_PREVIEW = BASE_DIR / "diff_preview.png"
OUT_MASK = BASE_DIR / "diff_mask.png"

OUT_FILLED.mkdir(parents=True, exist_ok=True)
OUT_MISSING.mkdir(parents=True, exist_ok=True)

THRESHOLD = 1
MIN_AREA = 100


def main():
    original = Image.open(ORIGINAL_PATH).convert("RGB")
    template = Image.open(TEMPLATE_PATH).convert("RGB")

    template = template.resize(original.size)

    orig_np = np.array(original)
    temp_np = np.array(template)

    orig_gray = cv2.cvtColor(orig_np, cv2.COLOR_RGB2GRAY)
    temp_gray = cv2.cvtColor(temp_np, cv2.COLOR_RGB2GRAY)

    orig_gray = cv2.GaussianBlur(orig_gray, (7, 7), 0)
    temp_gray = cv2.GaussianBlur(temp_gray, (7, 7), 0)

    diff = cv2.absdiff(orig_gray, temp_gray)

    _, mask = cv2.threshold(diff, THRESHOLD, 255, cv2.THRESH_BINARY)

    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    preview = temp_np.copy()
    boxes = []

    crop_index = 1

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        if area < MIN_AREA:
            continue

        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)

        # filled = crop fra original
        filled_crop = original.crop((x1, y1, x2, y2))
        filled_path = OUT_FILLED / f"crop_{crop_index:03d}.png"
        filled_crop.save(filled_path)

        # missing = crop fra template
        missing_crop = template.crop((x1, y1, x2, y2))
        missing_path = OUT_MISSING / f"crop_{crop_index:03d}.png"
        missing_crop.save(missing_path)

        boxes.append({
            "id": crop_index,
            "filled_crop": str(filled_path),
            "missing_crop": str(missing_path),
            "bbox": [x1, y1, x2, y2],
            "area": int(area)
        })

        cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(preview, str(crop_index), (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        crop_index += 1

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"boxes": boxes}, f, indent=2, ensure_ascii=False)

    Image.fromarray(preview).save(OUT_PREVIEW)
    Image.fromarray(mask).save(OUT_MASK)

    print("Ferdig!")
    print("Filled:", OUT_FILLED)
    print("Missing:", OUT_MISSING)
    print("JSON:", OUT_JSON)


if __name__ == "__main__":
    main()