from pathlib import Path
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import cv2
import json

BASE_DIR = Path(__file__).parent

ORIGINAL_PATH = BASE_DIR.parent / "image.png"
TEMPLATE_PATH = BASE_DIR.parent / "Prob Done v2.png"

OUT_DIR = BASE_DIR / "dataset_type"
METADATA_PATH = OUT_DIR / "metadata.json"

THRESHOLD = 1
MIN_AREA = 10000

HORIZONTAL_JOIN = 80
VERTICAL_JOIN = 8

CLASSES = {
    "1": "serial",
    "2": "serial_2",
    "3": "code",
    "4": "top_left_code",
    "5": "top_right_code",
    "6": "series",
    "7": "signature_1",
    "8": "signature_2",
    "9": "noise"
}


def get_candidate_boxes(original_img, template_img):
    template_img = template_img.resize(original_img.size)

    orig_np = np.array(original_img)
    temp_np = np.array(template_img)

    orig_gray = cv2.cvtColor(orig_np, cv2.COLOR_RGB2GRAY)
    temp_gray = cv2.cvtColor(temp_np, cv2.COLOR_RGB2GRAY)

    orig_gray = cv2.GaussianBlur(orig_gray, (7, 7), 0)
    temp_gray = cv2.GaussianBlur(temp_gray, (7, 7), 0)

    diff = cv2.absdiff(orig_gray, temp_gray)
    _, mask = cv2.threshold(diff, THRESHOLD, 255, cv2.THRESH_BINARY)

    small_kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small_kernel)

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (HORIZONTAL_JOIN, VERTICAL_JOIN)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, horizontal_kernel)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    cv2.imwrite(str(BASE_DIR / "debug_sorter_mask.png"), mask)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    boxes = []

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        if area < MIN_AREA:
            continue

        if w < 10 or h < 5:
            continue

        boxes.append({
            "bbox": [int(x), int(y), int(x + w), int(y + h)],
            "area": int(area)
        })

    return boxes, template_img


class CropSorter:
    def __init__(self, root):
        self.root = root
        self.root.title("Sorter crops")

        original = Image.open(ORIGINAL_PATH).convert("RGB")
        template = Image.open(TEMPLATE_PATH).convert("RGB")

        self.boxes, self.template = get_candidate_boxes(original, template)
        self.original = original
        self.metadata = []

        if METADATA_PATH.exists():
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        print(f"Fant {len(self.boxes)} kandidatbokser")
        print("Debug mask:", BASE_DIR / "debug_sorter_mask.png")

        self.index = 0

        for cls in CLASSES.values():
            (OUT_DIR / cls).mkdir(parents=True, exist_ok=True)

        self.label = tk.Label(root, text="", font=("Arial", 16))
        self.label.pack(pady=10)

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

        help_text = "\n".join([f"{k} = {v}" for k, v in CLASSES.items()]) + "\ns = skip\nq = quit"
        self.help_label = tk.Label(root, text=help_text, font=("Arial", 12))
        self.help_label.pack(pady=10)

        root.bind("<Key>", self.on_key)
        self.show_crop()

    def show_crop(self):
        if self.index >= len(self.boxes):
            self.label.config(text="Ferdig!")
            self.image_label.config(image="")
            return

        item = self.boxes[self.index]
        x1, y1, x2, y2 = item["bbox"]

        crop = self.original.crop((x1, y1, x2, y2))

        preview = crop.copy()
        preview.thumbnail((750, 350))

        self.tk_img = ImageTk.PhotoImage(preview)

        self.label.config(
            text=f"{self.index + 1}/{len(self.boxes)} bbox={item['bbox']} area={item['area']}"
        )
        self.image_label.config(image=self.tk_img)

    def save_metadata(self):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    def save_current(self, cls_name):
        item = self.boxes[self.index]
        x1, y1, x2, y2 = item["bbox"]

        crop = self.original.crop((x1, y1, x2, y2))

        img_w, img_h = self.original.size

        # lagrer 3 like crops
        for copy_idx in range(1, 4):
            filename = f"crop_{self.index + 1:04d}_{copy_idx}_{cls_name}.png"
            out_path = OUT_DIR / cls_name / filename

            crop.save(out_path)
            print("Lagret:", out_path)

            self.metadata.append({
                "crop_path": str(out_path),
                "label": cls_name,
                "bbox": [x1, y1, x2, y2],
                "image_size": [img_w, img_h]
            })

        self.save_metadata()
        print("Oppdatert metadata:", METADATA_PATH)

        self.index += 1
        self.show_crop()

    def on_key(self, event):
        key = event.char.lower()

        if key in CLASSES:
            self.save_current(CLASSES[key])

        elif key == "s":
            print("Skippet:", self.index + 1)
            self.index += 1
            self.show_crop()

        elif key == "q":
            self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = CropSorter(root)
    root.mainloop()