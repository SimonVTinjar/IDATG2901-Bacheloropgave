from pathlib import Path
import json
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import cv2

BASE_DIR = Path(__file__).parent

ORIGINAL_PATH = "image.png"
TEMPLATE_PATH = "Prob Done v2.png"

OUTPUT_JSON = BASE_DIR / "fields.json"

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


class LabelBoxesApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Label template boxes")

        self.original = Image.open(ORIGINAL_PATH).convert("RGB")
        template = Image.open(TEMPLATE_PATH).convert("RGB")

        self.boxes, self.template = get_candidate_boxes(self.original, template)
        self.index = 0
        self.fields = []

        self.label = tk.Label(root, text="", font=("Arial", 16))
        self.label.pack(pady=10)

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

        help_text = (
            "Venstre = original, høyre = template\n\n"
            + "\n".join([f"{k} = {v}" for k, v in CLASSES.items()])
            + "\ns = skip\nq = quit"
        )
        self.help_label = tk.Label(root, text=help_text, font=("Arial", 12))
        self.help_label.pack(pady=10)

        root.bind("<Key>", self.on_key)
        self.show_current()

    def show_current(self):
        if self.index >= len(self.boxes):
            self.save_fields()
            self.label.config(text="Ferdig! Lagret fields.json")
            self.image_label.config(image="")
            return

        item = self.boxes[self.index]
        x1, y1, x2, y2 = item["bbox"]

        original_crop = self.original.crop((x1, y1, x2, y2))
        template_crop = self.template.crop((x1, y1, x2, y2))

        original_crop.thumbnail((350, 250))
        template_crop.thumbnail((350, 250))

        w1, h1 = original_crop.size
        w2, h2 = template_crop.size

        combined = Image.new("RGB", (w1 + w2 + 20, max(h1, h2)), "white")
        combined.paste(original_crop, (0, 0))
        combined.paste(template_crop, (w1 + 20, 0))

        self.tk_img = ImageTk.PhotoImage(combined)
        self.image_label.config(image=self.tk_img)

        self.label.config(
            text=f"{self.index + 1}/{len(self.boxes)} bbox={item['bbox']} area={item['area']}"
        )

    def save_fields(self):
        data = {
            "image_size": {
                "width": self.original.size[0],
                "height": self.original.size[1]
            },
            "fields": self.fields
        }

        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print("Lagret:", OUTPUT_JSON)

    def on_key(self, event):
        key = event.char.lower()

        if key in CLASSES:
            cls_name = CLASSES[key]
            box = self.boxes[self.index]["bbox"]

            if cls_name != "noise":
                self.fields.append({
                    "label": cls_name,
                    "bbox": box
                })
                print("Lagret felt:", cls_name, box)
            else:
                print("Ignorerte noise:", box)

            self.index += 1
            self.show_current()

        elif key == "s":
            print("Skippet:", self.index + 1)
            self.index += 1
            self.show_current()

        elif key == "q":
            self.save_fields()
            self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = LabelBoxesApp(root)
    root.mainloop()
