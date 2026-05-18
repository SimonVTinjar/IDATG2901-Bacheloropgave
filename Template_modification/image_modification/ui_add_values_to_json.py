from pathlib import Path
import json
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk


# -------------------------------------------------
# KONFIG
# -------------------------------------------------

BASE_DIR = Path(__file__).parent

# Datasettet ditt ligger her:
# data/images/
# data/labels/
DATASET_DIR = BASE_DIR / "data"

IMAGE_DIR = DATASET_DIR / "images"
LABEL_DIR = DATASET_DIR / "labels"

VALID_EXTENSIONS = [".png", ".jpg", ".jpeg"]

DISPLAY_MAX_W = 1000
DISPLAY_MAX_H = 650

CROP_PREVIEW_SIZE = 320

LABELS = [
    "serial",
    "serial_2",
    "code",
    "top_left_code",
    "top_right_code",
    "series",
    "signature_1",
    "signature_2",
]


# -------------------------------------------------
# DATA HJELP
# -------------------------------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def normalize_label(label):
    if label == "top_rigth_code":
        return "top_right_code"
    return label


def find_image_by_stem(folder, stem):
    for ext in VALID_EXTENSIONS:
        path = folder / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def find_image_from_annotation(annotation, image_folder, json_path):
    """
    Prøver å finne riktig bilde på flere måter:
    1. annotation["image"]
    2. samme filnavn som JSON
    3. annotation["image_path"]
    """

    image_name = annotation.get("image")
    if image_name:
        path = image_folder / image_name
        if path.exists():
            return path

    path = find_image_by_stem(image_folder, json_path.stem)
    if path is not None:
        return path

    image_path = annotation.get("image_path")
    if image_path:
        path = Path(image_path)
        if path.exists():
            return path

    return None


# -------------------------------------------------
# UI APP
# -------------------------------------------------

class ValueEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("JSON Value Editor")

        if not IMAGE_DIR.exists():
            raise FileNotFoundError(f"Fant ikke bildemappen: {IMAGE_DIR}")

        if not LABEL_DIR.exists():
            raise FileNotFoundError(f"Fant ikke label/JSON-mappen: {LABEL_DIR}")

        self.json_files = sorted(LABEL_DIR.glob("*.json"))

        if not self.json_files:
            raise FileNotFoundError(f"Fant ingen JSON-filer i {LABEL_DIR}")

        self.file_index = 0
        self.box_index = 0

        self.data = None
        self.image_path = None
        self.original_image = None
        self.display_tk = None
        self.crop_tk = None
        self.scale = 1.0

        self.create_widgets()
        self.bind_keys()
        self.load_current_file()

    # -------------------------------------------------
    # UI
    # -------------------------------------------------

    def create_widgets(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        top_bar = ttk.Frame(main)
        top_bar.pack(fill=tk.X)

        self.file_label = ttk.Label(top_bar, text="")
        self.file_label.pack(side=tk.LEFT)

        ttk.Button(top_bar, text="Forrige fil", command=self.prev_file).pack(side=tk.RIGHT, padx=4)
        ttk.Button(top_bar, text="Neste fil", command=self.next_file).pack(side=tk.RIGHT, padx=4)
        ttk.Button(top_bar, text="Lagre", command=self.save_current_json).pack(side=tk.RIGHT, padx=4)

        content = ttk.Frame(main)
        content.pack(fill=tk.BOTH, expand=True, pady=10)

        self.canvas = tk.Canvas(content, bg="gray", width=DISPLAY_MAX_W, height=DISPLAY_MAX_H)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        side = ttk.Frame(content, width=360)
        side.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

        self.box_label = ttk.Label(side, text="", font=("Arial", 12, "bold"))
        self.box_label.pack(anchor="w", pady=(0, 10))

        self.crop_canvas = tk.Canvas(
            side,
            width=CROP_PREVIEW_SIZE,
            height=CROP_PREVIEW_SIZE,
            bg="white"
        )
        self.crop_canvas.pack(pady=5)

        ttk.Label(side, text="Label:").pack(anchor="w", pady=(10, 0))

        self.label_var = tk.StringVar()
        self.label_combo = ttk.Combobox(
            side,
            textvariable=self.label_var,
            values=LABELS
        )
        self.label_combo.pack(fill=tk.X)

        ttk.Label(side, text="Value / tekst som faktisk står i boksen:").pack(anchor="w", pady=(10, 0))

        self.value_var = tk.StringVar()
        self.value_entry = ttk.Entry(
            side,
            textvariable=self.value_var,
            font=("Arial", 14)
        )
        self.value_entry.pack(fill=tk.X)
        self.value_entry.focus()

        buttons = ttk.Frame(side)
        buttons.pack(fill=tk.X, pady=12)

        ttk.Button(
            buttons,
            text="Forrige boks",
            command=self.prev_box
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        ttk.Button(
            buttons,
            text="Neste boks",
            command=self.next_box
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        ttk.Button(
            side,
            text="Lagre value + neste",
            command=self.save_and_next
        ).pack(fill=tk.X, pady=4)

        ttk.Button(
            side,
            text="Skip boks",
            command=self.next_box
        ).pack(fill=tk.X, pady=4)

        self.progress_label = ttk.Label(side, text="")
        self.progress_label.pack(anchor="w", pady=10)

        help_text = (
            "Shortcuts:\n"
            "Enter = lagre + neste boks\n"
            "Ctrl+S = lagre JSON\n"
            "Pil høyre = neste boks\n"
            "Pil venstre = forrige boks\n"
            "Ctrl+Right = neste fil\n"
            "Ctrl+Left = forrige fil"
        )

        ttk.Label(side, text=help_text, justify=tk.LEFT).pack(anchor="w", pady=10)

    def bind_keys(self):
        self.root.bind("<Return>", lambda e: self.save_and_next())
        self.root.bind("<Control-s>", lambda e: self.save_current_json())
        self.root.bind("<Right>", lambda e: self.next_box())
        self.root.bind("<Left>", lambda e: self.prev_box())
        self.root.bind("<Control-Right>", lambda e: self.next_file())
        self.root.bind("<Control-Left>", lambda e: self.prev_file())

    # -------------------------------------------------
    # LOAD / SAVE
    # -------------------------------------------------

    def load_current_file(self):
        json_path = self.json_files[self.file_index]
        self.data = load_json(json_path)

        self.image_path = find_image_from_annotation(
            self.data,
            IMAGE_DIR,
            json_path
        )

        if self.image_path is None:
            messagebox.showerror(
                "Feil",
                f"Fant ikke bilde for {json_path.name}"
            )
            return

        self.original_image = Image.open(self.image_path).convert("RGB")

        image_size = self.data.get("image_size", {})
        width = image_size.get("width")
        height = image_size.get("height")

        if width and height and self.original_image.size != (width, height):
            self.original_image = self.original_image.resize((width, height))

        for box in self.data.get("boxes", []):
            if "label" in box:
                box["label"] = normalize_label(box["label"])

        self.box_index = 0
        self.update_display()

    def save_current_json(self):
        json_path = self.json_files[self.file_index]

        self.save_current_box_fields()
        save_json(json_path, self.data)

        self.update_display()
        print("Lagret:", json_path)

    def save_current_box_fields(self):
        boxes = self.data.get("boxes", [])

        if not boxes:
            return

        box = boxes[self.box_index]

        label = self.label_var.get().strip()
        value = self.value_var.get().strip()

        if label:
            box["label"] = normalize_label(label)

        if value:
            box["value"] = value
        else:
            box.pop("value", None)

    # -------------------------------------------------
    # NAVIGATION
    # -------------------------------------------------

    def next_file(self):
        self.save_current_json()

        if self.file_index < len(self.json_files) - 1:
            self.file_index += 1
            self.load_current_file()

    def prev_file(self):
        self.save_current_json()

        if self.file_index > 0:
            self.file_index -= 1
            self.load_current_file()

    def next_box(self):
        self.save_current_box_fields()

        boxes = self.data.get("boxes", [])

        if not boxes:
            return

        if self.box_index < len(boxes) - 1:
            self.box_index += 1
        else:
            if self.file_index < len(self.json_files) - 1:
                self.save_current_json()
                self.file_index += 1
                self.load_current_file()
                return

        self.update_display()

    def prev_box(self):
        self.save_current_box_fields()

        if self.box_index > 0:
            self.box_index -= 1
            self.update_display()

    def save_and_next(self):
        self.save_current_box_fields()
        self.save_current_json()
        self.next_box()

    # -------------------------------------------------
    # DISPLAY
    # -------------------------------------------------

    def update_display(self):
        self.draw_image()
        self.draw_crop()
        self.update_side_panel()

    def get_display_scale(self):
        width, height = self.original_image.size

        scale = min(
            DISPLAY_MAX_W / width,
            DISPLAY_MAX_H / height,
            1.0
        )

        return scale

    def draw_image(self):
        self.canvas.delete("all")

        scale = self.get_display_scale()
        self.scale = scale

        width, height = self.original_image.size
        display_size = (
            int(width * scale),
            int(height * scale)
        )

        img = self.original_image.resize(display_size)
        self.display_tk = ImageTk.PhotoImage(img)

        self.canvas.config(
            width=display_size[0],
            height=display_size[1]
        )

        self.canvas.create_image(
            0,
            0,
            anchor=tk.NW,
            image=self.display_tk
        )

        boxes = self.data.get("boxes", [])

        for i, box in enumerate(boxes):
            bbox = box.get("bbox")

            if not bbox or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox

            x1 *= scale
            y1 *= scale
            x2 *= scale
            y2 *= scale

            if i == self.box_index:
                color = "lime"
                width = 4
            else:
                color = "red"
                width = 2

            self.canvas.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                outline=color,
                width=width
            )

            self.canvas.create_text(
                x1 + 8,
                max(10, y1 - 10),
                anchor=tk.W,
                text=str(i + 1),
                fill=color,
                font=("Arial", 14, "bold")
            )

    def draw_crop(self):
        self.crop_canvas.delete("all")

        boxes = self.data.get("boxes", [])

        if not boxes:
            return

        box = boxes[self.box_index]
        bbox = box.get("bbox")

        if not bbox or len(bbox) != 4:
            return

        x1, y1, x2, y2 = bbox
        pad = 40

        width, height = self.original_image.size

        x1p = max(0, x1 - pad)
        y1p = max(0, y1 - pad)
        x2p = min(width, x2 + pad)
        y2p = min(height, y2 + pad)

        crop = self.original_image.crop((x1p, y1p, x2p, y2p))

        crop_w, crop_h = crop.size

        scale = min(
            CROP_PREVIEW_SIZE / crop_w,
            CROP_PREVIEW_SIZE / crop_h,
            1.0
        )

        crop_display = crop.resize((
            int(crop_w * scale),
            int(crop_h * scale)
        ))

        self.crop_tk = ImageTk.PhotoImage(crop_display)

        self.crop_canvas.create_image(
            0,
            0,
            anchor=tk.NW,
            image=self.crop_tk
        )

        bx1 = (x1 - x1p) * scale
        by1 = (y1 - y1p) * scale
        bx2 = (x2 - x1p) * scale
        by2 = (y2 - y1p) * scale

        self.crop_canvas.create_rectangle(
            bx1,
            by1,
            bx2,
            by2,
            outline="lime",
            width=3
        )

    def update_side_panel(self):
        json_path = self.json_files[self.file_index]
        boxes = self.data.get("boxes", [])

        self.file_label.config(
            text=f"Fil {self.file_index + 1}/{len(self.json_files)}: {json_path.name}"
        )

        if not boxes:
            self.box_label.config(text="Ingen bokser")
            self.progress_label.config(text="")
            return

        box = boxes[self.box_index]

        label = box.get("label", "")
        value = box.get("value", "")

        self.label_var.set(label)
        self.value_var.set(value)

        self.box_label.config(
            text=f"Boks {self.box_index + 1}/{len(boxes)}"
        )

        filled = sum(1 for b in boxes if b.get("value"))

        self.progress_label.config(
            text=f"Utfylt i denne filen: {filled}/{len(boxes)}"
        )


# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main():
    root = tk.Tk()
    app = ValueEditorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()