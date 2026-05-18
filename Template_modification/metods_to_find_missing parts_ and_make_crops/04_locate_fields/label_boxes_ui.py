import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

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


class BoxLabeler:
    def __init__(self, root):
        self.root = root
        self.root.title("Label bokser")
        self.root.geometry("1200x800")

        self.image_path = None
        self.original = None
        self.tk_img = None
        self.scale = 1.0

        self.selected_class = "serial"
        self.boxes = []

        self.start_x = None
        self.start_y = None
        self.temp_rect = None

        self.build_ui()

    def build_ui(self):
        top = tk.Frame(self.root)
        top.pack(fill=tk.X, padx=8, pady=8)

        tk.Button(top, text="Åpne bilde", command=self.open_image).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Lagre JSON", command=self.save_json).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Slett siste", command=self.undo_last).pack(side=tk.LEFT, padx=4)

        self.status = tk.StringVar(value="Åpne et bilde for å starte.")
        tk.Label(top, textvariable=self.status).pack(side=tk.LEFT, padx=12)

        class_frame = tk.Frame(self.root)
        class_frame.pack(fill=tk.X, padx=8)

        for key, name in CLASSES.items():
            tk.Button(
                class_frame,
                text=f"{key}: {name}",
                command=lambda n=name: self.set_class(n)
            ).pack(side=tk.LEFT, padx=3, pady=3)

        self.canvas = tk.Canvas(self.root, bg="#ddd", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self.mouse_down)
        self.canvas.bind("<B1-Motion>", self.mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.mouse_up)

        self.root.bind("<Key>", self.key_press)

    def get_labels_dir(self):
        """
        Forventet struktur:
        YOLO_training/
          images/
            bilde.png
          labels/
            bilde.json
        """
        if self.image_path is None:
            return None

        images_dir = self.image_path.parent
        root_dir = images_dir.parent
        labels_dir = root_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)

        return labels_dir

    def get_json_path_for_current_image(self):
        labels_dir = self.get_labels_dir()
        if labels_dir is None or self.image_path is None:
            return None

        return labels_dir / f"{self.image_path.stem}.json"

    def load_existing_json(self):
        json_path = self.get_json_path_for_current_image()

        if json_path is None or not json_path.exists():
            self.boxes = []
            self.status.set("Ingen eksisterende JSON funnet.")
            return

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.boxes = data.get("boxes", [])
            self.status.set(f"Lastet eksisterende bokser fra {json_path.name}")
            print("Lastet:", json_path)

        except Exception as e:
            self.boxes = []
            messagebox.showerror("Feil", f"Kunne ikke lese JSON:\n{e}")

    def set_class(self, name):
        self.selected_class = name
        self.status.set(f"Valgt klasse: {name}")

    def key_press(self, event):
        key = event.char.lower()

        if key in CLASSES:
            self.set_class(CLASSES[key])

        elif key == "u":
            self.undo_last()

    def open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp")]
        )

        if not path:
            return

        self.image_path = Path(path)
        self.original = Image.open(path).convert("RGB")

        self.load_existing_json()
        self.render()

    def render(self):
        if self.original is None:
            return

        cw = max(self.canvas.winfo_width(), 400)
        ch = max(self.canvas.winfo_height(), 300)

        iw, ih = self.original.size
        self.scale = min(cw / iw, ch / ih, 1.0)

        dw = int(iw * self.scale)
        dh = int(ih * self.scale)

        img = self.original.resize((dw, dh))
        self.tk_img = ImageTk.PhotoImage(img)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.tk_img, anchor="nw")

        for box in self.boxes:
            x1, y1, x2, y2 = box["bbox"]
            sx1, sy1, sx2, sy2 = [v * self.scale for v in [x1, y1, x2, y2]]

            self.canvas.create_rectangle(sx1, sy1, sx2, sy2, outline="red", width=2)
            self.canvas.create_text(
                sx1 + 4,
                sy1 + 4,
                text=box["label"],
                fill="red",
                anchor="nw",
                font=("Arial", 10, "bold")
            )

    def mouse_down(self, event):
        if self.original is None:
            return

        self.start_x = event.x
        self.start_y = event.y

        self.temp_rect = self.canvas.create_rectangle(
            event.x,
            event.y,
            event.x,
            event.y,
            outline="blue",
            width=2,
            dash=(4, 2)
        )

    def mouse_drag(self, event):
        if self.temp_rect:
            self.canvas.coords(
                self.temp_rect,
                self.start_x,
                self.start_y,
                event.x,
                event.y
            )

    def mouse_up(self, event):
        if self.original is None or self.temp_rect is None:
            return

        x1 = min(self.start_x, event.x)
        y1 = min(self.start_y, event.y)
        x2 = max(self.start_x, event.x)
        y2 = max(self.start_y, event.y)

        self.canvas.delete(self.temp_rect)
        self.temp_rect = None

        if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
            return

        ox1 = int(x1 / self.scale)
        oy1 = int(y1 / self.scale)
        ox2 = int(x2 / self.scale)
        oy2 = int(y2 / self.scale)

        self.boxes.append({
            "label": self.selected_class,
            "bbox": [ox1, oy1, ox2, oy2]
        })

        self.status.set(f"La til {self.selected_class}: {[ox1, oy1, ox2, oy2]}")
        self.render()

    def undo_last(self):
        if self.boxes:
            removed = self.boxes.pop()
            self.status.set(f"Slettet siste: {removed['label']}")
            self.render()

    def save_json(self):
        if self.original is None or self.image_path is None:
            messagebox.showinfo("Mangler bilde", "Åpne et bilde først.")
            return

        out_path = self.get_json_path_for_current_image()

        data = {
            "image": self.image_path.name,
            "image_path": str(self.image_path),
            "image_size": {
                "width": self.original.size[0],
                "height": self.original.size[1]
            },
            "boxes": self.boxes
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print("Lagret:", out_path)
        messagebox.showinfo("Lagret", f"Lagret automatisk:\n{out_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = BoxLabeler(root)
    root.mainloop()