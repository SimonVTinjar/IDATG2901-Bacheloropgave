import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk


class BoxAnnotatorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Box Annotator")
        self.root.geometry("1200x800")

        self.image_path = None
        self.original_image = None
        self.tk_image = None
        self.scale = 1.0

        self.boxes = []
        self.current_rect_id = None
        self.start_x = None
        self.start_y = None
        self.temp_x = None
        self.temp_y = None

        self._build_ui()

    def _build_ui(self):
        top = tk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        tk.Button(top, text="Åpne bilde", command=self.open_image).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Lagre JSON", command=self.save_json).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Slett valgt", command=self.delete_selected).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Tøm alle", command=self.clear_boxes).pack(side=tk.LEFT, padx=4)

        self.status_var = tk.StringVar(value="Åpne et bilde for å starte.")
        tk.Label(top, textvariable=self.status_var).pack(side=tk.LEFT, padx=12)

        main = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        main.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(main)
        right = tk.Frame(main, width=280)
        main.add(left, stretch="always")
        main.add(right)

        self.canvas = tk.Canvas(left, bg="#dddddd", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        tk.Label(right, text="Bokser", font=("Arial", 12, "bold")).pack(anchor="w", padx=8, pady=(8, 4))

        self.box_list = tk.Listbox(right)
        self.box_list.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        self.box_list.bind("<<ListboxSelect>>", self.on_select_box)

        help_text = (
            "Bruk:\n"
            "1. Åpne et bilde\n"
            "2. Dra med musa for å tegne en boks\n"
            "3. Skriv navn på boksen\n"
            "4. Lagre som JSON\n\n"
            "JSON lagrer koordinater i original bildestørrelse."
        )
        tk.Label(right, text=help_text, justify=tk.LEFT, anchor="w").pack(fill=tk.X, padx=8, pady=8)

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Velg bilde",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.webp")]
        )
        if not path:
            return

        try:
            self.image_path = path
            self.original_image = Image.open(path).convert("RGB")
            self.boxes = []
            self.refresh_box_list()
            self.render_image()
            self.status_var.set(f"Lastet bilde: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Feil", f"Kunne ikke åpne bildet:\n{e}")

    def render_image(self):
        if self.original_image is None:
            return

        canvas_w = max(self.canvas.winfo_width(), 200)
        canvas_h = max(self.canvas.winfo_height(), 200)
        img_w, img_h = self.original_image.size

        self.scale = min(canvas_w / img_w, canvas_h / img_h, 1.0)
        display_w = int(img_w * self.scale)
        display_h = int(img_h * self.scale)

        display_img = self.original_image.resize((display_w, display_h), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(display_img)

        self.canvas.delete("all")
        self.canvas.config(scrollregion=(0, 0, display_w, display_h))
        self.canvas.create_image(0, 0, image=self.tk_image, anchor="nw", tags="base_image")

        for i, box in enumerate(self.boxes):
            self.draw_saved_box(i, box)

    def draw_saved_box(self, index, box):
        x1, y1, x2, y2 = box["bbox"]
        sx1, sy1, sx2, sy2 = [v * self.scale for v in (x1, y1, x2, y2)]

        color = "red"
        self.canvas.create_rectangle(sx1, sy1, sx2, sy2, outline=color, width=2)
        self.canvas.create_text(
            sx1 + 4,
            sy1 + 4,
            text=f'{index + 1}: {box["label"]}',
            anchor="nw",
            fill=color,
            font=("Arial", 10, "bold")
        )

    def on_mouse_down(self, event):
        if self.original_image is None:
            return

        self.start_x = event.x
        self.start_y = event.y
        self.temp_x = event.x
        self.temp_y = event.y

        if self.current_rect_id is not None:
            self.canvas.delete(self.current_rect_id)
            self.current_rect_id = None

        self.current_rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline="blue", width=2, dash=(4, 2)
        )

    def on_mouse_drag(self, event):
        if self.current_rect_id is None:
            return

        self.temp_x = event.x
        self.temp_y = event.y
        self.canvas.coords(self.current_rect_id, self.start_x, self.start_y, self.temp_x, self.temp_y)

    def on_mouse_up(self, event):
        if self.current_rect_id is None or self.original_image is None:
            return

        end_x, end_y = event.x, event.y
        x1 = min(self.start_x, end_x)
        y1 = min(self.start_y, end_y)
        x2 = max(self.start_x, end_x)
        y2 = max(self.start_y, end_y)

        if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
            self.canvas.delete(self.current_rect_id)
            self.current_rect_id = None
            return

        label = simpledialog.askstring("Navn på boks", "Skriv label/navn på boksen:\nEksempel: title, subtitle, value")
        if not label:
            self.canvas.delete(self.current_rect_id)
            self.current_rect_id = None
            return

        ox1 = int(x1 / self.scale)
        oy1 = int(y1 / self.scale)
        ox2 = int(x2 / self.scale)
        oy2 = int(y2 / self.scale)

        self.boxes.append({
            "label": label.strip(),
            "bbox": [ox1, oy1, ox2, oy2]
        })

        self.canvas.delete(self.current_rect_id)
        self.current_rect_id = None

        self.refresh_box_list()
        self.render_image()
        self.status_var.set(f"La til boks: {label.strip()}")

    def refresh_box_list(self):
        self.box_list.delete(0, tk.END)
        for i, box in enumerate(self.boxes):
            self.box_list.insert(tk.END, f'{i + 1}. {box["label"]} - {box["bbox"]}')

    def on_select_box(self, _event=None):
        selection = self.box_list.curselection()
        if not selection:
            return
        idx = selection[0]
        box = self.boxes[idx]
        self.status_var.set(f'Valgt: {box["label"]} {box["bbox"]}')

    def delete_selected(self):
        selection = self.box_list.curselection()
        if not selection:
            messagebox.showinfo("Ingen valgt", "Velg en boks først.")
            return

        idx = selection[0]
        removed = self.boxes.pop(idx)
        self.refresh_box_list()
        self.render_image()
        self.status_var.set(f'Slettet boks: {removed["label"]}')

    def clear_boxes(self):
        if not self.boxes:
            return
        if not messagebox.askyesno("Bekreft", "Vil du slette alle boksene?"):
            return
        self.boxes = []
        self.refresh_box_list()
        self.render_image()
        self.status_var.set("Alle bokser ble slettet.")

    def save_json(self):
        if self.original_image is None or self.image_path is None:
            messagebox.showinfo("Ingen bilde", "Åpne et bilde først.")
            return

        output_path = filedialog.asksaveasfilename(
            title="Lagre JSON",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        if not output_path:
            return

        data = {
            "image": os.path.basename(self.image_path),
            "image_path": self.image_path,
            "image_size": {
                "width": self.original_image.size[0],
                "height": self.original_image.size[1]
            },
            "boxes": self.boxes
        }

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.status_var.set(f"Lagret JSON: {os.path.basename(output_path)}")
            messagebox.showinfo("Lagret", f"Lagret til:\n{output_path}")
        except Exception as e:
            messagebox.showerror("Feil", f"Kunne ikke lagre JSON:\n{e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = BoxAnnotatorApp(root)

    def on_resize(_event=None):
        app.render_image()

    app.canvas.bind("<Configure>", on_resize)
    root.mainloop()
