import cv2
import numpy as np
import json
import os
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_1_FILE = os.path.join(BASE_DIR, "image.png")
IMAGE_2_FILE = os.path.join(BASE_DIR, "scene.png")
RESULT_FILE = os.path.join(BASE_DIR, "resultat.png")
SETTINGS_FILE = os.path.join(BASE_DIR, "clip_ui_settings.json")
CLIP_FILE = os.path.join(BASE_DIR, "utklipp.png")

DEFAULT_SETTINGS = {
    "x": 100,
    "y": 100,
    "width": 200,
    "height": 200,
    "rotation": 0.0
}


def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            settings = DEFAULT_SETTINGS.copy()
            settings.update(data)
            return settings
        except Exception as e:
            print(f"Kunne ikke lese {SETTINGS_FILE}: {e}")
    return DEFAULT_SETTINGS.copy()


def save_settings(settings):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=4)


def add_alpha_if_missing(image):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif image.shape[2] == 3:
        b, g, r = cv2.split(image)
        alpha = np.ones_like(b) * 255
        image = cv2.merge([b, g, r, alpha])
    return image


def rotate_image_keep_alpha(image, angle):
    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos_val = abs(M[0, 0])
    sin_val = abs(M[0, 1])

    new_w = int((h * sin_val) + (w * cos_val))
    new_h = int((h * cos_val) + (w * sin_val))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(
        image,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    return rotated


def overlay_png(background, overlay, x, y, w, h, angle):
    result = background.copy()

    if w < 1 or h < 1:
        return result

    resized = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_AREA)
    rotated = rotate_image_keep_alpha(resized, angle)

    rh, rw = rotated.shape[:2]

    if x < 0 or y < 0 or x + rw > background.shape[1] or y + rh > background.shape[0]:
        return result

    overlay_rgb = rotated[:, :, :3]
    alpha = rotated[:, :, 3].astype(np.float32) / 255.0

    roi = result[y:y+rh, x:x+rw].astype(np.float32)
    overlay_rgb = overlay_rgb.astype(np.float32)

    alpha_3 = np.dstack([alpha, alpha, alpha])
    blended = overlay_rgb * alpha_3 + roi * (1.0 - alpha_3)

    result[y:y+rh, x:x+rw] = blended.astype(np.uint8)
    return result


def make_preview_image(img_bgr, max_width=1000, max_height=700):
    h, w = img_bgr.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    return ImageTk.PhotoImage(pil_img)


image1 = cv2.imread(IMAGE_1_FILE, cv2.IMREAD_UNCHANGED)
image2 = cv2.imread(IMAGE_2_FILE, cv2.IMREAD_COLOR)

if image1 is None:
    print(f"Fant ikke {IMAGE_1_FILE}")
    raise SystemExit

if image2 is None:
    print(f"Fant ikke {IMAGE_2_FILE}")
    raise SystemExit

image1 = add_alpha_if_missing(image1)
settings = load_settings()

print("\nMarker området du vil klippe ut fra bilde 1.")
display_img1 = image1[:, :, :3].copy()

roi = cv2.selectROI("Velg område fra bilde 1", display_img1, showCrosshair=True, fromCenter=False)
cv2.destroyWindow("Velg område fra bilde 1")

x1, y1, w1, h1 = roi

if w1 == 0 or h1 == 0:
    print("Ingen region valgt. Avslutter.")
    raise SystemExit

clip = image1[y1:y1+h1, x1:x1+w1].copy()

if clip.size == 0:
    print("Kunne ikke klippe ut området.")
    raise SystemExit

cv2.imwrite(CLIP_FILE, clip)
print(f"Utklipp lagret som {CLIP_FILE}")


class App:
    def __init__(self, root, background, overlay, initial_settings):
        self.root = root
        self.background = background
        self.overlay = overlay
        self.last_result = background.copy()

        self.max_w = background.shape[1]
        self.max_h = background.shape[0]

        self.x_var = tk.IntVar(value=max(0, min(int(initial_settings["x"]), self.max_w - 1)))
        self.y_var = tk.IntVar(value=max(0, min(int(initial_settings["y"]), self.max_h - 1)))
        self.w_var = tk.IntVar(value=max(1, min(int(initial_settings["width"]), self.max_w)))
        self.h_var = tk.IntVar(value=max(1, min(int(initial_settings["height"]), self.max_h)))
        self.r_var = tk.DoubleVar(value=float(initial_settings["rotation"]))

        self.build_ui()
        self.update_preview()

    def build_ui(self):
        self.root.title("Plassering UI")
        self.root.geometry("1350x900")

        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main)
        left.pack(side="left", fill="y", padx=(0, 15))

        right = ttk.Frame(main)
        right.pack(side="right", fill="both", expand=True)

        self.preview_label = ttk.Label(right)
        self.preview_label.pack(fill="both", expand=True)

        ttk.Label(left, text="X").pack(anchor="w")
        tk.Scale(left, from_=0, to=self.max_w, orient="horizontal", variable=self.x_var,
                 command=lambda e: self.update_from_slider(), length=280).pack()
        self.x_entry = ttk.Entry(left, width=12)
        self.x_entry.pack(pady=(0, 10))
        self.x_entry.insert(0, str(self.x_var.get()))

        ttk.Label(left, text="Y").pack(anchor="w")
        tk.Scale(left, from_=0, to=self.max_h, orient="horizontal", variable=self.y_var,
                 command=lambda e: self.update_from_slider(), length=280).pack()
        self.y_entry = ttk.Entry(left, width=12)
        self.y_entry.pack(pady=(0, 10))
        self.y_entry.insert(0, str(self.y_var.get()))

        ttk.Label(left, text="Width").pack(anchor="w")
        tk.Scale(left, from_=1, to=self.max_w, orient="horizontal", variable=self.w_var,
                 command=lambda e: self.update_from_slider(), length=280).pack()
        self.w_entry = ttk.Entry(left, width=12)
        self.w_entry.pack(pady=(0, 10))
        self.w_entry.insert(0, str(self.w_var.get()))

        ttk.Label(left, text="Height").pack(anchor="w")
        tk.Scale(left, from_=1, to=self.max_h, orient="horizontal", variable=self.h_var,
                 command=lambda e: self.update_from_slider(), length=280).pack()
        self.h_entry = ttk.Entry(left, width=12)
        self.h_entry.pack(pady=(0, 10))
        self.h_entry.insert(0, str(self.h_var.get()))

        ttk.Label(left, text="Rotation").pack(anchor="w")
        tk.Scale(left, from_=-360.0, to=360.0, resolution=0.1, orient="horizontal", variable=self.r_var,
                 command=lambda e: self.update_from_slider(), length=280).pack()
        self.r_entry = ttk.Entry(left, width=12)
        self.r_entry.pack(pady=(0, 10))
        self.r_entry.insert(0, str(self.r_var.get()))

        ttk.Button(left, text="Apply", command=self.apply_manual_values).pack(fill="x", pady=3)
        ttk.Button(left, text="Save", command=self.save_result).pack(fill="x", pady=3)

    def sync_entries_from_vars(self):
        self.x_entry.delete(0, tk.END)
        self.x_entry.insert(0, str(self.x_var.get()))

        self.y_entry.delete(0, tk.END)
        self.y_entry.insert(0, str(self.y_var.get()))

        self.w_entry.delete(0, tk.END)
        self.w_entry.insert(0, str(self.w_var.get()))

        self.h_entry.delete(0, tk.END)
        self.h_entry.insert(0, str(self.h_var.get()))

        self.r_entry.delete(0, tk.END)
        self.r_entry.insert(0, f"{self.r_var.get():.1f}")

    def update_from_slider(self):
        self.sync_entries_from_vars()
        self.update_preview()

    def apply_manual_values(self):
        try:
            x = int(float(self.x_entry.get().strip()))
            y = int(float(self.y_entry.get().strip()))
            w = int(float(self.w_entry.get().strip()))
            h = int(float(self.h_entry.get().strip()))
            r = float(self.r_entry.get().strip())

            x = max(0, min(x, self.max_w - 1))
            y = max(0, min(y, self.max_h - 1))
            w = max(1, min(w, self.max_w))
            h = max(1, min(h, self.max_h))
            r = max(-360.0, min(r, 360.0))

            self.x_var.set(x)
            self.y_var.set(y)
            self.w_var.set(w)
            self.h_var.set(h)
            self.r_var.set(r)

            self.sync_entries_from_vars()
            self.update_preview()

        except ValueError:
            messagebox.showerror("Feil", "Skriv gyldige tall i feltene.")

    def update_preview(self):
        x = self.x_var.get()
        y = self.y_var.get()
        w = self.w_var.get()
        h = self.h_var.get()
        r = self.r_var.get()

        self.last_result = overlay_png(self.background, self.overlay, x, y, w, h, r)
        tk_img = make_preview_image(self.last_result)
        self.preview_label.configure(image=tk_img)
        self.preview_label.image = tk_img

    def save_result(self):
        cv2.imwrite(RESULT_FILE, self.last_result)

        data = {
            "x": self.x_var.get(),
            "y": self.y_var.get(),
            "width": self.w_var.get(),
            "height": self.h_var.get(),
            "rotation": float(self.r_var.get())
        }
        save_settings(data)

        messagebox.showinfo("Lagret", f"Lagret som {RESULT_FILE}")


root = tk.Tk()
app = App(root, image2, clip, settings)
root.mainloop()