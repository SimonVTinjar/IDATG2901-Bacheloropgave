import os
from PIL import Image

INPUT_DIR = "master_images"
OUTPUT_DIR = "lora_dataset"
TARGET_WIDTH_FULL = 1024   # full view width (keeps aspect ratio)
TARGET_SIZE_CROP = 1024    # crop output size (square). Use 768 if needed.

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Helper: clamp value
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def crop_square_by_center(img, cx, cy, frac):
    """
    Crop a square centered at (cx, cy), where frac is side length as fraction
    of min(W, H). Ensures crop stays within image bounds.
    """
    W, H = img.size
    side = int(frac * min(W, H))
    side = max(64, side)  # safety minimum

    half = side // 2
    left = cx - half
    top = cy - half
    right = left + side
    bottom = top + side

    # Shift to fit within bounds
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > W:
        left -= (right - W)
        right = W
    if bottom > H:
        top -= (bottom - H)
        bottom = H

    # Final clamp
    left = clamp(left, 0, W - side)
    top = clamp(top, 0, H - side)
    right = left + side
    bottom = top + side

    return img.crop((left, top, right, bottom))

def save_crop(img, crop_img, out_name):
    crop_resized = crop_img.resize((TARGET_SIZE_CROP, TARGET_SIZE_CROP), Image.BICUBIC)
    crop_resized.save(os.path.join(OUTPUT_DIR, out_name), quality=95)

# Define square crops as (name, center_x_frac, center_y_frac, side_frac_of_min_dim)
# You can tweak these after a quick visual check.
CROPS = [
    ("portrait",       0.40, 0.52, 0.60),
    ("left_pattern",   0.18, 0.40, 0.38),
    ("right_pattern",  0.78, 0.52, 0.45),
    ("upper_band",     0.62, 0.12, 0.35),
    ("lower_pattern",  0.62, 0.82, 0.40),
    ("ribbon_area",    0.53, 0.40, 0.38),
]

for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        continue

    path = os.path.join(INPUT_DIR, filename)
    img = Image.open(path).convert("RGB")
    W, H = img.size
    base = os.path.splitext(filename)[0]

    # 1) Full view: resize width, keep aspect ratio
    new_h = int((TARGET_WIDTH_FULL / W) * H)
    full_resized = img.resize((TARGET_WIDTH_FULL, new_h), Image.BICUBIC)
    full_resized.save(os.path.join(OUTPUT_DIR, f"{base}_full.jpg"), quality=95)

    # 2) Square crops: crop in original, then resize to TARGET_SIZE_CROP
    for name, cx_f, cy_f, frac in CROPS:
        cx = int(cx_f * W)
        cy = int(cy_f * H)
        sq = crop_square_by_center(img, cx, cy, frac)
        save_crop(img, sq, f"{base}_{name}.jpg")

print("Done. Dataset written to:", OUTPUT_DIR)