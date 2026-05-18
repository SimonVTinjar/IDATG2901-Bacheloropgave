import cv2
import numpy as np
import json
import os

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_1_FILE = os.path.join(BASE_DIR, "image.png")
IMAGE_2_FILE = os.path.join(BASE_DIR, "scene.png")
RESULT_FILE = os.path.join(BASE_DIR, "resultat.png")
SETTINGS_FILE = os.path.join(BASE_DIR, "clip_ui_settings.json")
CLIP_FILE = os.path.join(BASE_DIR, "utklipp.png")

def load_settings():
    if not os.path.exists(SETTINGS_FILE):
        print(f"Fant ikke {SETTINGS_FILE}")
        raise SystemExit

    with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


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
        print("Det innlimte bildet havner utenfor bakgrunnsbildet.")
        return result

    overlay_rgb = rotated[:, :, :3]
    alpha = rotated[:, :, 3].astype(np.float32) / 255.0

    roi = result[y:y+rh, x:x+rw].astype(np.float32)
    overlay_rgb = overlay_rgb.astype(np.float32)

    alpha_3 = np.dstack([alpha, alpha, alpha])
    blended = overlay_rgb * alpha_3 + roi * (1.0 - alpha_3)

    result[y:y+rh, x:x+rw] = blended.astype(np.uint8)
    return result


image2 = cv2.imread(IMAGE_2_FILE, cv2.IMREAD_COLOR)
clip = cv2.imread(CLIP_FILE, cv2.IMREAD_UNCHANGED)

if image2 is None:
    print(f"Fant ikke {IMAGE_2_FILE}")
    raise SystemExit

if clip is None:
    print(f"Fant ikke {CLIP_FILE}")
    raise SystemExit

clip = add_alpha_if_missing(clip)
settings = load_settings()

x = int(settings.get("x", 100))
y = int(settings.get("y", 100))
w = int(settings.get("width", 200))
h = int(settings.get("height", 200))
rotation = float(settings.get("rotation", 0.0))

result = overlay_png(image2, clip, x, y, w, h, rotation)
cv2.imwrite(RESULT_FILE, result)

print(f"Lagret som {RESULT_FILE}")
print("Brukte innstillinger:")
print(f"X = {x}")
print(f"Y = {y}")
print(f"WIDTH = {w}")
print(f"HEIGHT = {h}")
print(f"ROTATION = {rotation}")