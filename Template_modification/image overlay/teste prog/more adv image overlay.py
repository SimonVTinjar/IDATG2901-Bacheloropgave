import cv2
import numpy as np
import json
import os


# =========================
# FILNAVN
# =========================
IMAGE_1_FILE = "image.png"   # bildet du klipper fra
IMAGE_2_FILE = "scene.png"   # bildet du limer inn på
RESULT_FILE = "resultat.png"
SETTINGS_FILE = "clip_ui_settings.json"

WINDOW_NAME = "Plassering UI"


# =========================
# STANDARDVERDIER
# =========================
DEFAULT_SETTINGS = {
    "x": 100,
    "y": 100,
    "width": 200,
    "height": 200,
    "rotation": 0.0
}


# =========================
# HJELPEFUNKSJONER
# =========================
def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            settings = DEFAULT_SETTINGS.copy()
            settings.update(data)
            print(f"Lastet inn innstillinger fra {SETTINGS_FILE}")
            return settings
        except Exception as e:
            print(f"Kunne ikke lese {SETTINGS_FILE}: {e}")

    return DEFAULT_SETTINGS.copy()


def save_settings(settings):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4)
        print(f"Innstillinger lagret i {SETTINGS_FILE}")
    except Exception as e:
        print(f"Kunne ikke lagre {SETTINGS_FILE}: {e}")


def nothing(x):
    pass


def add_alpha_if_missing(image):
    """
    Gjør BGR om til BGRA hvis bildet ikke allerede har alpha.
    """
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

    border_value = (0, 0, 0, 0)

    rotated = cv2.warpAffine(
        image,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value
    )

    return rotated


def overlay_png(background, overlay, x, y, w, h, angle):
    result = background.copy()

    if w < 1 or h < 1:
        return result

    resized = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_AREA)
    rotated = rotate_image_keep_alpha(resized, angle)

    rh, rw = rotated.shape[:2]

    # Ikke legg inn hvis bildet havner utenfor
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


def clamp_rotation(value):
    if value < -360.0:
        return -360.0
    if value > 360.0:
        return 360.0
    return value


# =========================
# LAST INN BILDER
# =========================
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


# =========================
# STEG 1: VELG OMRÅDE FRA BILDE 1
# =========================
print("\nSteg 1:")
print("Marker området du vil klippe ut fra bilde 1.")
print("Trykk ENTER eller SPACE for å godkjenne.")
print("Trykk C eller ESC for å avbryte.\n")

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

cv2.imwrite("utklipp.png", clip)
print("Utklipp lagret som utklipp.png")


# =========================
# STEG 2: PLASSER UTKLIPPET PÅ BILDE 2
# =========================
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1400, 900)

max_w = image2.shape[1]
max_h = image2.shape[0]

start_w = max(1, min(int(settings["width"]), max_w))
start_h = max(1, min(int(settings["height"]), max_h))
start_x = max(0, min(int(settings["x"]), max_w - 1))
start_y = max(0, min(int(settings["y"]), max_h - 1))

# Rotasjon lagres som float, slider lagrer x10 for 0.1 presisjon
start_rotation = float(settings["rotation"])
start_rotation_slider = int(round(clamp_rotation(start_rotation) * 10)) + 3600

cv2.createTrackbar("X", WINDOW_NAME, start_x, max_w, nothing)
cv2.createTrackbar("Y", WINDOW_NAME, start_y, max_h, nothing)
cv2.createTrackbar("Width", WINDOW_NAME, start_w, max_w, nothing)
cv2.createTrackbar("Height", WINDOW_NAME, start_h, max_h, nothing)
cv2.createTrackbar("Rotation x10", WINDOW_NAME, start_rotation_slider, 7200, nothing)

last_result = None
last_x = start_x
last_y = start_y
last_w = start_w
last_h = start_h
last_rotation = start_rotation

print("\nSteg 2:")
print("Juster plassering, størrelse og rotasjon.")
print("Taster:")
print(" - S = lagre")
print(" - Q eller ESC = avslutte")
print(" - A / D = roter -/+ 1.0 grad")
print(" - Z / C = roter -/+ 0.1 grad")
print(" - R = skriv inn eksakt rotasjon manuelt\n")

while True:
    x = cv2.getTrackbarPos("X", WINDOW_NAME)
    y = cv2.getTrackbarPos("Y", WINDOW_NAME)
    w = cv2.getTrackbarPos("Width", WINDOW_NAME)
    h = cv2.getTrackbarPos("Height", WINDOW_NAME)

    rotation_slider = cv2.getTrackbarPos("Rotation x10", WINDOW_NAME)
    rotation = (rotation_slider - 3600) / 10.0

    if w < 1:
        w = 1
    if h < 1:
        h = 1

    preview = overlay_png(image2, clip, x, y, w, h, rotation)

    last_result = preview
    last_x = x
    last_y = y
    last_w = w
    last_h = h
    last_rotation = rotation

    cv2.imshow(WINDOW_NAME, preview)

    key = cv2.waitKey(30) & 0xFF

    if key == ord("a"):
        rotation = clamp_rotation(rotation - 1.0)
        cv2.setTrackbarPos("Rotation x10", WINDOW_NAME, int(round(rotation * 10)) + 3600)

    elif key == ord("d"):
        rotation = clamp_rotation(rotation + 1.0)
        cv2.setTrackbarPos("Rotation x10", WINDOW_NAME, int(round(rotation * 10)) + 3600)

    elif key == ord("z"):
        rotation = clamp_rotation(rotation - 0.1)
        cv2.setTrackbarPos("Rotation x10", WINDOW_NAME, int(round(rotation * 10)) + 3600)

    elif key == ord("c"):
        rotation = clamp_rotation(rotation + 0.1)
        cv2.setTrackbarPos("Rotation x10", WINDOW_NAME, int(round(rotation * 10)) + 3600)

    elif key == ord("r"):
        try:
            user_value = input("Skriv inn rotasjon i grader (f.eks. 12.5 eller -3.2): ").strip()
            rotation = clamp_rotation(float(user_value))
            cv2.setTrackbarPos("Rotation x10", WINDOW_NAME, int(round(rotation * 10)) + 3600)
        except ValueError:
            print("Ugyldig tall.")

    elif key == ord("s"):
        cv2.imwrite(RESULT_FILE, last_result)

        current_settings = {
            "x": last_x,
            "y": last_y,
            "width": last_w,
            "height": last_h,
            "rotation": last_rotation
        }
        save_settings(current_settings)

        print("\nLagret:")
        print(f" - {RESULT_FILE}")
        print("Siste verdier:")
        print(f"X = {last_x}")
        print(f"Y = {last_y}")
        print(f"WIDTH = {last_w}")
        print(f"HEIGHT = {last_h}")
        print(f"ROTATION = {last_rotation:.1f}")

    elif key == ord("q") or key == 27:
        current_settings = {
            "x": last_x,
            "y": last_y,
            "width": last_w,
            "height": last_h,
            "rotation": last_rotation
        }
        save_settings(current_settings)
        print("Avslutter.")
        break

cv2.destroyAllWindows()