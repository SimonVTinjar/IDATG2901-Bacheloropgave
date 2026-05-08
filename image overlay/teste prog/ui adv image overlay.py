import cv2
import numpy as np
import json
import os


# =========================
# FILER
# =========================
TEMPLATE_FILE = "template.png"
SCENE_FILE = "scene.png"
INSERT_FILE = "insert.png"

RESULT_FILE = "resultat.png"
DEBUG_FOUND_FILE = "debug_found_bill.png"
DEBUG_FLAT_FILE = "debug_flat_bill.png"
DEBUG_FLAT_INSERT_FILE = "debug_flat_with_insert.png"
SETTINGS_FILE = "ui_settings.json"

WINDOW_NAME = "UI"


# =========================
# FAST STØRRELSE PÅ FLAT SEDDEL
# =========================
FLAT_WIDTH = 1200
FLAT_HEIGHT = 500


# =========================
# STANDARDVERDIER
# =========================
DEFAULT_SETTINGS = {
    "x": 350,
    "y": 120,
    "width": 220,
    "height": 220,
    "rotation": 0,
    "show_scene": 1
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
            print(f"Lastet innstillinger fra {SETTINGS_FILE}")
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


def order_points(pts):
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left

    return rect


def find_bill_with_sift(template, scene):
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(template_gray, None)
    kp2, des2 = sift.detectAndCompute(scene_gray, None)

    if des1 is None or des2 is None:
        print("Fant ikke nok features.")
        return None

    flann_index_kdtree = 1
    index_params = dict(algorithm=flann_index_kdtree, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.7 * n.distance:
                good.append(m)

    print("Gode matcher:", len(good))

    if len(good) < 10:
        print("For få gode matcher.")
        return None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None:
        print("Fant ikke homography.")
        return None

    h_t, w_t = template.shape[:2]
    corners = np.float32([
        [0, 0],
        [w_t - 1, 0],
        [w_t - 1, h_t - 1],
        [0, h_t - 1]
    ]).reshape(-1, 1, 2)

    projected = cv2.perspectiveTransform(corners, H).reshape(4, 2)
    rect = order_points(projected)

    return rect


def warp_bill_to_flat(scene, bill_rect, flat_width, flat_height):
    dst = np.float32([
        [0, 0],
        [flat_width - 1, 0],
        [flat_width - 1, flat_height - 1],
        [0, flat_height - 1]
    ])

    M = cv2.getPerspectiveTransform(bill_rect.astype(np.float32), dst)
    flat_bill = cv2.warpPerspective(scene, M, (flat_width, flat_height))
    return flat_bill, M


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

    if len(image.shape) == 3 and image.shape[2] == 4:
        border_value = (0, 0, 0, 0)
    else:
        border_value = (0, 0, 0)

    rotated = cv2.warpAffine(
        image,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value
    )

    return rotated


def overlay_png_on_flat(flat_bill, insert_img, x, y, w, h, angle):
    result = flat_bill.copy()

    if w < 1 or h < 1:
        return result

    resized = cv2.resize(insert_img, (w, h), interpolation=cv2.INTER_AREA)
    rotated = rotate_image_keep_alpha(resized, angle)

    rh, rw = rotated.shape[:2]

    if x < 0 or y < 0 or x + rw > flat_bill.shape[1] or y + rh > flat_bill.shape[0]:
        return result

    if len(rotated.shape) == 3 and rotated.shape[2] == 4:
        overlay_rgb = rotated[:, :, :3]
        alpha = rotated[:, :, 3] / 255.0
    else:
        overlay_rgb = rotated
        alpha = np.ones((rh, rw), dtype=np.float32)

    roi = result[y:y+rh, x:x+rw].astype(np.float32)
    overlay_rgb = overlay_rgb.astype(np.float32)

    alpha_3 = np.dstack([alpha, alpha, alpha])
    blended = overlay_rgb * alpha_3 + roi * (1.0 - alpha_3)

    result[y:y+rh, x:x+rw] = blended.astype(np.uint8)
    return result


def warp_flat_back_to_scene(scene, flat_with_insert, bill_rect, flat_width, flat_height):
    src = np.float32([
        [0, 0],
        [flat_width - 1, 0],
        [flat_width - 1, flat_height - 1],
        [0, flat_height - 1]
    ])

    M_back = cv2.getPerspectiveTransform(src, bill_rect.astype(np.float32))

    warped_back = cv2.warpPerspective(
        flat_with_insert,
        M_back,
        (scene.shape[1], scene.shape[0])
    )

    mask = np.zeros((flat_height, flat_width), dtype=np.uint8)
    cv2.rectangle(mask, (0, 0), (flat_width - 1, flat_height - 1), 255, -1)

    warped_mask = cv2.warpPerspective(
        mask,
        M_back,
        (scene.shape[1], scene.shape[0])
    )

    warped_mask = cv2.GaussianBlur(warped_mask, (7, 7), 0)

    alpha = warped_mask.astype(np.float32) / 255.0
    alpha_3 = np.dstack([alpha, alpha, alpha])

    scene_f = scene.astype(np.float32)
    warped_f = warped_back.astype(np.float32)

    result = warped_f * alpha_3 + scene_f * (1.0 - alpha_3)
    return result.astype(np.uint8)


def nothing(x):
    pass


# =========================
# LAST INN SISTE UI-VERDIER
# =========================
settings = load_settings()


# =========================
# LAST INN BILDER
# =========================
template = cv2.imread(TEMPLATE_FILE)
scene = cv2.imread(SCENE_FILE)
insert_img = cv2.imread(INSERT_FILE, cv2.IMREAD_UNCHANGED)

if template is None:
    print(f"Fant ikke {TEMPLATE_FILE}")
    exit()

if scene is None:
    print(f"Fant ikke {SCENE_FILE}")
    exit()

if insert_img is None:
    print(f"Fant ikke {INSERT_FILE}")
    exit()


# =========================
# FINN SEDDEL
# =========================
bill_rect = find_bill_with_sift(template, scene)

if bill_rect is None:
    print("Klarte ikke å finne seddelen.")
    exit()

print("Fant hjørner:")
print(bill_rect)

debug_found = scene.copy()
cv2.polylines(debug_found, [bill_rect.astype(np.int32)], True, (0, 255, 0), 4)
cv2.imwrite(DEBUG_FOUND_FILE, debug_found)

flat_bill, _ = warp_bill_to_flat(scene, bill_rect, FLAT_WIDTH, FLAT_HEIGHT)
cv2.imwrite(DEBUG_FLAT_FILE, flat_bill)


# =========================
# LAG UI
# =========================
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1400, 900)

cv2.createTrackbar("X", WINDOW_NAME, settings["x"], FLAT_WIDTH, nothing)
cv2.createTrackbar("Y", WINDOW_NAME, settings["y"], FLAT_HEIGHT, nothing)
cv2.createTrackbar("Width", WINDOW_NAME, settings["width"], FLAT_WIDTH, nothing)
cv2.createTrackbar("Height", WINDOW_NAME, settings["height"], FLAT_HEIGHT, nothing)
cv2.createTrackbar("Rotation", WINDOW_NAME, settings["rotation"], 360, nothing)
cv2.createTrackbar("Show Scene", WINDOW_NAME, settings["show_scene"], 1, nothing)

last_result = None
last_flat = None
last_x = settings["x"]
last_y = settings["y"]
last_w = settings["width"]
last_h = settings["height"]
last_rotation = settings["rotation"]
last_show_scene = settings["show_scene"]

print("\nKontroller:")
print(" - Flytt sliders for å justere bildet")
print(" - Trykk S for å lagre bilde + innstillinger")
print(" - Trykk Q eller ESC for å avslutte")
print(" - Innstillinger huskes automatisk\n")

while True:
    x = cv2.getTrackbarPos("X", WINDOW_NAME)
    y = cv2.getTrackbarPos("Y", WINDOW_NAME)
    w = cv2.getTrackbarPos("Width", WINDOW_NAME)
    h = cv2.getTrackbarPos("Height", WINDOW_NAME)
    rotation = cv2.getTrackbarPos("Rotation", WINDOW_NAME)
    show_scene = cv2.getTrackbarPos("Show Scene", WINDOW_NAME)

    if w < 1:
        w = 1
    if h < 1:
        h = 1

    preview_flat = overlay_png_on_flat(flat_bill, insert_img, x, y, w, h, rotation)

    result_scene = warp_flat_back_to_scene(
        scene,
        preview_flat,
        bill_rect,
        FLAT_WIDTH,
        FLAT_HEIGHT
    )

    last_result = result_scene
    last_flat = preview_flat
    last_x = x
    last_y = y
    last_w = w
    last_h = h
    last_rotation = rotation
    last_show_scene = show_scene

    if show_scene == 1:
        cv2.imshow(WINDOW_NAME, result_scene)
    else:
        cv2.imshow(WINDOW_NAME, preview_flat)

    key = cv2.waitKey(30) & 0xFF

    if key == ord("s"):
        cv2.imwrite(RESULT_FILE, last_result)
        cv2.imwrite(DEBUG_FLAT_INSERT_FILE, last_flat)

        current_settings = {
            "x": last_x,
            "y": last_y,
            "width": last_w,
            "height": last_h,
            "rotation": last_rotation,
            "show_scene": last_show_scene
        }
        save_settings(current_settings)

        print("Lagret:")
        print(f" - {RESULT_FILE}")
        print(f" - {DEBUG_FLAT_INSERT_FILE}")
        print("Verdier:")
        print(f"INSERT_X = {last_x}")
        print(f"INSERT_Y = {last_y}")
        print(f"INSERT_W = {last_w}")
        print(f"INSERT_H = {last_h}")
        print(f"INSERT_ROTATION = {last_rotation}")

    elif key == ord("q") or key == 27:
        current_settings = {
            "x": last_x,
            "y": last_y,
            "width": last_w,
            "height": last_h,
            "rotation": last_rotation,
            "show_scene": last_show_scene
        }
        save_settings(current_settings)
        print("Avslutter.")
        break

cv2.destroyAllWindows()