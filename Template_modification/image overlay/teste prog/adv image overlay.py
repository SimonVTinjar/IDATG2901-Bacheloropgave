import cv2
import numpy as np


# =========================
# FILNAVN
# =========================
TEMPLATE_FILE = "template.png"
SCENE_FILE = "scene.png"
INSERT_FILE = "insert.png"

RESULT_FILE = "resultat.png"
DEBUG_FOUND_FILE = "debug_found_bill.png"
DEBUG_FLAT_FILE = "debug_flat_bill.png"
DEBUG_FLAT_INSERT_FILE = "debug_flat_with_insert.png"


# =========================
# FAST STØRRELSE PÅ "FLAT" SEDDEL
# =========================
FLAT_WIDTH = 1200
FLAT_HEIGHT = 500

# Plassering av bildet i den flate seddelen
INSERT_X = 350
INSERT_Y = 120
INSERT_W = 220
INSERT_H = 220


# =========================
# HJELPEFUNKSJONER
# =========================
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


def overlay_png_on_flat(flat_bill, insert_img, x, y, w, h):
    result = flat_bill.copy()

    resized = cv2.resize(insert_img, (w, h), interpolation=cv2.INTER_AREA)

    if len(resized.shape) == 3 and resized.shape[2] == 4:
        overlay_rgb = resized[:, :, :3]
        alpha = resized[:, :, 3] / 255.0
    else:
        overlay_rgb = resized
        alpha = np.ones((h, w), dtype=np.float32)

    # Sjekk at området er innenfor
    if x < 0 or y < 0 or x + w > flat_bill.shape[1] or y + h > flat_bill.shape[0]:
        print("Bildet havner utenfor seddelen.")
        return result

    roi = result[y:y+h, x:x+w].astype(np.float32)
    overlay_rgb = overlay_rgb.astype(np.float32)

    alpha_3 = np.dstack([alpha, alpha, alpha])

    blended = overlay_rgb * alpha_3 + roi * (1.0 - alpha_3)
    result[y:y+h, x:x+w] = blended.astype(np.uint8)

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


# =========================
# HOVEDPROGRAM
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

bill_rect = find_bill_with_sift(template, scene)

if bill_rect is None:
    print("Klarte ikke å finne seddelen.")
    exit()

print("Fant hjørner:")
print(bill_rect)

# Debug: tegn funnet seddel
debug_found = scene.copy()
cv2.polylines(debug_found, [bill_rect.astype(np.int32)], True, (0, 255, 0), 4)
cv2.imwrite(DEBUG_FOUND_FILE, debug_found)

# 1. Rett opp seddelen
flat_bill, _ = warp_bill_to_flat(scene, bill_rect, FLAT_WIDTH, FLAT_HEIGHT)
cv2.imwrite(DEBUG_FLAT_FILE, flat_bill)

# 2. Legg inn bildet på flat seddel
flat_with_insert = overlay_png_on_flat(
    flat_bill,
    insert_img,
    INSERT_X,
    INSERT_Y,
    INSERT_W,
    INSERT_H
)
cv2.imwrite(DEBUG_FLAT_INSERT_FILE, flat_with_insert)

# 3. Warp tilbake til originalbildet
result = warp_flat_back_to_scene(
    scene,
    flat_with_insert,
    bill_rect,
    FLAT_WIDTH,
    FLAT_HEIGHT
)

cv2.imwrite(RESULT_FILE, result)

print(f"Ferdig! Lagret som {RESULT_FILE}")
print(f"Debug lagret som {DEBUG_FOUND_FILE}")
print(f"Flat seddel lagret som {DEBUG_FLAT_FILE}")
print(f"Flat seddel med bilde lagret som {DEBUG_FLAT_INSERT_FILE}")