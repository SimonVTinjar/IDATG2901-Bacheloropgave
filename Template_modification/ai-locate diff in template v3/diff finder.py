from pathlib import Path
from PIL import Image
import numpy as np
import cv2

BASE_DIR = Path(__file__).parent

ORIGINAL_PATH = "image.png"
TEMPLATE_PATH = "Prob Done v2.png"

OUTPUT_MASK = BASE_DIR / "clean_diff_mask.png"
OUTPUT_BOXES = BASE_DIR / "clean_diff_boxes.png"

THRESHOLD = 1
MIN_AREA = 100

original = Image.open(ORIGINAL_PATH).convert("RGB")
template = Image.open(TEMPLATE_PATH).convert("RGB")

# Gjør samme størrelse
template = template.resize(original.size)

orig = np.array(original)
temp = np.array(template)

# Gråskala
orig_gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
temp_gray = cv2.cvtColor(temp, cv2.COLOR_RGB2GRAY)

# Blur for å ignorere små tekstur/fargeforskjeller
orig_gray = cv2.GaussianBlur(orig_gray, (7, 7), 0)
temp_gray = cv2.GaussianBlur(temp_gray, (7, 7), 0)

# Diff
diff = cv2.absdiff(orig_gray, temp_gray)

# Threshold
_, mask = cv2.threshold(diff, THRESHOLD, 255, cv2.THRESH_BINARY)

# Rydd støy
kernel = np.ones((15, 15), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Finn bokser
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

vis = temp.copy()

for i in range(1, num_labels):
    x, y, w, h, area = stats[i]

    if area < MIN_AREA:
        continue

    cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 3)
    print("Forskjell:", [x, y, x + w, y + h], "area:", area)

Image.fromarray(mask).save(OUTPUT_MASK)
Image.fromarray(vis).save(OUTPUT_BOXES)

print("Lagret:", OUTPUT_MASK)
print("Lagret:", OUTPUT_BOXES)