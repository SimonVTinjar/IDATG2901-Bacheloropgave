#!/usr/bin/env python3
"""
crop_banknote.py – Fjerner alt utenfor seddelen i et bilde.

Bruk:
    python3 crop_banknote.py <input.png> [output.png]

Krever: opencv-python, numpy (pip install opencv-python numpy)
"""

import sys
import cv2
import numpy as np

def crop_banknote(input_path: str, output_path: str) -> None:
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Finner ikke bilde: {input_path}")

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── Steg 1: finn seddelprofilen via Canny + konvekst skall ──────────────
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 10, 50)
    kernel  = np.ones((3, 3), np.uint8)
    edges   = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours     = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
    all_pts      = np.vstack(contours)
    hull         = cv2.convexHull(all_pts)

    rect = cv2.minAreaRect(hull)
    (cx, cy), (rw, rh), angle = rect

    # Sørg for at rw er den lange aksen (liggende orientering)
    if rw < rh:
        rw, rh  = rh, rw
        angle  += 90

    # ── Steg 2: rett ut evt. skjevhet og beskjær ────────────────────────────
    M       = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
                              flags=cv2.INTER_LANCZOS4,
                              borderMode=cv2.BORDER_REPLICATE)

    pad = 2
    x1 = max(0, int(cx - rw / 2) - pad)
    y1 = max(0, int(cy - rh / 2) - pad)
    x2 = min(w, int(cx + rw / 2) + pad)
    y2 = min(h, int(cy + rh / 2) + pad)
    cropped = rotated[y1:y2, x1:x2]

    # ── Steg 3: trim resterende lys bakgrunn (rad/kolonne-gjennomsnitt) ─────
    gray2      = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    row_means  = gray2.mean(axis=1)
    col_means  = gray2.mean(axis=0)
    h2, w2     = cropped.shape[:2]
    THRESHOLD  = 178          # piksler lysere enn dette regnes som bakgrunn

    def first_below(arr, thr):
        for i, v in enumerate(arr):
            if v < thr:
                return i
        return 0

    def last_below(arr, thr):
        for i in range(len(arr) - 1, -1, -1):
            if arr[i] < thr:
                return i
        return len(arr) - 1

    top   = first_below(row_means,  THRESHOLD)
    bot   = last_below(row_means,   THRESHOLD)
    left  = first_below(col_means,  THRESHOLD)
    right = last_below(col_means,   THRESHOLD)

    result = cropped[top:bot + 1, left:right + 1]

    cv2.imwrite(output_path, result)
    print(f"Ferdig!  {w}×{h}  →  {result.shape[1]}×{result.shape[0]}")
    print(f"Lagret:  {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Bruk: python3 crop_banknote.py <input.png> [output.png]")
        sys.exit(1)

    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else inp.rsplit(".", 1)[0] + "_cropped.png"
    crop_banknote(inp, out)
