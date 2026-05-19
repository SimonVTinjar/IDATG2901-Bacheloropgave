#!/usr/bin/env python3
"""
Normaliser bilder til 1344x576 PNG for QuickEval.
Beholder mappestruktur fra rescale_bilder og lagrer i normalized_bilder.

Kjør fra: C:\\Users\\benja\\Desktop\\BachelorOppgaveRepo\\IDATG2901-Bacheloropgave\\
"""

from pathlib import Path
from PIL import Image

# --- Konfigurasjon ---
INPUT_DIR  = Path("rescale_bilder")
OUTPUT_DIR = Path("normalized_bilder")
TARGET_SIZE = (1344, 576)
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
# ---------------------

def normalize(input_path: Path, output_path: Path):
    with Image.open(input_path) as img:
        img = img.convert("RGB")
        img = img.resize(TARGET_SIZE, Image.LANCZOS)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, format="PNG", optimize=True)
        print(f"  ✓  {input_path.relative_to(INPUT_DIR)}  →  {output_path.relative_to(OUTPUT_DIR)}")

def main():
    files = [f for f in INPUT_DIR.rglob("*") if f.suffix.lower() in EXTENSIONS]
    if not files:
        print("Ingen bilder funnet i", INPUT_DIR.resolve())
        return

    print(f"Fant {len(files)} bilder. Konverterer til {TARGET_SIZE[0]}×{TARGET_SIZE[1]} PNG...\n")

    for f in sorted(files):
        relative = f.relative_to(INPUT_DIR)
        output_path = OUTPUT_DIR / relative.parent / (f.stem + ".png")
        normalize(f, output_path)

    print(f"\nFerdig! Bilder lagret i: {OUTPUT_DIR.resolve()}")
    print("\nMappestruktur:")
    for folder in sorted(set(p.parent for p in OUTPUT_DIR.rglob("*.png"))):
        count = len(list(folder.glob("*.png")))
        print(f"  {folder.relative_to(OUTPUT_DIR)}  ({count} bilder)")

if __name__ == "__main__":
    main()