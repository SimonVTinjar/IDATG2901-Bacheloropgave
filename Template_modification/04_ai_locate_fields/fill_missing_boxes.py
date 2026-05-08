from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


BASE_DIR = Path(__file__).parent

IMAGE_PATH = BASE_DIR.parent / "Prob Done v2.png"
OUTPUT_PATH = BASE_DIR.parent / "final_output.png"

# Eksempelbokser (fra AI)
missing_boxes = [
    [100, 200, 300, 250],
    [600, 400, 800, 450]
]

img = Image.open(IMAGE_PATH).convert("RGB")
draw = ImageDraw.Draw(img)

font = ImageFont.load_default()

for i, box in enumerate(missing_boxes):
    x1, y1, x2, y2 = box

    text = input(f"Hva vil du skrive i boks {i+1}? ")

    draw.text((x1 + 5, y1 + 5), text, fill="black", font=font)

img.save(OUTPUT_PATH)
print("Lagret:", OUTPUT_PATH)