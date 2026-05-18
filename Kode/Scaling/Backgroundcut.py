from PIL import Image, ImageFilter
from collections import deque
import os

input_folder = r"C:\Users\Lcmol\Desktop\Trening2bilder"
output_folder = r"C:\Users\Lcmol\Desktop\Trening2cut"

os.makedirs(output_folder, exist_ok=True)

# Høyere = fjerner mer hvitt
THRESHOLD = 210

def is_background(pixel):
    r, g, b, a = pixel

    # Hvor nær hvit pikselen er
    brightness = (r + g + b) / 3

    # Krev også at kanalene er ganske like
    return (
        brightness > THRESHOLD
        and abs(r - g) < 20
        and abs(r - b) < 20
        and abs(g - b) < 20
    )

for filename in os.listdir(input_folder):

    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    path = os.path.join(input_folder, filename)

    img = Image.open(path).convert("RGBA")

    # Litt blur hjelper mot støy
    img = img.filter(ImageFilter.GaussianBlur(radius=1))

    pixels = img.load()

    width, height = img.size

    visited = set()
    queue = deque()

    # Start fra kantene
    for x in range(width):
        queue.append((x, 0))
        queue.append((x, height - 1))

    for y in range(height):
        queue.append((0, y))
        queue.append((width - 1, y))

    while queue:
        x, y = queue.popleft()

        if (x, y) in visited:
            continue

        visited.add((x, y))

        if not is_background(pixels[x, y]):
            continue

        pixels[x, y] = (255, 255, 255, 0)

        for nx, ny in [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
        ]:
            if 0 <= nx < width and 0 <= ny < height:
                queue.append((nx, ny))

    output_path = os.path.join(
        output_folder,
        filename.rsplit(".", 1)[0] + ".png"
    )

    img.save(output_path)

print("Ferdig!")