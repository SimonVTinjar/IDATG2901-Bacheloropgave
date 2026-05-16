from PIL import Image
from pathlib import Path

input_dir = Path("generated_images384x160")          # mappa med dine 172 bilder
output_dir = Path("scaled_images1152x480") # ny mappe
output_dir.mkdir(exist_ok=True)

target_width = 1152
target_height = 480

for img_path in input_dir.glob("*"):
    if img_path.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
        continue

    try:
        img = Image.open(img_path).convert("RGB")

        # 👉 Bevar ratio: skaler så bildet dekker målformatet
        scale = max(
            target_width / img.width,
            target_height / img.height
        )

        new_size = (
            int(img.width * scale),
            int(img.height * scale)
        )

        img_resized = img.resize(new_size, Image.Resampling.LANCZOS)

        # 👉 Center crop
        left = (img_resized.width - target_width) // 2
        top = (img_resized.height - target_height) // 2
        right = left + target_width
        bottom = top + target_height

        img_cropped = img_resized.crop((left, top, right, bottom))

        # Lagre
        output_path = output_dir / img_path.name
        img_cropped.save(output_path)

    except Exception as e:
        print(f"Feil med {img_path.name}: {e}")

print(f"Ferdig! Lagret i: {output_dir}")