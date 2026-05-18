from PIL import Image
from pathlib import Path

input_dir = Path("F:\\BachmapperMedBilder\\Datasett\\170BilderDatasettet")          # mappa med dine 172 bilder
output_dir = Path("F:\\BachmapperMedBilder\\Datasett\\170_usd_391x366") # ny mappe
output_dir.mkdir(exist_ok=True)

target_size = (391, 366)  # (width, height)

valid_ext = [".png", ".jpg", ".jpeg"]

for img_path in input_dir.iterdir():
    if img_path.suffix.lower() not in valid_ext:
        continue

    try:
        img = Image.open(img_path).convert("RGB")

        # Resize til eksakt størrelse
        resized = img.resize(target_size, Image.Resampling.LANCZOS)

        # Lagre med samme navn
        output_path = output_dir / img_path.name
        resized.save(output_path)

    except Exception as e:
        print(f"Feil med {img_path.name}: {e}")

print(f"Ferdig! Bilder lagret i: {output_dir}")