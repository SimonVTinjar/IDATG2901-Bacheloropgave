from PIL import Image
from pathlib import Path

input_dir = Path("Textdiffuser")
output_dir = Path("output_1024x512")
output_dir.mkdir(exist_ok=True)

target_size = (1024, 512)

methods = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}

for img_path in input_dir.glob("*"):
    if img_path.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
        continue

    img = Image.open(img_path).convert("RGB")

    for method_name, method in methods.items():
        resized = img.resize(target_size, method)

        # 👉 legg til prefix
        new_name = f"{method_name}_{img_path.name}"
        output_path = output_dir / new_name

        resized.save(output_path)

print("Ferdig!")