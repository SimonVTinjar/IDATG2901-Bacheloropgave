from PIL import Image, ImageOps
from pathlib import Path

input_folder = Path(r"F:\thai 20 front")
output_folder = Path(r"C:\Users\Lcmol\Desktop\Trening4-bilder-rotated")
reference_image = Path(r"C:\Users\Lcmol\Desktop\learning4test\reference.jpeg")

output_folder.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"]

def get_orientation(img):
    w, h = img.size
    return "landscape" if w >= h else "portrait"

ref = ImageOps.exif_transpose(Image.open(reference_image))
ref_orientation = get_orientation(ref)

for file in input_folder.iterdir():
    if file.suffix.lower() not in IMAGE_EXTENSIONS:
        continue

    img = ImageOps.exif_transpose(Image.open(file))
    orientation = get_orientation(img)

    # Roter bare hvis bildet har annen orientering enn referansen
    if orientation != ref_orientation:
        img = img.rotate(90, expand=True)

    output_file = output_folder / file.name

    # Lagre JPG riktig hvis bildet har alpha/transparency
    if output_file.suffix.lower() in [".jpg", ".jpeg"]:
        img = img.convert("RGB")

    img.save(output_file)
    print(f"Lagret: {output_file}")

print("Ferdig!")