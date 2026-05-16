import os
import shutil
from PIL import Image

# =========================
# SETTINGS
# =========================

# Folder containing the 500 images
source_folder = r"f:\GenBilde\Scaleddataset"

# Single reference image
reference_image_path = r"C:\Users\Lcmol\Desktop\IDATG2901-Bacheloropgave\Kode\Scaling\full\24100-E_28202229.jpg"

# Output folder
output_folder = r"C:\Users\Lcmol\Desktop\paired_output"

# =========================
# CREATE OUTPUT FOLDER
# =========================

os.makedirs(output_folder, exist_ok=True)

# Supported image formats
valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

# Load reference image once
reference_image = Image.open(reference_image_path).convert("RGB")

# =========================
# PROCESS ALL IMAGES
# =========================

for idx, file_name in enumerate(sorted(os.listdir(source_folder)), start=1):

    if not file_name.lower().endswith(valid_extensions):
        continue

    image_path = os.path.join(source_folder, file_name)

    # Open target image
    target_image = Image.open(image_path).convert("RGB")

    # Get target size
    width, height = target_image.size

    # Resize reference image to match target image
    resized_reference = reference_image.resize((width, height), Image.LANCZOS)

    # Create pair folder
    pair_folder = os.path.join(output_folder, f"pair_{idx:04d}")
    os.makedirs(pair_folder, exist_ok=True)

    # Save original target image
    target_save_path = os.path.join(pair_folder, "scaled.png")
    target_image.save(target_save_path)

    # Save resized reference image
    reference_save_path = os.path.join(pair_folder, "ref.png")
    resized_reference.save(reference_save_path)

    print(f"Created pair_{idx:04d} ({width}x{height})")

print("\nDone!")
print(f"All image pairs saved to:\n{output_folder}")