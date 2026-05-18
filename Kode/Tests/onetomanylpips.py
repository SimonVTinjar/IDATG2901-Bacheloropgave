import os
import torch
import lpips
from PIL import Image
from torchvision import transforms


reference_image_path = r"C:\Users\Lcmol\Desktop\IDATG2901-Bacheloropgave\Kode\Scaling\full\24100-E_28202229.jpg"
image_folder = r"C:\Users\Lcmol\Desktop\IDATG2901-Bacheloropgave\Kode\Tests\paired_output"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loss_fn = lpips.LPIPS(net="alex").to(device)

transform = transforms.Compose([
    transforms.Resize((927, 376)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

def load_image(path):
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)

reference_tensor = load_image(reference_image_path)

results = []

valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

for file_name in os.listdir(image_folder):
    if not file_name.lower().endswith(valid_extensions):
        continue

    image_path = os.path.join(image_folder, file_name)

    img_tensor = load_image(image_path)

    with torch.no_grad():
        score = loss_fn(reference_tensor, img_tensor).item()

    results.append((file_name, score))


results.sort(key=lambda x: x[1])

print("\n===== Best LPIPS matches =====")
for file_name, score in results:
    print(f"{file_name}: LPIPS = {score:.4f}")

print("\n=============================")
print(f"Best match: {results[0][0]}")
print(f"Best LPIPS: {results[0][1]:.4f}")
print("=============================")