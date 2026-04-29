import torch
import lpips
from PIL import Image
from pathlib import Path
from torchvision import transforms

real_dir = Path("full_clean")          # ekte bilder
generated_dir = Path("generated_images")  # genererte bilder

device = "cuda" if torch.cuda.is_available() else "cpu"

loss_fn = lpips.LPIPS(net="alex").to(device)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # LPIPS forventer [-1, 1]
])

real_images = sorted(list(real_dir.glob("*.png")) + list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.jpeg")))
gen_images = sorted(list(generated_dir.glob("*.png")) + list(generated_dir.glob("*.jpg")) + list(generated_dir.glob("*.jpeg")))

num_pairs = min(len(real_images), len(gen_images))

scores = []

for i in range(num_pairs):
    real = Image.open(real_images[i]).convert("RGB")
    gen = Image.open(gen_images[i]).convert("RGB")

    real_tensor = transform(real).unsqueeze(0).to(device)
    gen_tensor = transform(gen).unsqueeze(0).to(device)

    with torch.no_grad():
        score = loss_fn(real_tensor, gen_tensor)

    scores.append(score.item())

mean_lpips = sum(scores) / len(scores)

print(f"Antall sammenligninger: {num_pairs}")
print(f"Gjennomsnittlig LPIPS: {mean_lpips:.4f}")