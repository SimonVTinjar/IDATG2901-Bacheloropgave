import json
from pathlib import Path
import torch
import lpips
import numpy as np
from PIL import Image
from torchvision import transforms

real_dir = Path("F:\\BachmapperMedBilder\\Datasett\\170_USD_384x160")
generated_dir = Path("F:\\BachmapperMedBilder\\Generert\\ULDM\\generated_images1152x480")
output_json = Path("lpips_resultsasdas.json")

target_size = (160, 384)  # height, width
device = "cuda" if torch.cuda.is_available() else "cpu"

loss_fn = lpips.LPIPS(net="alex").to(device)

transform = transforms.Compose([
    transforms.Resize(target_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def get_images(folder):
    valid_ext = {".png", ".jpg", ".jpeg", ".webp"}
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in valid_ext])

def image_to_tensor(path):
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)

real_images = get_images(real_dir)
gen_images = get_images(generated_dir)

if len(real_images) == 0 or len(gen_images) == 0:
    raise ValueError("Fant ingen bilder. Sjekk mappenavn og filtyper.")

scores = []
per_generated = {}

for gen_path in gen_images:
    gen_tensor = image_to_tensor(gen_path)
    gen_scores = []
    for real_path in real_images:
        with torch.no_grad():
            score = loss_fn(gen_tensor, image_to_tensor(real_path)).item()
        gen_scores.append(float(score))
    scores.extend(gen_scores)
    per_generated[gen_path.name] = {
        "mean_lpips": float(np.mean(gen_scores)),
        "std_lpips": float(np.std(gen_scores, ddof=1))
    }

result = {
    "real_dir": str(real_dir),
    "generated_dir": str(generated_dir),
    "num_real": len(real_images),
    "num_generated": len(gen_images),
    "num_pairs": len(scores),
    "mean_lpips": float(np.mean(scores)),
    "std_lpips": float(np.std(scores, ddof=1)),
    "per_generated": per_generated
}

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=4)

print(f"\nLPIPS: {result['mean_lpips']:.4f} ± {result['std_lpips']:.4f}")
print(f"Antall par: {result['num_pairs']}")
print(f"Lagret til: {output_json}")

best_gen = min(per_generated.items(), key=lambda x: x[1]["mean_lpips"])
worst_gen = max(per_generated.items(), key=lambda x: x[1]["mean_lpips"])

print(f"\nBeste genererte bilde:  {best_gen[0]} (LPIPS: {best_gen[1]['mean_lpips']:.4f})")
print(f"Dårligste genererte bilde: {worst_gen[0]} (LPIPS: {worst_gen[1]['mean_lpips']:.4f})")

print("\nAlle genererte bilder sortert:")
for name, data in sorted(per_generated.items(), key=lambda x: x[1]["mean_lpips"]):
    print(f"  {name}: {data['mean_lpips']:.4f} ± {data['std_lpips']:.4f}")