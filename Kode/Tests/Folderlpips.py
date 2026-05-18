import os
import json
import torch
import lpips
import numpy as np
from PIL import Image
from torchvision import transforms


root_folder = r"C:\Users\Lcmol\Desktop\IDATG2901-Bacheloropgave\Kode\Tests\paired_output"

output_json = "lpips_results_prompttesting.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loss_fn = lpips.LPIPS(net="alex").to(device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

scores = []
runs_data = []

for idx, folder_name in enumerate(sorted(os.listdir(root_folder))):
    folder_path = os.path.join(root_folder, folder_name)

    if not os.path.isdir(folder_path):
        continue

    real_path = None
    gen_path = None

    for file_name in os.listdir(folder_path):
        lower = file_name.lower()

        if "ref" in lower:
            real_path = os.path.join(folder_path, file_name)

        elif "scaled" in lower:
            gen_path = os.path.join(folder_path, file_name)

    if real_path is None or gen_path is None:
        print(f"Skipping {folder_name}: missing scaled or ref image")
        continue

    real_img = Image.open(real_path).convert("RGB")
    gen_img = Image.open(gen_path).convert("RGB")

    real_tensor = transform(real_img).unsqueeze(0).to(device)
    gen_tensor = transform(gen_img).unsqueeze(0).to(device)

    with torch.no_grad():
        score = loss_fn(real_tensor, gen_tensor).item()

    scores.append(float(score))

    # Struktur som plotting-koden forventer
    runs_data.append({
        "run": idx + 1,
        "mean_lpips": float(score),
        "std_lpips": 0.0,
        "scores": [float(score)]
    })

    print(f"{folder_name}: LPIPS = {score:.4f}")

if scores:
    avg_lpips = float(np.mean(scores))
    std_lpips = float(np.std(scores))

    print("\n==============================")
    print(f"Total images compared: {len(scores)}")
    print(f"Average LPIPS: {avg_lpips:.4f}")
    print(f"Standard deviation: {std_lpips:.4f}")
    print(f"Result: {avg_lpips:.4f} ± {std_lpips:.4f}")
    print("==============================")

    # JSON-struktur kompatibel med plotting-scriptet
    results = {
        "root_folder": root_folder,
        "num_pairs_per_run": len(scores),
        "num_runs": len(runs_data),
        "overall_mean_lpips": avg_lpips,
        "overall_std_lpips": std_lpips,
        "runs": runs_data
    }

    # Lagre JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to {output_json}")

else:
    print("No valid image pairs found.")