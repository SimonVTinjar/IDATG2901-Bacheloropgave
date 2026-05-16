import os
import json
import random
import torch
import lpips
import numpy as np
from PIL import Image
from torchvision import transforms

real_dir = r"C:\Users\Lcmol\Desktop\IDATG2901-Bacheloropgave\Kode\full"
generated_dir = r"C:\Users\Lcmol\Desktop\IDATG2901-Bacheloropgave\Kode\Scaleddataset"

output_json = "lpips_random_results_10x170prompt.json"

num_pairs_per_run = 170
num_runs = 10
seed = 42

# Evalueringsoppløsning
resize_height = 160
resize_width = 384

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loss_fn = lpips.LPIPS(net="alex").to(device)

transform = transforms.Compose([
    transforms.Resize((resize_height, resize_width)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])


def load_image(path):
    image = Image.open(path).convert("RGB")
    image = transform(image).unsqueeze(0)
    return image.to(device)

def get_images(folder):
    valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(valid_ext)
    ])


real_images = get_images(real_dir)
generated_images = get_images(generated_dir)

if len(real_images) < num_pairs_per_run:
    raise ValueError("Ikke nok ekte bilder.")

if len(generated_images) < num_pairs_per_run:
    raise ValueError("Ikke nok genererte bilder.")

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


all_runs = []
run_means = []

for run_idx in range(num_runs):

    selected_real = random.sample(real_images, num_pairs_per_run)
    selected_generated = random.sample(generated_images, num_pairs_per_run)

    scores = []

    for real_path, gen_path in zip(selected_real, selected_generated):

        real_tensor = load_image(real_path)
        gen_tensor = load_image(gen_path)

        with torch.no_grad():
            score = loss_fn(real_tensor, gen_tensor).item()

        scores.append(float(score))

    mean_lpips = float(np.mean(scores))
    std_lpips = float(np.std(scores))

    run_means.append(mean_lpips)

    all_runs.append({
        "run": run_idx + 1,
        "mean_lpips": mean_lpips,
        "std_lpips": std_lpips,
        "scores": scores
    })

    print(f"Run {run_idx+1}: "
          f"mean={mean_lpips:.4f}, "
          f"std={std_lpips:.4f}")
    
    all_scores = []

for run in all_runs:
    all_scores.extend(run["scores"])

overall_mean_all_scores = float(np.mean(all_scores))
overall_std_all_scores = float(np.std(all_scores))

overall_mean_runs = float(np.mean(run_means))
overall_std_runs = float(np.std(run_means))

print("\n==============================")
print("OVERALL RESULTS")
print(f"Mean LPIPS (all scores): {overall_mean_all_scores:.6f}")
print(f"Std LPIPS (all scores):  {overall_std_all_scores:.6f}")
print(f"Mean LPIPS (runs):       {overall_mean_runs:.6f}")
print(f"Std LPIPS (runs):        {overall_std_runs:.6f}")
print("==============================")

results = {
    "real_dir": real_dir,
    "generated_dir": generated_dir,
    "num_pairs_per_run": num_pairs_per_run,
    "num_runs": num_runs,
    "seed": seed,
    "resize": {
        "height": resize_height,
        "width": resize_width
    },
    "overall_mean_lpips": float(np.mean(run_means)),
    "overall_std_lpips": float(np.std(run_means)),
    "runs": all_runs
}

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print(f"\nSaved results to {output_json}")