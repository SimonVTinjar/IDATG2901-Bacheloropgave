import os
import json
import random
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

generated_folder = r"F:\BachmapperMedBilder\Generert\Textdiffuser\Sammescale"
real_folder = r"F:\BachmapperMedBilder\Datasett\170_usd_391x366"
output_json = "ssim_textdiffuse.json"
target_size = (391, 366)

num_runs = 100
random_seed = 42

valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

random.seed(random_seed)

generated_images = sorted([
    f for f in os.listdir(generated_folder)
    if f.lower().endswith(valid_extensions)
])

real_images = sorted([
    f for f in os.listdir(real_folder)
    if f.lower().endswith(valid_extensions)
])

num_pairs = min(len(generated_images), len(real_images))

all_run_results = []
all_scores = []

for run in range(1, num_runs + 1):
    sampled_generated = random.sample(generated_images, num_pairs)
    sampled_real = random.sample(real_images, num_pairs)

    pair_results = []
    run_scores = []

    print(f"\nRun {run}/{num_runs}")

    for gen_name, real_name in zip(sampled_generated, sampled_real):

        gen_path = os.path.join(generated_folder, gen_name)
        real_path = os.path.join(real_folder, real_name)

        gen_img = Image.open(gen_path).convert("RGB")
        real_img = Image.open(real_path).convert("RGB")

        gen_img = gen_img.resize(target_size, Image.LANCZOS)
        real_img = real_img.resize(target_size, Image.LANCZOS)

        gen_arr = np.array(gen_img)
        real_arr = np.array(real_img)

        score = ssim(
            real_arr,
            gen_arr,
            channel_axis=2,
            data_range=255
        )

        run_scores.append(score)
        all_scores.append(score)

        pair_results.append({
            "generated_image": gen_name,
            "real_image": real_name,
            "ssim": round(float(score), 4)
        })

    run_avg = float(np.mean(run_scores))
    run_std = float(np.std(run_scores))

    all_run_results.append({
        "run": run,
        "average_ssim": round(run_avg, 4),
        "standard_deviation_over_pairs": round(run_std, 4),
        "pairs": pair_results
    })

    print(f"Run average SSIM: {run_avg:.4f}")
    print(f"Run std over pairs: {run_std:.4f}")

overall_avg = float(np.mean(all_scores))
overall_std = float(np.std(all_scores))

json_data = {
    "method": "Random pairing SSIM",
    "num_runs": num_runs,
    "pairs_per_run": num_pairs,
    "total_pair_comparisons": len(all_scores),
    "target_size": {
        "width": target_size[0],
        "height": target_size[1]
    },
    "random_seed": random_seed,
    "generated_folder": generated_folder,
    "real_folder": real_folder,
    "overall_average_ssim": round(overall_avg, 4),
    "overall_standard_deviation_over_all_pairs": round(overall_std, 4),
    "result": f"{overall_avg:.4f} ± {overall_std:.4f}",
    "runs": all_run_results
}

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(json_data, f, indent=4, ensure_ascii=False)

print("\n==============================")
print(f"Random runs: {num_runs}")
print(f"Pairs per run: {num_pairs}")
print(f"Total pair comparisons: {len(all_scores)}")
print(f"Overall average SSIM: {overall_avg:.4f}")
print(f"Overall std over all pairs: {overall_std:.4f}")
print(f"Result: {overall_avg:.4f} ± {overall_std:.4f}")
print(f"Saved JSON to: {output_json}")
print("==============================")