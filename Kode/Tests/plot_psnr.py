import json
import os
import matplotlib.pyplot as plt



TITLE_FONT_SIZE = 24
LABEL_FONT_SIZE = 20
TICK_FONT_SIZE = 20



folder = "PSNR"

ordered_files = [
    "psnr_lora_usd2",
    "psnr_lora_usd3",
    "psnr_lora_thai4",
    "psnr_nanobanana",
    "psnr_textdiffuse",
    "psnr_random_runs_results_prompttesting",
    "psnr_uldm_256",
    "psnr_uldm_384x192",
    "psnr_uldm_384x160",
    "psnr_uldm_1152x480",
]

label_map = {
    "psnr_lora_usd2": "Lora USD\nTrening 2",
    "psnr_lora_usd3": "Lora USD\nTrening 3",
    "psnr_lora_thai4": "Lora Thai\nTrening 4",
    "psnr_nanobanana": "NanoBanana Pro",
    "psnr_textdiffuse": "TextDiffuser",
    "psnr_random_runs_results_prompttesting": "Generert datasett \nPrompttesting",
    "psnr_uldm_256": "ULDM\n256x256",
    "psnr_uldm_384x192": "ULDM\n384x192",
    "psnr_uldm_384x160": "ULDM\n384x160",
    "psnr_uldm_1152x480": "ULDM\n1152x480",
}

all_scores = []
labels = []


for filename in ordered_files:

    file_path = os.path.join(folder, filename + ".json")

    if not os.path.exists(file_path):
        print(f"Fant ikke fil: {file_path}")
        continue

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scores = []

    # hent ut alle PSNR-verdier fra pairs
    for run in data["runs"]:
        for pair in run["pairs"]:
            scores.append(pair["psnr"])

    if len(scores) == 0:
        print(f"Ingen PSNR-scores i: {file_path}")
        continue

    all_scores.append(scores)

    label = label_map.get(filename, filename)
    labels.append(label)



plt.figure(figsize=(20, 10))

plt.boxplot(all_scores, patch_artist=True)

plt.xticks(
    range(1, len(labels) + 1),
    labels,
    rotation=15,
    ha='center',
    fontsize=TICK_FONT_SIZE
)

plt.yticks(fontsize=TICK_FONT_SIZE)

plt.ylabel(
    "PSNR score",
    fontsize=LABEL_FONT_SIZE
)

plt.title(
    "PSNR sammenligning av ulike metoder",
    fontsize=TITLE_FONT_SIZE
)

plt.grid(True, axis='y', alpha=0.3)

plt.tight_layout()

output_path = os.path.join(folder, "psnr_comparison.pdf")

plt.savefig(output_path, bbox_inches='tight')

plt.show()

print(f"Saved plot to: {output_path}")