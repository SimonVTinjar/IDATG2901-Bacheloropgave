import json
import os
import matplotlib.pyplot as plt


TITLE_FONT_SIZE = 24
LABEL_FONT_SIZE = 20
TICK_FONT_SIZE = 20


folder = "SSIM"

ordered_files = [
    "ssim_lora_usd2",
    "ssim_lora_usd3",
    "ssim_lora_thai4",
    "ssim_nanobanana",
    "ssim_textdiffuse",
    "ssim_random_runs_resultsprompttesting",
    "ssim_uldm_256",
    "ssim_uldm_384x192",
    "ssim_uldm_384x160",
    "ssim_uldm_1152x480",
]


label_map = {
    "ssim_random_runs_resultsprompttesting": "Generert datasett\nPrompttesting",
    "ssim_lora_usd2": "Lora USD\nTrening 2",
    "ssim_lora_usd3": "Lora USD\nTrening 3",
    "ssim_lora_thai4": "Lora Thai\nTrening 4",
    "ssim_nanobanana": "NanoBanana Pro",
    "ssim_textdiffuse": "TextDiffuser",
    "ssim_uldm_256": "ULDM\n256x256",
    "ssim_uldm_384x192": "ULDM\n384x192",
    "ssim_uldm_384x160": "ULDM\n384x160",
    "ssim_uldm_1152x480": "ULDM\n1152x480",
}

all_scores = []
labels = []


for filename in ordered_files:

    file_path = os.path.join(folder, filename + ".json")

    # hopper over filer som ikke finnes
    if not os.path.exists(file_path):
        print(f"Fant ikke fil: {file_path}")
        continue

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scores = []

    # hent ut alle SSIM-verdier
    for run in data["runs"]:
        for pair in run["pairs"]:
            scores.append(pair["ssim"])

    # hopper over tomme filer
    if len(scores) == 0:
        print(f"Ingen SSIM-scores i: {file_path}")
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
    "SSIM score",
    fontsize=LABEL_FONT_SIZE
)

plt.title(
    "SSIM sammenligning av ulike metoder",
    fontsize=TITLE_FONT_SIZE
)

plt.grid(True, axis='y', alpha=0.3)

plt.tight_layout()

output_path = os.path.join(folder, "ssim_comparison.pdf")

plt.savefig(output_path, bbox_inches='tight')

plt.show()

print(f"Saved plot to: {output_path}")