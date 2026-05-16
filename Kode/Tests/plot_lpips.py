import os
import json
import numpy as np
import matplotlib.pyplot as plt


TITLE_FONT_SIZE = 24
LABEL_FONT_SIZE = 20
TICK_FONT_SIZE = 20

json_folder = "LPIPS"

files = [
    ("lpips_Lora_USD2.json", "Lora USD \nTrening 2"),
    ("lpips_Lora_USD3.json", "Lora USD \nTrening 3"),
    ("lpips_Lora_Thai4.json", "Lora Thai \nTrening 4"),
    ("lpips_nanobanana.json", "NanoBanana \nPro"),
    ("lpips_textdiffuse.json", "TextDiffuser"),
    ("lpips_random_results_10x170prompt.json", "Generert datasett\nPrompt testing"),
    ("lpips_random_results_256x256.json", "ULDM \n256x256"),
    ("lpips_random_results_384x192.json", "ULDM \n384x192"),
    ("lpips_random_results_384x160.json", "ULDM \n384x160"),
    ("lpips_uldm1152x480.json", "ULDM \n1152x480"),
]

all_scores = []
labels = []

for filename, label in files:
    file_path = os.path.join(json_folder, filename)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scores = []

    for run in data["runs"]:

        # støtter både "scores" og "pairs"
        if "scores" in run:
            scores.extend(run["scores"])

        elif "pairs" in run:
            scores.extend([
                pair["lpips"]
                for pair in run["pairs"]
            ])

    all_scores.append(scores)
    labels.append(label)

plt.figure(figsize=(20, 10))

plt.boxplot(all_scores, patch_artist=True)

plt.xticks(
    range(1, len(labels) + 1),
    labels,
    rotation=20,
    ha='center',
    fontsize=TICK_FONT_SIZE
)

plt.yticks(fontsize=TICK_FONT_SIZE)

plt.ylabel(
    "LPIPS score",
    fontsize=LABEL_FONT_SIZE
)

plt.title(
    "LPIPS sammenligning av ulike metoder og treningsoppsett",
    fontsize=TITLE_FONT_SIZE
)

plt.grid(True, axis='y', alpha=0.3)

plt.tight_layout()

output_path = os.path.join(json_folder, "lpips_comparison.pdf")

plt.savefig(output_path, bbox_inches='tight')

plt.show()

print(f"Saved plot to: {output_path}")