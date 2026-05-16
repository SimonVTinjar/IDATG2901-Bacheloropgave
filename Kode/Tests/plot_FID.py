import json
import os
import matplotlib.pyplot as plt


TITLE_FONT_SIZE = 24
LABEL_FONT_SIZE = 20
TICK_FONT_SIZE = 20
VALUE_FONT_SIZE = 16

folder = "FID"



ordered_files = [
    "fid_lora_usd2",
    "fid_lora_usd3",
    "fid_lora_thai4",
    "fid_nanobanana",
    "fid_textdiffuse",
    "fid_basedataset_384x160",
    "fid_ULDM_256x256",
    "fid_ULDM_384x192",
    "fid_ULDM_384x160",
    "fid_ULDM_1152x480",
]


label_map = {
    "fid_ULDM_256x256": "ULDM\n256x256",
    "fid_ULDM_384x192": "ULDM\n384x192",
    "fid_ULDM_384x160": "ULDM\n384x160",
    "fid_ULDM_1152x480": "ULDM\n1152x480",
    "fid_nanobanana": "NanoBanana",
    "fid_textdiffuse": "TextDiffuser",
    "fid_basedataset_384x160": "Generert datasett\nPrompt testing",
    "fid_lora_usd2": "Lora USD\nTrening 2",
    "fid_lora_usd3": "Lora USD\nTrening 3",
    "fid_lora_thai4": "Lora Thai\nTrening 4",
}

fid_scores = []
labels = []


for filename in ordered_files:

    file_path = os.path.join(folder, filename + ".json")

    if not os.path.exists(file_path):
        print(f"Fant ikke fil: {file_path}")
        continue

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fid_value = data["fid"]

    fid_scores.append(fid_value)

    label = label_map.get(filename, filename)

    labels.append(label)


plt.figure(figsize=(20, 10))

bars = plt.bar(
    range(len(fid_scores)), 
    fid_scores,
    width=0.7)

plt.xticks(
    range(len(labels)),
    labels,
    rotation=25,
    ha='center',
    fontsize=TICK_FONT_SIZE
)

plt.yticks(fontsize=TICK_FONT_SIZE)

plt.ylabel(
    "FID score",
    fontsize=LABEL_FONT_SIZE
)

plt.title(
    "FID sammenligning av ulike metoder",
    fontsize=TITLE_FONT_SIZE
)

plt.grid(True, axis='y', alpha=0.3)

# skriv verdi over hver stolpe
for i, value in enumerate(fid_scores):
    plt.text(
        i,
        value + 0.5,
        f"{value:.2f}",
        ha='center',
        fontsize=VALUE_FONT_SIZE
    )

plt.tight_layout()

output_path = os.path.join(folder, "fid_comparison.pdf")

plt.savefig(output_path, bbox_inches='tight')

plt.show()

print(f"Saved plot to: {output_path}")