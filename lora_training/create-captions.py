import os

CAPTIONS = {
    "_full": "photorealistic banknote, front side, full banknote, engraved portrait, fine line engraving, guilloche pattern, crisp print, sharp focus, high detail",
    "_portrait": "photorealistic banknote, close-up, engraved portrait, fine line engraving, crisp print, sharp focus, high detail",
    "_left_pattern": "photorealistic banknote, close-up, fine line engraving, guilloche pattern, crisp print, sharp focus, high detail",
    "_right_pattern": "photorealistic banknote, close-up, fine line engraving, guilloche pattern, crisp print, sharp focus, high detail",
    "_upper_band": "photorealistic banknote, close-up, engraved print details, fine line structure, crisp print, sharp focus, high detail",
    "_lower_pattern": "photorealistic banknote, close-up, dense engraving pattern, fine line structure, crisp print, sharp focus, high detail",
    "_ribbon_area": "photorealistic banknote, close-up, security pattern area, fine line engraving, crisp print, sharp focus, high detail",
}

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")

DATASET_DIR = "lora_dataset"  # change this

for root, _, files in os.walk(DATASET_DIR):
    for fn in files:
        if not fn.lower().endswith(IMG_EXTS):
            continue

        base = os.path.splitext(fn)[0]
        matched = None
        for key in sorted(CAPTIONS.keys(), key=len, reverse=True):
            if base.endswith(key):
                matched = key
                break

        if matched is None:
            print("SKIP (no suffix match):", fn)
            continue

        txt_path = os.path.join(root, base + ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(CAPTIONS[matched] + "\n")

print("Done.")