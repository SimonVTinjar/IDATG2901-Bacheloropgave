from pathlib import Path

# 👉 Endre denne til rotmappa di (dataset_currency)
ROOT_DIR = Path(r"I:\Ai prosjekt\Datasett 4 currensy\dataset_currency")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

created = 0
skipped = 0

# 🔥 Går gjennom ALLE undermapper
for img in ROOT_DIR.rglob("*"):
    if img.is_file() and img.suffix.lower() in IMAGE_EXTS:
        txt_path = img.with_suffix(".txt")

        if txt_path.exists():
            skipped += 1
        else:
            txt_path.write_text("usd_bill, ", encoding="utf-8")
            created += 1

print(f"✅ Ferdig!")
print(f"Laget: {created} txt-filer")
print(f"Hoppet over: {skipped}")
