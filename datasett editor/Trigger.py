import os

# === CONFIG ===
IMAGE_FOLDER = r"I:\Ai prosjekt\IDATG2901-Bacheloropgave\Datasett copy"  # <-- CHANGE THIS
TRIGGER = "[trigger]"
BASE_CAPTION = "ultra detailed banknote texture, 100 usd dollar bill, sharp focus, macro photo"

# Supported formats
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# =================

def main():
    count = 0

    for filename in os.listdir(IMAGE_FOLDER):
        if filename.lower().endswith(IMAGE_EXTENSIONS):

            image_path = os.path.join(IMAGE_FOLDER, filename)

            name, _ = os.path.splitext(filename)
            txt_path = os.path.join(IMAGE_FOLDER, name + ".txt")

            # Skip if txt already exists
            if os.path.exists(txt_path):
                print(f"Skipping (exists): {txt_path}")
                continue

            caption = f"{TRIGGER} {BASE_CAPTION}"

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(caption)

            print(f"Created: {txt_path}")
            count += 1

    print(f"\nDone. Created {count} caption files.")


if __name__ == "__main__":
    main()