import os
import random
import shutil
from pathlib import Path

# --- Innstillinger ---
SOURCE_DIR = Path("archive/Currency_data")            # mappa du har nå
OUT_DIR = Path("dataset_currency")       # ny mappe som lages
SPLIT = (0.70, 0.15, 0.15)               # train, val, test
SEED = 42
MOVE_FILES = False                       # False = kopier, True = flytt

# Hvilke filtyper som regnes som bilder
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS


def safe_copy_or_move(src: Path, dst: Path, move: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        # Hvis det allerede finnes en fil med samme navn, lag et unikt navn
        stem = dst.stem
        ext = dst.suffix
        i = 1
        while dst.exists():
            dst = dst.with_name(f"{stem}_{i}{ext}")
            i += 1
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))


def split_list(items, split):
    n = len(items)
    n_train = int(n * split[0])
    n_val = int(n * split[1])
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return train, val, test


def main():
    random.seed(SEED)

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Fant ikke SOURCE_DIR: {SOURCE_DIR.resolve()}")

    classes = [d for d in SOURCE_DIR.iterdir() if d.is_dir()]
    if not classes:
        raise RuntimeError(f"Ingen undermapper (klasser) funnet i {SOURCE_DIR.resolve()}")

    print("Fant klasser:", ", ".join([c.name for c in classes]))
    print("Output:", OUT_DIR.resolve())
    print("Split:", SPLIT, "Move:" , MOVE_FILES)

    total = 0

    for cls_dir in classes:
        cls_name = cls_dir.name

        images = [p for p in cls_dir.rglob("*") if p.is_file() and is_image(p)]
        if not images:
            print(f"[ADVARSEL] Ingen bilder funnet i {cls_dir}")
            continue

        random.shuffle(images)
        train, val, test = split_list(images, SPLIT)

        for split_name, split_items in [("train", train), ("val", val), ("test", test)]:
            for src in split_items:
                dst = OUT_DIR / split_name / cls_name / src.name
                safe_copy_or_move(src, dst, MOVE_FILES)

        total += len(images)
        print(f"{cls_name}: {len(images)} bilder -> train {len(train)}, val {len(val)}, test {len(test)}")

    print(f"Ferdig! Totalt behandlet: {total} bilder")
    print("Datasett ligger i:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
