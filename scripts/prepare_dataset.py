import random
import shutil
from pathlib import Path

random.seed(42)

RAW_IMAGES = Path("data/raw/images")
RAW_MASKS = Path("data/raw/masks")
OUT = Path("data/processed")

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def make_dirs():
    for split in ["train", "val", "test"]:
        (OUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT / split / "masks").mkdir(parents=True, exist_ok=True)

def main():
    make_dirs()

    images = sorted([p for p in RAW_IMAGES.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif"]])
    pairs = []

    for img in images:
        mask = RAW_MASKS / img.name
        if mask.exists():
            pairs.append((img, mask))

    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train+n_val],
        "test": pairs[n_train+n_val:]
    }

    for split, items in splits.items():
        for img, mask in items:
            shutil.copy(img, OUT / split / "images" / img.name)
            shutil.copy(mask, OUT / split / "masks" / mask.name)

    print("Total pairs:", n)
    for split, items in splits.items():
        print(split, len(items))

if __name__ == "__main__":
    main()
