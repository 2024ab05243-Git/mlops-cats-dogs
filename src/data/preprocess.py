import os
import shutil
import random
from pathlib import Path

RAW_DIR = Path("data/raw/PetImages")
PROCESSED_DIR = Path("data/processed")

SPLIT_RATIO = (0.8, 0.1, 0.1)
RANDOM_SEED = 42


def create_dirs():
    for split in ["train", "val", "test"]:
        for label in ["cats", "dogs"]:
            (PROCESSED_DIR / split / label).mkdir(parents=True, exist_ok=True)


def split_data():
    random.seed(RANDOM_SEED)

    for label_map in [("Cat", "cats"), ("Dog", "dogs")]:
        src_folder = RAW_DIR / label_map[0]
        files = [f for f in src_folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]

        random.shuffle(files)

        total = len(files)
        train_end = int(total * SPLIT_RATIO[0])
        val_end = train_end + int(total * SPLIT_RATIO[1])

        splits = {
            "train": files[:train_end],
            "val": files[train_end:val_end],
            "test": files[val_end:]
        }

        for split, split_files in splits.items():
            for file in split_files:
                dest = PROCESSED_DIR / split / label_map[1] / file.name
                shutil.copy(file, dest)


if __name__ == "__main__":
    create_dirs()
    split_data()
    print("Preprocessing completed.")