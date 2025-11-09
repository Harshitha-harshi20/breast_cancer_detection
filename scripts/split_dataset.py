import os
import csv
import random
from pathlib import Path
from PIL import Image

# Paths
LABELS_CSV = r"C:\Users\sonuh\breast_cancer_detection\scripts\labels.csv"
DATASET_ROOT = r"C:\Users\sonuh\breast_cancer_detection\dataset\BreaKHis_v1\BreaKHis_v1\histology_slides\breast"
OUTPUT_DIR = r"C:\Users\sonuh\breast_cancer_detection\dataset\split_dataset"

# Split ratios
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}

# Create output folders
for split in SPLIT_RATIOS:
    for label in ["benign", "malignant"]:
        Path(OUTPUT_DIR, split, label).mkdir(parents=True, exist_ok=True)

# Read labels.csv
with open(LABELS_CSV, newline="") as f:
    reader = csv.DictReader(f)
    data = [(row["filename"], row["label"]) for row in reader]

# Shuffle data
random.shuffle(data)

# Split data
total = len(data)
train_end = int(total * SPLIT_RATIOS["train"])
val_end = train_end + int(total * SPLIT_RATIOS["val"])

splits = {
    "train": data[:train_end],
    "val": data[train_end:val_end],
    "test": data[val_end:]
}

# Copy files, convert to RGB, and create CSV for each split
for split, items in splits.items():
    csv_path = Path(OUTPUT_DIR, f"{split}_labels.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        for rel_path, label in items:
            src_path = Path(DATASET_ROOT) / rel_path
            dest_folder = Path(OUTPUT_DIR, split, label)
            dest_path = dest_folder / src_path.name

            # Convert to RGB and save
            img = Image.open(src_path)
            img = img.convert("RGB")
            img.save(dest_path)

            writer.writerow([str(dest_path), label])
    print(f"✅ {split} split done with {len(items)} images. CSV saved at {csv_path}")
