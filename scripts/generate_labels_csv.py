import os
import csv
from pathlib import Path

# 🔹 Root dataset folder
DATASET_DIR = r"C:\Users\sonuh\breast_cancer_detection\dataset\BreaKHis_v1\BreaKHis_v1\histology_slides\breast"

# 🔹 Output CSV
OUTPUT_CSV = r"C:\Users\sonuh\breast_cancer_detection\scripts\labels.csv"

# 🔹 Allowed image extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# 🔹 Collect files with labels
files_with_labels = []
for root, _, filenames in os.walk(DATASET_DIR):
    for fname in filenames:
        if Path(fname).suffix.lower() in IMAGE_EXTENSIONS:
            full_path = Path(root) / fname
            # Determine top-level parent folder: benign or malignant
            parts = full_path.parts
            if "benign" in parts:
                label = "benign"
            elif "malignant" in parts:
                label = "malignant"
            else:
                continue  # skip any unknown folders
            # Save relative path from dataset root
            rel_path = full_path.relative_to(DATASET_DIR)
            files_with_labels.append((str(rel_path), label))

# 🔹 Write CSV
with open(OUTPUT_CSV, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "label"])
    for f, l in files_with_labels:
        writer.writerow([f, l])

print(f"✅ labels.csv created with {len(files_with_labels)} entries at {OUTPUT_CSV}")
