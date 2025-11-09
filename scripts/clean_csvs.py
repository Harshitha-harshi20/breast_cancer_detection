import pandas as pd
from pathlib import Path

# Paths to CSVs
csv_files = {
    "train": "dataset/split_dataset/train_labels.csv",
    "val": "dataset/split_dataset/val_labels.csv",
    "test": "dataset/split_dataset/test_labels.csv"
}

# Base folders for datasets
base_dirs = {
    "train": Path("dataset/train"),
    "val": Path("dataset/val"),
    "test": Path("dataset/test")
}

for split, csv_path in csv_files.items():
    print(f"\nProcessing {split} CSV: {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Drop rows with missing filenames or labels
    df = df.dropna(subset=['filename', 'label'])
    
    # Ensure columns are strings
    df['filename'] = df['filename'].astype(str)
    df['label'] = df['label'].astype(str)
    
    # Check missing files
    missing_files = []
    for fname in df['filename']:
        file_path = base_dirs[split] / fname
        if not file_path.exists():
            missing_files.append(fname)
    
    if missing_files:
        print(f"⚠️ Missing {len(missing_files)} files in {split} folder:")
        for f in missing_files:
            print(f"   {f}")
    else:
        print(f"✅ All files exist in {split} folder.")
    
    # Save cleaned CSV back
    df.to_csv(csv_path, index=False)
    print(f"✅ Cleaned CSV saved: {csv_path}")
