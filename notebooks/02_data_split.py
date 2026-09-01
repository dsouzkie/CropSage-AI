"""
Data preparation script to split the PlantVillage dataset into train/val/test sets.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

def main():
    dataset_dir = Path(r"c:\Users\chris\Downloads\FYP\dataset\plantvillage dataset\color")
    output_dir = Path(r"c:\Users\chris\Downloads\FYP\notebooks\outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found at {dataset_dir}")
        return

    # 1. Scan class folders
    classes = sorted([d.name for d in dataset_dir.iterdir() if d.is_dir()])
    if not classes:
        print("No classes found in dataset directory.")
        return
        
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    # 4. Save class indices mapping
    with open(output_dir / "class_indices.json", "w") as f:
        json.dump(class_to_idx, f, indent=4)

    data = []
    for cls_name in classes:
        cls_dir = dataset_dir / cls_name
        # use relative path for CSV
        for img_path in cls_dir.glob("*.*"):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                rel_path = img_path.relative_to(dataset_dir)
                data.append({
                    "filepath": str(rel_path).replace("\\", "/"), # Standardize paths in CSV
                    "class_name": cls_name,
                    "class_index": class_to_idx[cls_name]
                })

    df = pd.DataFrame(data)
    if df.empty:
        print("No images found.")
        return
        
    print(f"Total images found: {len(df)}")

    # 2. Perform 80/10/10 stratified split
    # First split: 80% train, 20% temp (val + test)
    train_df, temp_df = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df["class_index"]
    )
    # Second split: split temp into 50% val, 50% test (10% / 10% of total)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df["class_index"]
    )

    # 3. Save splits to CSV
    train_df.to_csv(output_dir / "train_split.csv", index=False)
    val_df.to_csv(output_dir / "val_split.csv", index=False)
    test_df.to_csv(output_dir / "test_split.csv", index=False)

    # 5. Print split statistics
    print(f"Train set: {len(train_df)} images")
    print(f"Val set: {len(val_df)} images")
    print(f"Test set: {len(test_df)} images")

    # 6. Compute class weights
    classes_unique = np.unique(train_df["class_index"])
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes_unique,
        y=train_df["class_index"]
    )
    
    # Map index to weight for saving
    class_weights_dict = {str(k): float(v) for k, v in zip(classes_unique, weights)}
    with open(output_dir / "class_weights.json", "w") as f:
        json.dump(class_weights_dict, f, indent=4)
        
    print("Class weights computed and saved.")

    # 7. Create bar chart for distribution
    plt.figure(figsize=(15, 8))
    
    # We ensure all classes are present even if counts are 0 in some splits
    train_counts = train_df["class_name"].value_counts().reindex(classes, fill_value=0)
    val_counts = val_df["class_name"].value_counts().reindex(classes, fill_value=0)
    test_counts = test_df["class_name"].value_counts().reindex(classes, fill_value=0)

    indices = np.arange(len(classes))
    width = 0.25

    plt.bar(indices - width, train_counts.values, width, label='Train')
    plt.bar(indices, val_counts.values, width, label='Val')
    plt.bar(indices + width, test_counts.values, width, label='Test')

    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.title('Class Distribution across Splits')
    plt.xticks(indices, classes, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "split_distribution.png")
    plt.close()
    
    print(f"Distribution chart saved to {output_dir / 'split_distribution.png'}")

if __name__ == "__main__":
    main()
