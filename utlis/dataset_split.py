# 5-Fold Cross Validation Split
import os
import shutil
import random
from sklearn.model_selection import KFold

# Set random seed to ensure reproducibility
random.seed(3407)
print("Random seed:")
print(random.seed)

# Define folder paths
base_folder = "data"  # Main data directory
source_folder = os.path.join(base_folder, "data-raw")  # Original data directory
output_folder = os.path.join(base_folder, "data-fold")  # Output directory for folds

# Define class folders
class_folders = ['class_0', 'class_1', 'class_2']

# Create output directories for each of the 5 folds
for fold in range(5):
    fold_train_folder = os.path.join(output_folder, f"fold_{fold+1}", "train")
    fold_val_folder = os.path.join(output_folder, f"fold_{fold+1}", "val")
    for class_folder in class_folders:
        os.makedirs(os.path.join(fold_train_folder, class_folder), exist_ok=True)
        os.makedirs(os.path.join(fold_val_folder, class_folder), exist_ok=True)

# Iterate through each class folder to perform 5-fold splitting
for class_folder in class_folders:
    src_class_folder = os.path.join(source_folder, class_folder)

    # Get all sample subdirectories in the current class folder
    all_samples = [d for d in os.listdir(src_class_folder) if os.path.isdir(os.path.join(src_class_folder, d))]
    if len(all_samples) == 0:
        print(f"No sample folders found in class {class_folder}, skipping split.")
        continue

    random.shuffle(all_samples)  # Shuffle samples randomly

    # Dynamically adjust the number of splits to ensure n_splits is not greater than the number of samples
    n_splits = min(5, len(all_samples))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=30)

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_samples)):
        fold_train_folder = os.path.join(output_folder, f"fold_{fold+1}", "train", class_folder)
        fold_val_folder = os.path.join(output_folder, f"fold_{fold+1}", "val", class_folder)

        # Retrieve training and validation sample folders based on indices
        train_samples = [all_samples[i] for i in train_idx]
        val_samples = [all_samples[i] for i in val_idx]

        # Copy sample subfolders to the training set directory
        for sample in train_samples:
            src_path = os.path.join(src_class_folder, sample)
            dest_path = os.path.join(fold_train_folder, sample)
            shutil.copytree(src_path, dest_path)

        # Copy sample subfolders to the validation set directory
        for sample in val_samples:
            src_path = os.path.join(src_class_folder, sample)
            dest_path = os.path.join(fold_val_folder, sample)
            shutil.copytree(src_path, dest_path)

        print(f"{class_folder} - Fold {fold+1} dataset split complete. Train: {len(train_samples)}, Val: {len(val_samples)}")

print("5-Fold dataset splitting complete!")