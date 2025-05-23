import os
import shutil
import random

# Set source and target folders
SOURCE_DIR = 'dataset'
TRAIN_DIR = os.path.join(SOURCE_DIR, 'train')
VAL_DIR = os.path.join(SOURCE_DIR, 'val')

# Make sure train/ and val/ folders exist
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

# Loop through each class folder (e.g., mehedi, asif, akash)
for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)
    if not os.path.isdir(class_path) or class_name in ['train', 'val']:
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    split_idx = int(len(images) * 0.8)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    # Make subfolders in train/ and val/
    os.makedirs(os.path.join(TRAIN_DIR, class_name), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR, class_name), exist_ok=True)

    # Copy files
    for img in train_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(TRAIN_DIR, class_name, img))

    for img in val_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(VAL_DIR, class_name, img))

    print(f"Split done for {class_name}: {len(train_images)} train, {len(val_images)} val")

print("✅ Dataset split complete.")
