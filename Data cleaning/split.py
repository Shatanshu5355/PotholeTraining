import os
import shutil
import random
from pathlib import Path

# USER SETTINGS - CHANGE THESE VALUES
images_dir = "final_data/images"  # Your images folder
labels_dir = "final_data/labels"  # Your labels folder
output_dir = "Final_Data_split"   # Where to save the split dataset

total_images = None  # Set to a number (like 5000) if you want to limit training images, or None to use all
train_ratio = 0.8    # Percentage for training (0.8 = 80%)
val_ratio = 0.1      # Percentage for validation (0.1 = 10%)
test_ratio = 0.1     # Percentage for testing (0.1 = 10%)

# Create output folders
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)

# Get all image files
image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
print(f"Found {len(image_files)} images")

# Shuffle images
random.seed(42)  # For reproducibility
random.shuffle(image_files)

# Limit total images if needed
if total_images is not None and total_images < len(image_files):
    print(f"Limiting to {total_images} images")
    image_files = image_files[:total_images]

# Calculate split sizes
num_images = len(image_files)
num_train = int(train_ratio * num_images)
num_val = int(val_ratio * num_images)

# Split the dataset
train_files = image_files[:num_train]
val_files = image_files[num_train:num_train + num_val]
test_files = image_files[num_train + num_val:]

print(f"Training: {len(train_files)} images")
print(f"Validation: {len(val_files)} images")
print(f"Testing: {len(test_files)} images")

# Copy files for each split
for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
    print(f"Copying {split} files...")
    
    for img_file in files:
        # Get filename without extension
        base_name = os.path.splitext(img_file)[0]
        
        # Copy image
        shutil.copy2(
            os.path.join(images_dir, img_file), 
            os.path.join(output_dir, split, 'images', img_file)
        )
        
        # Copy label if it exists
        label_file = base_name + '.txt'  # YOLOv8 uses .txt files
        label_path = os.path.join(labels_dir, label_file)
        
        if os.path.exists(label_path):
            shutil.copy2(
                label_path,
                os.path.join(output_dir, split, 'labels', label_file)
            )

print("✅ Done! Dataset split complete.")
