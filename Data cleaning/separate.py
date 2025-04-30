import os
import shutil
from pathlib import Path

def segregate_files(source_folder, images_folder, labels_folder):
    # Create destination folders if they don't exist
    Path(images_folder).mkdir(exist_ok=True)
    Path(labels_folder).mkdir(exist_ok=True)
    
    # Get all files in the source folder
    files = os.listdir(source_folder)
    
    # Track missing files
    missing_files = []
    
    # Process jpg files and find corresponding label files
    jpg_files = [f for f in files if f.lower().endswith('.jpg')]
    
    for jpg_file in jpg_files:
        jpg_path = os.path.join(source_folder, jpg_file)
        base_name = os.path.splitext(jpg_file)[0]
        
        # Assuming label files have the same name but different extension
        label_file = f"{base_name}.txt"  # Adjust extension as needed
        label_path = os.path.join(source_folder, label_file)
        
        # Check if label file exists
        if label_file in files:
            # Only copy files if both image and label exist
            shutil.copy2(jpg_path, os.path.join(images_folder, jpg_file))
            shutil.copy2(label_path, os.path.join(labels_folder, label_file))
        else:
            missing_files.append(f"Missing label for: {jpg_file}")
    
    # Find label files that might not have corresponding jpg files
    label_files = [f for f in files if f.lower().endswith('.txt')]  # Adjust extension as needed
    
    for label_file in label_files:
        base_name = os.path.splitext(label_file)[0]
        jpg_file = f"{base_name}.jpg"
        
        # If already processed, skip
        if jpg_file in jpg_files:
            continue
        
        # If jpg file doesn't exist, record it
        if jpg_file not in files:
            missing_files.append(f"Missing image for: {label_file}")
    
    # Print missing files
    if missing_files:
        print("Missing files:")
        for message in missing_files:
            print(f"- {message}")
        print(f"Total missing pairs: {len(missing_files)}")
    else:
        print("No missing files found. All pairs were transferred successfully.")

if __name__ == "__main__":
    # Set your folder paths here
    source_folder = "Dataset 4/Sunset"
    images_folder = "Dataset 4/images"
    labels_folder = "Dataset 4/labels"
    
    segregate_files(source_folder, images_folder, labels_folder)