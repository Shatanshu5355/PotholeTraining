import os
import shutil
import glob
from pathlib import Path

def clean_yolo_dataset(images_dir, labels_dir, output_dir, min_annotations=2, max_annotations=6):
    """
    Clean YOLO dataset based on specified criteria:
    1. Between min_annotations and max_annotations annotations per image
    2. Only class "0" annotations
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing label files
        output_dir: Directory to save cleaned dataset
        min_annotations: Minimum number of annotations required (default: 3)
        max_annotations: Maximum number of annotations allowed (default: 8)
    """
    # Create output directories if they don't exist
    output_images_dir = os.path.join(output_dir, "images")
    output_labels_dir = os.path.join(output_dir, "labels")
    
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    # Get all label files
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    
    # Statistics for console output
    total_files = len(label_files)
    files_with_min_annotations = 0
    files_with_max_annotations = 0
    files_with_only_class_0 = 0
    files_copied = 0
    
    print(f"Starting cleaning process for {total_files} files...")
    
    for label_file in label_files:
        # Get corresponding image file
        filename = Path(label_file).stem
        possible_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_file = None
        
        for ext in possible_extensions:
            potential_image = os.path.join(images_dir, filename + ext)
            if os.path.exists(potential_image):
                image_file = potential_image
                break
        
        if not image_file:
            print(f"Warning: No image found for {filename}")
            continue
        
        # Read label file
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        # Check if it has enough annotations but not too many
        annotation_count = len(lines)
        if annotation_count < min_annotations:
            continue
            
        files_with_min_annotations += 1
        
        if annotation_count > max_annotations:
            continue
            
        files_with_max_annotations += 1
        
        # Check if all annotations are class 0
        all_class_0 = all(line.strip().startswith('0 ') for line in lines)
        
        if not all_class_0:
            continue
        
        files_with_only_class_0 += 1
        
        # Copy files to output directory
        image_extension = os.path.splitext(image_file)[1]
        shutil.copy(image_file, os.path.join(output_images_dir, filename + image_extension))
        shutil.copy(label_file, os.path.join(output_labels_dir, filename + '.txt'))
        
        files_copied += 1
    
    # Print statistics
    print("\n--- Cleaning Results ---")
    print(f"Total files processed: {total_files}")
    print(f"Files with at least {min_annotations} annotations: {files_with_min_annotations}")
    print(f"Files with at most {max_annotations} annotations: {files_with_max_annotations}")
    print(f"Files with annotation count between {min_annotations}-{max_annotations}: {files_with_max_annotations}")
    print(f"Files with only class 0: {files_with_only_class_0}")
    print(f"Files that met all criteria and were copied: {files_copied}")
    print(f"Files discarded: {total_files - files_copied}")
    print(f"\nCleaned dataset saved to: {output_dir}")

# Example usage
if __name__ == "__main__":
    # Update these paths to your actual directories
    images_dir = "Dataset(1,2,3,4)/images"
    labels_dir = "Dataset(1,2,3,4)/labels"
    output_dir = "final_data"
    
    clean_yolo_dataset(images_dir, labels_dir, output_dir, min_annotations=2, max_annotations=6)