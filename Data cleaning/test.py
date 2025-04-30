import os
import cv2
import numpy as np
from pathlib import Path
import yaml

def visualize_yolo_annotations(images_folder, labels_folder, output_folder, yaml_path=None):
    """
    Visualize YOLO format annotations on images and save them to a new folder
    
    Args:
        images_folder (str): Path to folder containing images
        labels_folder (str): Path to folder containing YOLO format labels
        output_folder (str): Path to save visualized images
        yaml_path (str, optional): Path to data.yaml file for class names
    """
    # Set default class names
    class_names = ['pothole']
    
    # Read class names from yaml if provided
    if yaml_path and os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
            if 'names' in yaml_data:
                class_names = yaml_data['names']
                print(f"Loaded class names from {yaml_path}: {class_names}")
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(Path(images_folder).glob(f'*{ext}')))
        image_files.extend(list(Path(images_folder).glob(f'*{ext.upper()}')))
    
    print(f"Found {len(image_files)} images")
    
    # Process each image
    processed_count = 0
    skipped_count = 0
    missing_labels = []
    
    for img_path in image_files:
        # Get corresponding label file
        base_name = img_path.stem
        label_path = Path(labels_folder) / f"{base_name}.txt"
        
        # Check if label file exists
        if not label_path.exists():
            print(f"Warning: No label file found for {img_path.name}")
            missing_labels.append(img_path.name)
            skipped_count += 1
            continue
        
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            skipped_count += 1
            continue
        
        # Get image dimensions
        height, width = img.shape[:2]
        
        # Read annotations
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # Draw bounding boxes
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"Warning: Invalid annotation format in {label_path}")
                continue
                
            class_id = int(parts[0])
            x_center = float(parts[1]) * width
            y_center = float(parts[2]) * height
            box_width = float(parts[3]) * width
            box_height = float(parts[4]) * height
            
            # Calculate box coordinates
            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)
            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)
            
            # Get class name
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
            
            # Generate a color based on class ID
            color = (0, 255, 0)  # Green for pothole
            if class_id > 0:
                # If there are more classes, generate different colors
                colors = [(0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255)]
                color = colors[class_id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw class name and confidence (if available)
            label = f"{class_name}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(img, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Save the visualized image
        output_path = Path(output_folder) / img_path.name
        cv2.imwrite(str(output_path), img)
        
        processed_count += 1
        
        # Print progress every 10 images
        if processed_count % 10 == 0:
            print(f"Processed {processed_count} images")
    
    print(f"\nVisualization complete!")
    print(f"Processed {processed_count} images")
    print(f"Skipped {skipped_count} images")
    
    if missing_labels:
        print(f"\nMissing label files for {len(missing_labels)} images:")
        for img_name in missing_labels[:10]:  # Show first 10
            print(f"- {img_name}")
        if len(missing_labels) > 10:
            print(f"... and {len(missing_labels) - 10} more")


# ===== MODIFY THESE PATHS =====
# Set your input paths here
IMAGES_FOLDER = "Dataset(1,2,3,4)/images"  # Replace with your images folder path
LABELS_FOLDER = "Dataset(1,2,3,4)/labels"  # Replace with your labels folder path
OUTPUT_FOLDER = "test_visualizations"  # Replace with your desired output folder path
DATA_YAML_PATH = "data.yaml"  # Replace with your data.yaml path or leave as is to use default class names

# ===== RUN THE SCRIPT =====
if __name__ == "__main__":
    # Make sure yaml is installed
    try:
        import yaml
    except ImportError:
        print("PyYAML is required. Please install it using: pip install pyyaml")
        exit(1)
        
    # Run the visualization
    visualize_yolo_annotations(IMAGES_FOLDER, LABELS_FOLDER, OUTPUT_FOLDER, DATA_YAML_PATH)