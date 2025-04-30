import os

def cross_check_files(images_path, labels_path):
    """
    Compare filenames between images and labels folders.
    
    Args:
        images_path (str): Path to the images folder
        labels_path (str): Path to the labels folder
        
    Returns:
        tuple: (match_status (bool), missing_in_images, missing_in_labels)
    """
    # Get filenames without extensions
    image_files = {os.path.splitext(f)[0] for f in os.listdir(images_path) 
                  if os.path.isfile(os.path.join(images_path, f))}
    
    label_files = {os.path.splitext(f)[0] for f in os.listdir(labels_path)
                  if os.path.isfile(os.path.join(labels_path, f))}
    
    # Find mismatches
    missing_in_images = label_files - image_files
    missing_in_labels = image_files - label_files
    
    # Check if there's complete matching
    match_status = len(missing_in_images) == 0 and len(missing_in_labels) == 0
    
    return match_status, missing_in_images, missing_in_labels

def main():
    # Use hardcoded paths instead of input
    images_path = "Dataset(1,2,3,4)/images"
    labels_path = "Dataset(1,2,3,4)/labels"
    
    # Validate paths
    if not os.path.isdir(images_path):
        print(f"Error: Images path '{images_path}' is not a valid directory")
        return
    
    if not os.path.isdir(labels_path):
        print(f"Error: Labels path '{labels_path}' is not a valid directory")
        return
    
    # Check files
    match_status, missing_in_images, missing_in_labels = cross_check_files(images_path, labels_path)
    
    # Report results
    if match_status:
        print("✅ All files match between images and labels folders.")
    else:
        print("❌ Mismatches found between images and labels folders:")
        
        if missing_in_images:
            print(f"\nFiles in labels folder but missing in images folder ({len(missing_in_images)}):")
            for filename in sorted(missing_in_images):
                print(f"  - {filename}")
        
        if missing_in_labels:
            print(f"\nFiles in images folder but missing in labels folder ({len(missing_in_labels)}):")
            for filename in sorted(missing_in_labels):
                print(f"  - {filename}")

if __name__ == "__main__":
    main()