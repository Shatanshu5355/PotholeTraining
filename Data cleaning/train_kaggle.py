
# 4. Training Configuration
from ultralytics import YOLO

# Load YOLOv8n (nano model for mobile)
model = YOLO('/root/.cache/kagglehub/datasets/shatanshur/my-data/versions/1/Final_Data_split/yolo11n.pt')

# Train the model (adjust paths to your dataset.yaml)
results = model.train(
    data="/kaggle/input/version1-update/data_new.yaml",
    epochs=150,                   # Increased from 100 for better convergence
    batch=48,                     # Back to original batch size to avoid memory issues
    imgsz=640,                    # Maintaining image size for pothole detection
    optimizer='SGD',              # Changed from AdamW to SGD
    lr0=0.003,                    # Increased from 0.002 
    lrf=0.02,                     # Increased from 0.01
    momentum=0.95,                # Increased momentum for SGD
    weight_decay=0.0007,          # Increased from 0.0005
    augment=True,                 # Keep data augmentation enabled
    flipud=0.3,                   # Keep vertical flip augmentation
    fliplr=0.7,                   # Increased from 0.5
    hsv_h=0.01,                   # Reduced from 0.015
    hsv_s=0.7,                    # Keep saturation variation
    hsv_v=0.4,                    # Keep brightness variation
    cos_lr=True,                  # Added cosine learning rate scheduler
    close_mosaic=15,              # Increased from 10 to 15
    patience=35,                  # Increased from 25
    single_cls=True,              # Keep single-class mode
    device='',                    # Let YOLO automatically detect available devices
    workers=4,                    # Reduced from default 8 to avoid potential issues
    verbose=True,                 # Keep verbose mode
    amp=True,                     # Keep mixed precision
    exist_ok=True,                # Allow overwriting existing experiment
    cache=False,                  # Disable caching to reduce memory usage
    resume=False                  # Don't resume from previous checkpoint
)

# 5. Validate on Test Set
metrics = model.val(
    data='/root/.cache/kagglehub/datasets/shatanshur/my-data/versions/1/Final_Data_split/data.yaml',
    split='test',  # Validate on test set
    conf=0.25,     # Confidence threshold
    iou=0.45       # IoU threshold
)

# Print key metrics
print(f"mAP50-95: {metrics.box.map}")
print(f"mAP50: {metrics.box.map50}")

# 6. Export to Core ML (for iOS)
model.export(
    format='coreml',
    imgsz=640,
    nms=True,       # Include NMS in the model
    task='detect'   # Specify object detection task
)

'''import shutil

# Compress the runs folder
shutil.make_archive('runs', 'zip', '/content/runs')

# Download the zip file
files.download('runs.zip')'''