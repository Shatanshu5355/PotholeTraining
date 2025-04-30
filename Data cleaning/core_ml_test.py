from ultralytics import YOLO
import coremltools as ct

# Load the Core ML model
model = ct.models.MLModel('model3333/All_Classes.mlpackage/Data/com.apple.CoreML/model.mlmodel')

# Inspect model inputs/outputs
print("Inputs:", model.input_description)
print("Outputs:", model.output_description)

# For more details, check the model spec
print(model._spec.description)  # Full model specification


'''import cv2
import numpy as np
from PIL import Image
import coremltools as ct

# Load Core ML model
model = ct.models.MLModel('detect_v2/train/weights/best_v2.mlpackage/Data/com.apple.CoreML/model.mlmodel')

# Video input setup
video_path = '2c800243-6664-4855-ab12-036a9cb4b8f5.MP4'  
cap = cv2.VideoCapture(video_path)

# Set thresholds (adjust these if needed)
conf_threshold = 0.25  # Default from model spec
iou_threshold = 0.35     # Default from model spec

while cap.isOpened():
    ret, frame = cap.read();
    if not ret:
        break

    # Preprocess frame
    resized_frame = cv2.resize(frame, (640, 640))
    pil_image = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))

    # Run inference with thresholds
    predictions = model.predict({
        'image': pil_image,
        'confidenceThreshold': conf_threshold,  # Optional: set custom threshold
        'iouThreshold': iou_threshold           # Optional: set custom threshold
    })

    # Get outputs
    confidences = predictions['confidence']    # Shape: (num_detections, 1)
    coordinates = predictions['coordinates']   # Shape: (num_detections, 4) [x_center, y_center, width, height] (relative coords)

    # Convert relative coordinates to absolute pixels
    h, w = frame.shape[:2]  # Original frame dimensions
    for i in range(len(confidences)):
        conf = confidences[i][0]
        if conf < conf_threshold:
            continue

        # Get box coordinates (relative to 640x640)
        x_center_rel, y_center_rel, width_rel, height_rel = coordinates[i]
        
        # Convert to absolute coordinates (original frame size)
        x_center = x_center_rel * w
        y_center = y_center_rel * h
        box_width = width_rel * w
        box_height = height_rel * h

        # Convert to (x1, y1, x2, y2)
        x1 = int(x_center - box_width/2)
        y1 = int(y_center - box_height/2)
        x2 = int(x_center + box_width/2)
        y2 = int(y_center + box_height/2)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'Pothole {conf:.2f}', (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display output
    cv2.imshow('Pothole Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()'''

'''# Load YOLOv11 model
model = YOLO('model3333/All_Classes.pt')
model.export(
    format='coreml',
    imgsz=640,
    nms=True,       # Include NMS in the model
    task='detect'
)'''