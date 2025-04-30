import cv2
import torch
import numpy as np
from ultralytics import YOLO
import argparse
import time
import os

def detect_potholes_in_video(video_path, output_path=None, conf_threshold=0.25):
    """
    Detects potholes in a video using YOLOv11.
    
    Args:
        video_path (str): Path to the input video
        output_path (str, optional): Path to save the output video. If None, output is not saved.
        conf_threshold (float): Confidence threshold for detections
    """
    # Load the YOLOv11 model
    print("Loading YOLOv11 model...")
    try:
        model = YOLO("/Users/shatanshuraj/Desktop/Coding/Pothole Detection/detect/train/weights/best.pt")  # Load the nano version of YOLO11
        # Alternatively, you can use "yolo11s.pt", "yolo11m.pt", or "yolo11l.pt" based on your needs
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have the latest Ultralytics package installed: pip install ultralytics>=8.1.0")
        return

    # Open the video file
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup video writer if output path is provided
    writer = None
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process the video
    frame_count = 0
    start_time = time.time()
    
    print("Processing video frames...")
    while cap.isOpened():
        success, frame = cap.read()
        
        if not success:
            break
        
        frame_count += 1
        if frame_count % 30 == 0:  # Print status every 30 frames
            elapsed = time.time() - start_time
            fps_processed = frame_count / elapsed
            remaining = (total_frames - frame_count) / fps_processed if fps_processed > 0 else 0
            print(f"Processing frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%) - ETA: {remaining:.1f}s")
        
        # Perform detection
        results = model(frame, conf=conf_threshold, verbose=False)
        
        # Process the results
        annotated_frame = frame.copy()
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                # Get class name - YOLOv11 might need to be fine-tuned for pothole detection
                # or you might need to identify which class_id corresponds to potholes
                class_name = result.names[class_id]
                
                # Only proceed if the detection is a pothole (or the class you're interested in)
                # You may need to adjust this based on your specific model
                if "pothole" in class_name.lower() or True:  # Remove 'True' once you know the correct class name
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add label
                    label = f"{class_name}: {confidence:.2f}"
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(annotated_frame, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), (0, 255, 0), -1)
                    cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Display the frame
        cv2.imshow("YOLOv11 Pothole Detection", annotated_frame)
        
        # Write the frame to the output video if requested
        if writer:
            writer.write(annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    print(f"Processed {frame_count} frames in {time.time() - start_time:.2f} seconds")
    if output_path:
        print(f"Output video saved to: {output_path}")

if __name__ == "__main__":
    input_video = "/Users/shatanshuraj/Desktop/Coding/Pothole Detection/2c800243-6664-4855-ab12-036a9cb4b8f5.MP4"  # Replace with your video path
    output_video = "/Users/shatanshuraj/Desktop/Coding/Pothole Detection/sample_video_output.mp4"  # Replace with your output path
    confidence = 0.3  # Replace with your desired confidence threshold
    
    detect_potholes_in_video(input_video, output_video, confidence)