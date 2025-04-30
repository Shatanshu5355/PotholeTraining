from ultralytics import YOLO

# Load the model
model = YOLO('yolo11n.pt')

# Access the internal PyTorch model
pytorch_model = model.model

# Print the model structure
print(pytorch_model)
