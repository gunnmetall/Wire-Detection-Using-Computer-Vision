import os

# Define paths
model_weights = "best.pt"
custom_image_path = "image.jpg"
confidence_threshold = 0.25

# Build the inference command
inference_command = (
    f"!yolo task=segment mode=predict model={model_weights} "
    f"conf={confidence_threshold} source={custom_image_path} save=true"
)

# Execute the inference command
os.system(inference_command)
