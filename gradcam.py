import os
import argparse
import cv2
import numpy as np
import torch
from torchvision import transforms
from models.efficientnet_model import ChestXRayEfficientNet

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    HAS_GRAD_CAM = True
except ImportError:
    HAS_GRAD_CAM = False

# Hyperparameters
NUM_CLASSES = 3
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_LOAD_PATH = os.path.join(BASE_DIR, "best_model.pth")

def main():
    parser = argparse.ArgumentParser(description='Grad-CAM for Chest X-Ray EfficientNet-B0')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--output_path', type=str, default='gradcam_result.jpg', help='Path to save output heatmap')
    args = parser.parse_args()

    if not HAS_GRAD_CAM:
        print("Error: 'grad-cam' package is required. Install using: pip install grad-cam")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    model = ChestXRayEfficientNet(num_classes=NUM_CLASSES)
    try:
        model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=device))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Warning: Model weights not found at {MODEL_LOAD_PATH}. Using untrained weights.")
    
    model = model.to(device)
    model.eval()

    # Target Layer for EfficientNet-B0 - typically the last feature map before global pool
    target_layers = [model.features[-1]]

    # Initialize GradCAM
    cam = GradCAM(model=model, target_layers=target_layers)

    # Load Image
    try:
        # Load as BGR, convert to RGB
        img = cv2.imread(args.image_path)
        if img is None:
             raise ValueError("Image not found or unable to read.")
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb_img = cv2.resize(rgb_img, (224, 224))
        rgb_img_float = np.float32(rgb_img) / 255
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Transforms (must match training)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(rgb_img_float).unsqueeze(0).to(device)

    # We want to explain the highest scoring class or a specific class
    # If targets is None, it targets the highest probability category.
    targets = None 

    # Generate CAM
    # You can also pass e.g. targets=[ClassifierOutputTarget(1)] for class 1
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # The result is a batch of CAMs, take the first one
    grayscale_cam = grayscale_cam[0, :]
    
    # Overlay on original image
    visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)
    
    # Save Image
    # Convert back to BGR for cv2.imwrite
    cv2.imwrite(args.output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"Grad-CAM visualization saved to {args.output_path}")

if __name__ == "__main__":
    main()
