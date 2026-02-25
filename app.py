import os
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from models.efficientnet_model import ChestXRayEfficientNet

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    HAS_GRAD_CAM = True
except ImportError:
    HAS_GRAD_CAM = False

# Constants
NUM_CLASSES = 3
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_LOAD_PATH = os.path.join(BASE_DIR, "best_model.pth")
CLASSES = ["Normal", "Pneumonia", "Abnormal"] # Adjust depending on your actual dataset subfolders

st.set_page_config(page_title="Chest X-Ray Classifier", layout="wide")

@st.cache_resource
def load_model():
    """Load model with Streamlit caching to avoid reloading on every interaction."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChestXRayEfficientNet(num_classes=NUM_CLASSES)
    try:
        model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=device))
        model.to(device)
        model.eval()
        return model, device
    except FileNotFoundError:
        return None, device

def process_image(image):
    """Convert uploaded image to PyTorch tensor."""
    img_rgb = image.convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(img_rgb).unsqueeze(0)
    return input_tensor, img_rgb

def generate_gradcam(model, input_tensor, img_rgb):
    """Generate Grad-CAM visualization for interpretability."""
    if not HAS_GRAD_CAM:
        return None
    
    # Target the last feature map before global pool
    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Prepare image for visualization overlay
    img_cv = np.array(img_rgb)
    img_cv = cv2.resize(img_cv, (224, 224))
    img_float = np.float32(img_cv) / 255.0
    
    # Generate Heatmap
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
    visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
    
    return visualization

def main():
    st.title("🩺 Medical AI: Chest X-Ray Analysis")
    st.write("Upload a chest X-ray image to get an EfficientNet-B0 prediction and visualize the Grad-CAM heatmap.")

    model, device = load_model()
    
    if model is None:
        st.error(f"⚠️ Model weights not found at `{MODEL_LOAD_PATH}`. Please run `train.py` first.")
        st.stop()
        
    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
            
        with st.spinner('Analyzing image with EfficientNet-B0...'):
            input_tensor, img_rgb = process_image(image)
            input_tensor = input_tensor.to(device)
            
            with torch.no_grad():
                logits = model(input_tensor)
                probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
                
                prediction_idx = np.argmax(probabilities)
                predicted_class = CLASSES[prediction_idx]
                confidence = probabilities[prediction_idx] * 100
                
        st.write("---")
        st.header(f"Prediction: **{predicted_class}** ({confidence:.2f}% Confidence)")
        
        # Display probabilities bar charts
        st.write("Class Probabilities:")
        for idx, class_name in enumerate(CLASSES):
            st.progress(float(probabilities[idx]), text=f"{class_name}: {probabilities[idx]*100:.2f}%")
            
        st.write("---")
        
        with col2:
            st.subheader("Grad-CAM Explainability Heatmap")
            with st.spinner('Generating Heatmap...'):
                heatmap = generate_gradcam(model, input_tensor, img_rgb)
                if heatmap is not None:
                    st.image(heatmap, use_container_width=True, caption="Areas of Model Focus")
                else:
                    st.warning("Install the `grad-cam` library (`pip install grad-cam`) to view interpretability heatmaps.")

if __name__ == "__main__":
    main()
