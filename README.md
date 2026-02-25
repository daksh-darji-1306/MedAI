# MedAI: Chest X-Ray Classification

MedAI is a medical image classification project that uses deep learning to identify abnormalities in chest X-rays. Using a state-of-the-art EfficientNet-B0 backbone, it categorizes images into three classes (e.g., Normal, Pneumonia, Abnormal). It includes a fully-featured training pipeline, evaluation scripts, interpretability with Grad-CAM, and a Streamlit-based web application for easy interaction.

## 🏗️ Project Architecture

### 🧠 Model (`models/efficientnet_model.py`)
The project utilizes an **EfficientNet-B0** backbone for high accuracy with lower computational requirements.

- **Pretrained Weights**: Uses default ImageNet weights (`EfficientNet_B0_Weights.DEFAULT`).
- **Feature Extractor**: Extracts features up to the last convolutional block.
- **Global Average Pooling**: Reduces spatial dimensions robustly (`AdaptiveAvgPool2d`).
- **Custom Classifier**: 
  - Fully Connected Layer (128 units)
  - ReLU Activation
  - Dropout (0.5) to prevent overfitting
  - Final Fully Connected Layer (3 output classes)

### 🏋️ Training Pipeline (`train.py`)
- **Optimizer**: Adam with L2 Regularization (weight decay `1e-4`).
- **Loss Function**: Cross-Entropy Loss.
- **Learning Rate Scheduler**: `ReduceLROnPlateau` to decrease LR when validation loss plateaus.
- **Early Stopping**: Halts training if validation accuracy doesn't improve for 4 epochs.
- **Data Augmentation**: `RandomResizedCrop`, `RandomHorizontalFlip`, `RandomRotation`, and `ColorJitter` applied to training data to improve generalization.

### 🌐 Web Interface (`app.py`)
A **Streamlit** application that allows users to:
- Upload Chest X-Ray images (JPG, PNG).
- Receive real-time predictions and confidence scores for the 3 classes.
- View interpretability through **Grad-CAM heatmaps**, highlighting which areas of the X-ray the model focused on to make its prediction.

## 📁 File Structure

```text
MedAI/
│
├── data/                       # Directory for dataset
│   ├── Train/train/            # Training data (subfolders for each class: Normal, Pneumonia, Abnormal)
│   └── Test/test/              # Testing/Validation data (subfolders for each class)
│
├── models/
│   └── efficientnet_model.py   # Model architecture definition (EfficientNet-B0 + Custom Head)
│
├── app.py                      # Streamlit web application with Grad-CAM visualization
├── train.py                    # Training script with augmentation, early stopping, and LR scheduling
├── evaluate.py                 # Script to evaluate model performance on the test set
├── gradcam.py                  # Standalone module to generate Grad-CAM heatmaps
│
├── best_model.pth              # Saved PyTorch model weights (generated after training)
├── confusion_matrix.png        # Generated plot of the evaluation confusion matrix
├── gradcam_result.jpg          # Generated output image from standalone gradcam script
├── requirements.txt            # Python dependencies (assuming standard setup)
└── README.md                   # Project documentation
```

## 🚀 Getting Started

### Prerequisites

Ensure you have Python installed along with the necessary libraries. You can install the dependencies using:

```bash
pip install torch torchvision streamlit pillow numpy opencv-python grad-cam
```

### 1. Data Setup
Place your chest X-ray images in the `data/` directory. Ensure they are structured by class:
```
data/Train/train/Normal/
data/Train/train/Pneumonia/
data/Train/train/Abnormal/
```

### 2. Training the Model
To train the EfficientNet model on your dataset, run:
```bash
python train.py
```
This will automatically save the best performing model weights to `best_model.pth`.

### 3. Evaluating the Model
Evaluate the trained model on your test dataset:
```bash
python evaluate.py
```

### 4. Running the Web App
Launch the interactive Streamlit application to predict on new images and view Grad-CAM heatmaps:
```bash
streamlit run app.py
```

## 🔍 Interpretability (Grad-CAM)
MedAI integrates Grad-CAM (Gradient-weighted Class Activation Mapping) to provide visual explanations for the model's decisions. Heatmaps are generated overlaying the original X-ray, highlighting crucial regions that led to the specific classification. This is critical for medical AI to ensure trustworthiness.