import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.efficientnet_model import ChestXRayEfficientNet
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Hyperparameters
BATCH_SIZE = 8
NUM_CLASSES = 3

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(BASE_DIR, "data/Test/test")
MODEL_LOAD_PATH = os.path.join(BASE_DIR, "best_model.pth")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=data_transforms)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    except FileNotFoundError:
        print(f"Warning: Test directory not found: {TEST_DIR}")
        return

    model = ChestXRayEfficientNet(num_classes=NUM_CLASSES)
    
    try:
        model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=device))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model weights not found at {MODEL_LOAD_PATH}. Train the model first.")
        return

    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    print("Evaluating model...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if len(test_dataset.classes) != NUM_CLASSES:
        print(f"Warning: Number of classes in dataset ({len(test_dataset.classes)}) does not match expected ({NUM_CLASSES}).")
    
    class_names = test_dataset.classes
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved as confusion_matrix.png")

if __name__ == "__main__":
    main()
