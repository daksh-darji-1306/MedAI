import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.efficientnet_model import ChestXRayEfficientNet

# Hyperparameters
BATCH_SIZE = 8
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
NUM_CLASSES = 3

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "data/Train/train")
TEST_DIR = os.path.join(BASE_DIR, "data/Test/test")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "best_model.pth")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Robust Data Augmentation to prevent overfitting
    data_transforms = {
        'Train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'Test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # DataLoaders
    # Note: Ensure data/Train and data/Test contain subfolders for each class: Normal, Pneumonia, Abnormal
    try:
        train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=data_transforms['Train'])
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

        val_dataset = datasets.ImageFolder(root=TEST_DIR, transform=data_transforms['Test'])
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    except FileNotFoundError:
        print(f"Warning: Ensure data directories exist and have proper subfolders for classes.")
        print(f"Expected structure: {TRAIN_DIR}/Normal, {TRAIN_DIR}/Pneumonia, {TRAIN_DIR}/Abnormal")
        return

    model = ChestXRayEfficientNet(num_classes=NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Added L2 Regularization (Weight Decay) to prevent large weights
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # Learning Rate Scheduler to reduce LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)

    best_acc = 0.0
    
    # Early Stopping parameters
    early_stop_patience = 4
    epochs_no_improve = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = corrects.double() / len(train_dataset)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                
        val_epoch_loss = val_loss / len(val_dataset)
        val_epoch_acc = val_corrects.double() / len(val_dataset)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

        # Step the scheduler based on validation loss
        scheduler.step(val_epoch_loss)

        # Save Best Model and check Early Stopping
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Saved Best Model at Epoch {epoch+1} with Acc: {best_acc:.4f}")
            epochs_no_improve = 0 # Reset patience
        else:
            epochs_no_improve += 1
            print(f"Validation accuracy did not improve for {epochs_no_improve} epochs.")
            
        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping triggered! Training stopped after {epoch+1} epochs.")
            break

if __name__ == "__main__":
    main()
