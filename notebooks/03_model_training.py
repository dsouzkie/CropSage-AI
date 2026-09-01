"""
03_model_training.py
PyTorch model training script for Crop Disease Detection using MobileNetV2.
"""

import os
import json
import logging
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE_DIR = Path(r"c:\Users\chris\Downloads\FYP")
DATASET_DIR = BASE_DIR / "dataset" / "plantvillage dataset" / "color"
SPLITS_DIR = Path(BASE_DIR) / "notebooks" / "outputs"
MODELS_DIR = SPLITS_DIR / "models"
METRICS_DIR = SPLITS_DIR / "metrics"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# Training params
BATCH_SIZE = 32
NUM_CLASSES = 38
PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Custom Dataset
# -----------------------------------------------------------------------------
class PlantDiseaseDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Ensure class indices are numeric
        if 'class_idx' not in self.data_frame.columns and 'label' in self.data_frame.columns:
            labels = sorted(self.data_frame['label'].unique())
            label_to_idx = {lbl: idx for idx, lbl in enumerate(labels)}
            self.data_frame['class_idx'] = self.data_frame['label'].map(label_to_idx)
            
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_rel_path = self.data_frame.iloc[idx]['filepath']
        img_path = self.root_dir / img_rel_path
        image = Image.open(img_path).convert('RGB')
        label = int(self.data_frame.iloc[idx]['class_index'])
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# -----------------------------------------------------------------------------
# Transforms
# -----------------------------------------------------------------------------
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def load_class_weights(json_path):
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                weights_dict = json.load(f)
            # Assuming weights_dict maps string class index to weight
            weights = [weights_dict.get(str(i), 1.0) for i in range(NUM_CLASSES)]
            logger.info(f"Loaded class weights from {json_path}")
            return torch.tensor(weights, dtype=torch.float32).to(DEVICE)
        except Exception as e:
            logger.warning(f"Failed to load class weights: {e}")
    return None

def train_model(model, dataloaders, criterion, optimizer, num_epochs, best_model_path):
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch+1}/{num_epochs}')
        logger.info('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            dl = tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Phase")
            for inputs, labels in dl:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                dl.set_postfix(loss=loss.item())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            logger.info(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"Saved new best model with accuracy {best_acc:.4f}")

    logger.info(f'Best val Acc: {best_acc:4f}')
    return model, history

def plot_metrics(history1, history2, save_path):
    epochs_p1 = len(history1['train_loss'])
    
    train_loss = history1['train_loss'] + history2['train_loss']
    val_loss = history1['val_loss'] + history2['val_loss']
    train_acc = history1['train_acc'] + history2['train_acc']
    val_acc = history1['val_acc'] + history2['val_acc']

    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.axvline(x=epochs_p1-1, color='r', linestyle='--', label='Unfreeze Base')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train Acc')
    plt.plot(val_acc, label='Val Acc')
    plt.axvline(x=epochs_p1-1, color='r', linestyle='--', label='Unfreeze Base')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    logger.info(f"Saved metrics plot to {save_path}")

def main():
    logger.info(f"Using device: {DEVICE}")
    
    train_csv = SPLITS_DIR / "train_split.csv"
    val_csv = SPLITS_DIR / "val_split.csv"
    
    if not train_csv.exists() or not val_csv.exists():
        logger.error(f"Missing split CSVs in {SPLITS_DIR}. Please ensure data splits are generated.")
        return

    train_dataset = PlantDiseaseDataset(train_csv, DATASET_DIR, data_transforms['train'])
    val_dataset = PlantDiseaseDataset(val_csv, DATASET_DIR, data_transforms['val'])
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    }

    # Model setup using modern torchvision weights API
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    
    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace classifier
    model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
    model = model.to(DEVICE)

    # Loss function with optional class weights
    weights_path = SPLITS_DIR / "class_weights.json"
    class_weights = load_class_weights(weights_path)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_model_path = MODELS_DIR / "crop_disease_model.pth"

    # -----------------------------------------
    # Phase 1: Train Classifier Only
    # -----------------------------------------
    logger.info("Starting Phase 1: Training classifier only...")
    optimizer_ft = optim.Adam(model.classifier.parameters(), lr=1e-3)
    
    model, history1 = train_model(
        model, dataloaders, criterion, optimizer_ft, 
        num_epochs=PHASE1_EPOCHS, best_model_path=best_model_path
    )

    # -----------------------------------------
    # Phase 2: Fine-tuning
    # -----------------------------------------
    logger.info("Starting Phase 2: Fine-tuning top layers...")
    # Unfreeze top layers (e.g., last few MobileNetV2 inverted residual blocks)
    for param in model.features[14:].parameters():
        param.requires_grad = True
        
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    # Load best model from Phase 1 to continue
    model.load_state_dict(torch.load(best_model_path, weights_only=True))

    model, history2 = train_model(
        model, dataloaders, criterion, optimizer_ft, 
        num_epochs=PHASE2_EPOCHS, best_model_path=best_model_path
    )
    
    # Plotting
    plot_path = METRICS_DIR / "training_history.png"
    plot_metrics(history1, history2, plot_path)
    logger.info("Training complete.")

if __name__ == '__main__':
    main()
