import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageOps
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
BASE_DIR = r"c:\Users\chris\Downloads\FYP"
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "plantvillage dataset", "color")
MODEL_PATH = os.path.join(BASE_DIR, "notebooks", "outputs", "models", "crop_disease_model.pth")
OUTPUT_DIR = os.path.join(BASE_DIR, "notebooks", "outputs", "evaluation")

os.makedirs(OUTPUT_DIR, exist_ok=True)

class PlantVillageTestDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data_frame)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx]['filepath'])
        image = Image.open(img_name).convert('RGB')
        label = int(self.data_frame.iloc[idx]['class_index'])
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def main():
    test_csv_path = os.path.join(BASE_DIR, "notebooks", "outputs", "test_split.csv")
    class_indices_path = os.path.join(BASE_DIR, "notebooks", "outputs", "class_indices.json")

    logging.info(f"Using test CSV: {test_csv_path}")

    # Transformations for evaluation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = PlantVillageTestDataset(csv_file=test_csv_path, root_dir=DATASET_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    with open(class_indices_path, "r") as f:
        class_indices = json.load(f)
    
    # Ensure classes are sorted by their integer index
    classes = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]
    num_classes = len(classes)
    logging.info(f"Found {num_classes} classes from JSON.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load model
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    
    logging.info(f"Loading model weights from {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    
    logging.info("Running inference on the test set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    logging.info("Generating evaluation reports...")
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    
    # Save JSON summary
    summary_path = os.path.join(OUTPUT_DIR, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(report, f, indent=4)
    logging.info(f"Saved evaluation summary to {summary_path}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved confusion matrix to {cm_path}")

    # Top 10 most confused pairs
    confused_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                confused_pairs.append({
                    'actual': classes[i],
                    'predicted': classes[j],
                    'count': int(cm[i, j])
                })
    
    confused_pairs = sorted(confused_pairs, key=lambda x: x['count'], reverse=True)[:10]
    logging.info("Top 10 confused pairs:")
    for pair in confused_pairs:
        logging.info(f"Actual: {pair['actual']} | Predicted: {pair['predicted']} | Count: {pair['count']}")

    # Save top 10 confused pairs
    confused_path = os.path.join(OUTPUT_DIR, "top_10_confused_pairs.json")
    with open(confused_path, "w") as f:
        json.dump(confused_pairs, f, indent=4)
        
    # Generate sample predictions grid (green/red border)
    logging.info("Generating sample predictions grid...")
    num_samples = min(16, len(test_dataset))
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        img_path = os.path.join(DATASET_DIR, test_dataset.data_frame.iloc[idx]['filepath'])
        actual_label = test_dataset.data_frame.iloc[idx]['class']
        
        # Load image for visualization
        pil_img = Image.open(img_path).convert('RGB')
        
        # Inference for a single image
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            _, pred = torch.max(output, 1)
            pred_label = classes[pred.item()]
            
        color = 'green' if actual_label == pred_label else 'red'
        
        # Add border
        bordered_img = ImageOps.expand(pil_img, border=10, fill=color)
        
        axes[i].imshow(bordered_img)
        title = f"Actual: {actual_label}\nPred: {pred_label}"
        axes[i].set_title(title, color=color, fontsize=8)
        axes[i].axis('off')
        
    plt.tight_layout()
    grid_path = os.path.join(OUTPUT_DIR, "sample_predictions.png")
    plt.savefig(grid_path)
    plt.close()
    logging.info(f"Saved sample predictions grid to {grid_path}")
    logging.info("Evaluation complete.")

if __name__ == "__main__":
    main()
