"""
CropSage - Model Prediction Utilities
Handles loading the trained PyTorch model and running inference.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torchvision import models, transforms

logger = logging.getLogger(__name__)

# Global model cache — loaded once, reused across requests
_model = None
_class_indices: Optional[Dict[int, str]] = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path: str, num_classes: int = 38):
    """
    Load the trained PyTorch model. Caches it globally for reuse.
    """
    global _model
    if _model is not None:
        return _model

    try:
        logger.info(f"Loading PyTorch model from {model_path} onto {_device}...")
        
        # Initialize the model architecture
        import torchvision
        model = torchvision.models.mobilenet_v2(pretrained=False)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
        
        # Load weights
        state_dict = torch.load(model_path, map_location=_device, weights_only=True)
        model.load_state_dict(state_dict)
        
        model = model.to(_device)
        model.eval()  # Set to evaluation mode
        
        _model = model
        logger.info("PyTorch model loaded successfully.")
        return _model
    except Exception as e:
        logger.error(f"Failed to load PyTorch model: {e}")
        raise

def load_class_indices(class_indices_path: str) -> Dict[int, str]:
    """
    Load the class index → class name mapping from JSON.
    """
    global _class_indices
    if _class_indices is not None:
        return _class_indices

    try:
        with open(class_indices_path, "r") as f:
            name_to_idx = json.load(f)

        # Invert: {"Apple___healthy": 0} → {0: "Apple___healthy"}
        _class_indices = {int(v): k for k, v in name_to_idx.items()}
        logger.info(f"Loaded {len(_class_indices)} class labels.")
        return _class_indices
    except Exception as e:
        logger.error(f"Failed to load class indices: {e}")
        raise

def predict(
    img_tensor: torch.Tensor,
    model_path: str,
    class_indices_path: str,
    top_k: int = 3,
) -> List[Tuple[str, float]]:
    """
    Run inference on a preprocessed image tensor.
    """
    model = load_model(model_path)
    class_indices = load_class_indices(class_indices_path)

    # Ensure tensor is on the right device and has batch dimension
    if len(img_tensor.shape) == 3:
        img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(_device)

    # Run inference
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    # Get top-k indices sorted by probability
    top_k_probs, top_k_indices = torch.topk(probs, top_k)
    
    results = []
    for prob, idx in zip(top_k_probs, top_k_indices):
        idx_val = idx.item()
        class_name = class_indices.get(idx_val, f"Unknown_{idx_val}")
        confidence = prob.item()
        results.append((class_name, confidence))

    logger.info(f"Top prediction: {results[0][0]} ({results[0][1]:.2%})")
    return results

def format_class_name(raw_name: str) -> str:
    """
    Convert raw class name to human-readable format.
    """
    parts = raw_name.split("___")
    if len(parts) == 2:
        crop = parts[0].replace("_", " ")
        condition = parts[1].replace("_", " ").title()
        return f"{crop} — {condition}"
    return raw_name.replace("_", " ").title()

