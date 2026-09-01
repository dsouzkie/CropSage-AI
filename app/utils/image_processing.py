"""
CropSage - Image Processing Utilities
Handles image loading, preprocessing, and validation for model inference.
"""

import io
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torchvision import transforms
from PIL import Image

logger = logging.getLogger(__name__)

# MobileNetV2 expected input
TARGET_SIZE: Tuple[int, int] = (224, 224)


# Standard ImageNet normalization for PyTorch models
_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# PyTorch Inference Transform
_INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    _NORMALIZE
])


def validate_image(image_bytes: bytes) -> bool:
    """
    Check if the uploaded bytes are a valid image.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
        return True
    except (UnidentifiedImageError, ValueError) as e:
        logger.warning(f"Invalid image uploaded: {e}")
        return False


def preprocess_image(image_bytes: bytes) -> Optional[torch.Tensor]:
    """
    Preprocess image bytes for PyTorch MobileNetV2 inference.
    """
    try:
        # Load image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Apply transforms
        tensor = _INFERENCE_TRANSFORM(img)
        
        # Add batch dimension: shape becomes (1, 3, 224, 224)
        tensor = tensor.unsqueeze(0)
        
        return tensor
    except Exception as e:
        logger.error(f"Failed to preprocess image: {e}")
        return None


def load_image_from_path(filepath: str) -> Optional[torch.Tensor]:
    """
    Load and preprocess an image from a local file path.
    """
    try:
        with open(filepath, "rb") as f:
            image_bytes = f.read()
        return preprocess_image(image_bytes)
    except Exception as e:
        logger.error(f"Failed to load image from {filepath}: {e}")
        return None
