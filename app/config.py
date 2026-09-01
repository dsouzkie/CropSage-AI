"""
CropSage - App Configuration
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_DIR = BASE_DIR / "model"
AGENT_DIR = BASE_DIR / "agent"

# ============================================================
# Model Configuration
# ============================================================
MODEL_PATH = MODEL_DIR / "crop_disease_model.h5"
CLASS_INDICES_PATH = MODEL_DIR / "class_indices.json"
IMAGE_SIZE = (224, 224)  # MobileNetV2 input size
NUM_CLASSES = 38
CONFIDENCE_THRESHOLD = 0.70  # Below this, advise farmer to retake photo
TOP_K_PREDICTIONS = 3

# ============================================================
# Agentic AI Configuration
# ============================================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash"
LLM_TEMPERATURE = 0.3  # Factual, not creative
LLM_MAX_TOKENS = 1024

# ============================================================
# Weather API Configuration
# ============================================================
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

# ============================================================
# Knowledge Base
# ============================================================
KNOWLEDGE_BASE_PATH = AGENT_DIR / "knowledge_base.json"

# ============================================================
# App Settings
# ============================================================
APP_NAME = "CropSage"
APP_TAGLINE = "AI-Powered Crop Disease Detection & Smart Farming Advisor"
APP_ICON = "🌿"
MAX_UPLOAD_SIZE_MB = 10
