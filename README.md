# 🌿 CropSage — Agentic AI Smart Farming System

**Crop Disease Detection & Agricultural Productivity Enhancement**

> An agentic AI-powered system that detects crop diseases from leaf images using deep learning (MobileNetV2) and provides intelligent, context-aware farming recommendations through a multi-step reasoning AI agent.

## Features

- 🔬 **Disease Detection** — Upload a leaf photo, get instant diagnosis (38 diseases across 14 crops)
- 🤖 **Agentic AI Recommendations** — Multi-step reasoning agent provides treatments, fertilizers, irrigation advice
- 🌦️ **Weather-Aware** — Considers local weather for contextual recommendations
- 📱 **Mobile-Friendly** — Works on any phone browser with camera access
- 💰 **100% Free** — No paid APIs or hosting

## Tech Stack

| Component | Technology |
|---|---|
| ML Model | MobileNetV2 (Transfer Learning) |
| Agent | LangChain + Google Gemini 2.0 Flash |
| Frontend | Streamlit |
| Hosting | Hugging Face Spaces |
| Dataset | PlantVillage (54,305 images, 38 classes) |

## Quick Start

```bash
# Clone the repo
git clone <repo-url>
cd FYP

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the app
streamlit run app/app.py
```

## Project Structure

```
FYP/
├── app/                    # Main Streamlit application
│   ├── app.py              # Entry point
│   ├── model/              # Trained model + class indices
│   ├── agent/              # Agentic AI (LangChain + tools)
│   └── utils/              # Image processing, visualization
├── notebooks/              # Training & EDA notebooks
├── dataset/                # PlantVillage dataset (not in git)
├── requirements.txt
└── README.md
```

## License

This project is for academic purposes (Final Year Project).
