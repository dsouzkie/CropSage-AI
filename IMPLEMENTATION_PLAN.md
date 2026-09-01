# Agentic AI-Based Smart Farming System for Crop Disease Detection

## 🎓 Project Context — Final Year Project (CHRIST University)

| Field | Value |
|---|---|
| **Title** | Agentic AI-Based Smart Farming System for Crop Disease Detection and Agricultural Productivity Enhancement |
| **University** | CHRIST (Deemed to be University) |
| **Review Stage** | Zeroth Review completed (PPT submitted) |
| **Dataset** | PlantVillage Dataset — 54,305 color images, 38 classes, ~0.79 GB |
| **Budget** | ₹0 — All hosting, APIs, and tools must be **completely free** |

---

## 1. PROJECT OBJECTIVES (from PPT)

1. Develop an **agentic AI-driven system** that autonomously detects crop diseases from leaf images using deep learning and computer vision.
2. Enable **multi-step, tool-using AI agents** that reason over image, sensor, and environmental data for accurate diagnosis.
3. Generate **intelligent, context-aware farming recommendations** — fertilizers, pesticides, irrigation — based on detected conditions.
4. Design a **farmer-friendly web/mobile interface** for real-time disease alerts and actionable guidance.
5. Enhance overall **agricultural productivity and sustainability** through automated, data-driven decision-making.

---

## 2. RESEARCH GAPS ADDRESSED (from Literature Survey)

This project fills the following gaps identified across 12 IEEE/Springer/ACM papers:

| Gap | How We Address It |
|---|---|
| No agentic reasoning in disease detection pipelines | LLM-based agentic pipeline with tool-use (ReAct pattern) |
| Diagnosis-only systems with no recommendations | Context-aware treatment + fertilizer + irrigation recommendations via LLM agent |
| No multi-crop support | 14 crop species, 38 disease/healthy classes |
| No farmer-facing real-time interface | Web app with image upload → instant diagnosis + recommendations |
| No autonomous multi-step decision-making | Agentic AI orchestrator that chains: image analysis → disease lookup → weather context → recommendation generation |

---

## 3. DATASET — PlantVillage

### 3.1 Overview

| Property | Value |
|---|---|
| **Source** | PlantVillage (public, open-access) |
| **Variants Available** | `color/`, `grayscale/`, `segmented/` |
| **Variant We Use** | `color/` (RGB images, most information-rich) |
| **Total Images** | 54,305 |
| **Total Size** | ~0.79 GB |
| **Number of Classes** | 38 |
| **Image Format** | JPG |
| **Typical Image Size** | 256×256 px, ~8–20 KB each |
| **Crops Covered** | Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato |

### 3.2 Class Distribution (All 38 Classes)

| # | Class Name | Count | Category |
|---|---|---|---|
| 1 | Apple___Apple_scab | 630 | Disease |
| 2 | Apple___Black_rot | 621 | Disease |
| 3 | Apple___Cedar_apple_rust | 275 | Disease |
| 4 | Apple___healthy | 1,645 | Healthy |
| 5 | Blueberry___healthy | 1,502 | Healthy |
| 6 | Cherry___Powdery_mildew | 1,052 | Disease |
| 7 | Cherry___healthy | 854 | Healthy |
| 8 | Corn___Cercospora_leaf_spot | 513 | Disease |
| 9 | Corn___Common_rust | 1,192 | Disease |
| 10 | Corn___Northern_Leaf_Blight | 985 | Disease |
| 11 | Corn___healthy | 1,162 | Healthy |
| 12 | Grape___Black_rot | 1,180 | Disease |
| 13 | Grape___Esca_(Black_Measles) | 1,383 | Disease |
| 14 | Grape___Leaf_blight | 1,076 | Disease |
| 15 | Grape___healthy | 423 | Healthy |
| 16 | Orange___Haunglongbing | 5,507 | Disease |
| 17 | Peach___Bacterial_spot | 2,297 | Disease |
| 18 | Peach___healthy | 360 | Healthy |
| 19 | Pepper___Bacterial_spot | 997 | Disease |
| 20 | Pepper___healthy | 1,478 | Healthy |
| 21 | Potato___Early_blight | 1,000 | Disease |
| 22 | Potato___Late_blight | 1,000 | Disease |
| 23 | Potato___healthy | 152 | Healthy |
| 24 | Raspberry___healthy | 371 | Healthy |
| 25 | Soybean___healthy | 5,090 | Healthy |
| 26 | Squash___Powdery_mildew | 1,835 | Disease |
| 27 | Strawberry___Leaf_scorch | 1,109 | Disease |
| 28 | Strawberry___healthy | 456 | Healthy |
| 29 | Tomato___Bacterial_spot | 2,127 | Disease |
| 30 | Tomato___Early_blight | 1,000 | Disease |
| 31 | Tomato___Late_blight | 1,909 | Disease |
| 32 | Tomato___Leaf_Mold | 952 | Disease |
| 33 | Tomato___Septoria_leaf_spot | 1,771 | Disease |
| 34 | Tomato___Spider_mites | 1,676 | Disease |
| 35 | Tomato___Target_Spot | 1,404 | Disease |
| 36 | Tomato___Yellow_Leaf_Curl_Virus | 5,357 | Disease |
| 37 | Tomato___Tomato_mosaic_virus | 373 | Disease |
| 38 | Tomato___healthy | 1,591 | Healthy |

**Class Imbalance Notes:**
- **Smallest class:** Potato___healthy (152 images)
- **Largest class:** Orange___Haunglongbing (5,507 images)
- **Ratio:** ~36:1 imbalance → Must use data augmentation and/or class weighting

### 3.3 Data Split Strategy

| Split | Percentage | Approx. Images |
|---|---|---|
| Training | 80% | ~43,444 |
| Validation | 10% | ~5,430 |
| Test | 10% | ~5,431 |

- Use **stratified splitting** to preserve class distribution across splits.
- Apply **data augmentation** only on training set (rotation, flip, zoom, brightness, contrast).

---

## 4. SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────┐
│                    FARMER / USER                            │
│              (Browser / Mobile Browser)                     │
└──────────────────────┬──────────────────────────────────────┘
                       │  Upload leaf image
                       ▼
┌─────────────────────────────────────────────────────────────┐
│               FRONTEND (React / Streamlit)                  │
│         Hosted on: Vercel / Streamlit Cloud                 │
│                                                             │
│  • Image upload widget                                      │
│  • Disease prediction display                               │
│  • Confidence scores + visual chart                         │
│  • Agentic AI recommendation panel                          │
│  • Chat interface for follow-up questions                   │
└──────────────────────┬──────────────────────────────────────┘
                       │  API calls
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              BACKEND (FastAPI / Flask)                       │
│       Hosted on: Hugging Face Spaces / Render               │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Image       │  │  Disease     │  │  Agentic AI      │  │
│  │  Preprocessor│─▶│  Classifier  │─▶│  Orchestrator    │  │
│  │  (PIL/CV2)   │  │  (CNN Model) │  │  (LangChain/     │  │
│  │              │  │              │  │   Custom ReAct)   │  │
│  └──────────────┘  └──────────────┘  └────────┬─────────┘  │
│                                                │            │
│                    ┌───────────────────────────┤            │
│                    ▼               ▼           ▼            │
│           ┌──────────────┐ ┌────────────┐ ┌──────────┐     │
│           │ Disease      │ │ Weather    │ │ LLM API  │     │
│           │ Knowledge    │ │ API (free) │ │ (Gemini  │     │
│           │ Base (JSON)  │ │            │ │  Free)   │     │
│           └──────────────┘ └────────────┘ └──────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### 4.1 Component Breakdown

| Component | Technology | Purpose |
|---|---|---|
| **Frontend** | Streamlit (recommended) OR React | Farmer-facing UI for image upload + results |
| **Backend** | FastAPI (Python) | REST API serving model inference + agent orchestration |
| **CNN Model** | MobileNetV2 / EfficientNet-B0 (PyTorch) | Image classification (38 classes) |
| **Agentic AI** | LangChain + Google Gemini API (free tier) | Multi-step reasoning agent for recommendations |
| **Knowledge Base** | JSON/SQLite file (bundled) | Disease info, treatments, fertilizers, pesticides |
| **Weather API** | OpenWeatherMap (free tier) | Environmental context for recommendations |
| **Hosting** | Hugging Face Spaces (Streamlit SDK) | Free, zero-cost deployment |

---

## 5. DEEP LEARNING MODEL

### 5.1 Model Selection: MobileNetV2 (Transfer Learning)

**Why MobileNetV2:**
- Pre-trained on ImageNet (1000 classes, millions of images)
- Extremely lightweight: **~3.4M parameters**, model file **~14 MB**
- Optimized for mobile/edge: depthwise separable convolutions
- Fits easily in free-tier hosting (Hugging Face Spaces: 16 GB RAM, 2 vCPU)
- State-of-the-art accuracy on PlantVillage (~97-99% reported in literature)

**Alternative (if MobileNetV2 underperforms):** EfficientNet-B0 (~5.3M params, ~20 MB)

### 5.2 Model Architecture

```
Input Image (224 × 224 × 3)
        │
        ▼
┌─────────────────────┐
│  MobileNetV2 Base   │  ← Pre-trained on ImageNet
│  (frozen initially) │  ← Unfreeze top layers for fine-tuning
│  include_top=False   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Global Average     │
│  Pooling 2D         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Dense(256, ReLU)   │
│  + BatchNorm        │
│  + Dropout(0.3)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Dense(38, Softmax) │  ← 38 output classes
└─────────────────────┘
```

### 5.3 Training Configuration

| Parameter | Value | Rationale |
|---|---|---|
| **Input Size** | 224 × 224 × 3 | MobileNetV2 default input size |
| **Batch Size** | 32 | Balance between memory and convergence |
| **Optimizer** | Adam | Adaptive learning rate, standard for transfer learning |
| **Learning Rate** | 1e-4 (fine-tuning), 1e-3 (head only) | Lower LR for pre-trained layers |
| **Loss Function** | Categorical Cross-Entropy | Multi-class classification |
| **Epochs** | 20–30 (with early stopping) | Prevent overfitting |
| **Early Stopping** | patience=5, monitor=val_loss | Auto-stop when validation loss plateaus |
| **LR Scheduler** | ReduceLROnPlateau (factor=0.5, patience=3) | Adaptive LR reduction |
| **Class Weights** | Computed from class distribution | Handle class imbalance |

### 5.4 Training Strategy (2-Phase)

**Phase 1: Feature Extraction (5 epochs)**
- Freeze all MobileNetV2 base layers
- Train only the custom head (Dense + Softmax)
- Learning rate: 1e-3

**Phase 2: Fine-Tuning (15-25 epochs)**
- Unfreeze top 50 layers of MobileNetV2
- Train with lower learning rate: 1e-4
- Early stopping with patience=5

### 5.5 Data Augmentation Pipeline

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(
    rescale=1./255  # Only rescaling for validation/test
)
```

### 5.6 Evaluation Metrics

| Metric | Target | Why |
|---|---|---|
| **Accuracy** | ≥ 95% | Overall correctness |
| **Precision** | ≥ 93% per class | Minimize false positives (wrong treatment) |
| **Recall** | ≥ 93% per class | Minimize false negatives (missed disease) |
| **F1-Score** | ≥ 93% macro-avg | Balanced performance across imbalanced classes |
| **Confusion Matrix** | Generate for all 38 classes | Identify problematic class pairs |
| **Top-3 Accuracy** | ≥ 99% | Allow showing top 3 predictions to farmer |

### 5.7 Model Export

- Save as **TensorFlow SavedModel** format for serving
- Also export as **TensorFlow Lite** (.tflite) for potential mobile deployment
- Save class labels mapping as `class_indices.json`

---

## 6. AGENTIC AI SYSTEM

### 6.1 What Makes This "Agentic"

Traditional ML pipeline: `Image → Model → Class Label → Done`

**Our Agentic pipeline:**
```
Image → Model → Disease Prediction
    │
    ▼
Agent receives prediction + confidence
    │
    ├─── Tool 1: Look up disease in Knowledge Base
    │         → Get disease description, severity, spread pattern
    │
    ├─── Tool 2: Fetch current weather data
    │         → Temperature, humidity, rainfall, season
    │
    ├─── Tool 3: Query treatment database
    │         → Organic & chemical treatment options
    │
    ├─── Tool 4: Reasoning step
    │         → Consider: disease severity + weather + crop stage
    │         → Generate personalized recommendation
    │
    └─── Output: Structured recommendation to farmer
              → Disease name & description
              → Severity assessment
              → Immediate actions
              → Fertilizer recommendations
              → Pesticide options (organic + chemical)
              → Irrigation adjustments
              → Preventive measures
```

### 6.2 Agent Architecture (ReAct Pattern)

```python
# Pseudo-code for the Agentic AI Orchestrator
class FarmingAgent:
    def __init__(self):
        self.llm = GoogleGenerativeAI(model="gemini-2.0-flash")  # Free tier
        self.tools = [
            DiseaseKnowledgeTool(),    # Looks up disease info from JSON KB
            WeatherTool(),              # Fetches weather from OpenWeatherMap
            TreatmentDatabaseTool(),    # Queries treatment options
            CropCalendarTool(),         # Provides crop stage info
        ]
    
    def diagnose_and_recommend(self, image, location=None):
        # Step 1: CNN prediction
        prediction = self.model.predict(image)
        disease_name = decode_prediction(prediction)
        confidence = prediction.max()
        
        # Step 2: Agent reasoning loop
        agent_prompt = f"""
        A farmer has uploaded a leaf image.
        CNN Prediction: {disease_name} (confidence: {confidence:.2%})
        Location: {location or 'Not provided'}
        
        Use your tools to:
        1. Look up details about this disease
        2. Check current weather conditions
        3. Find appropriate treatments
        4. Generate a comprehensive recommendation
        """
        
        response = self.agent.run(agent_prompt)
        return response
```

### 6.3 Knowledge Base Structure

Create a comprehensive JSON knowledge base bundled with the app:

```json
{
  "Apple___Apple_scab": {
    "disease_name": "Apple Scab",
    "scientific_name": "Venturia inaequalis",
    "crop": "Apple",
    "description": "Fungal disease causing olive-green to dark brown lesions on leaves and fruit.",
    "severity": "Medium-High",
    "spread": "Wind and rain splash; favored by cool, wet weather.",
    "symptoms": ["Olive-green spots on leaves", "Dark scabby lesions on fruit", "Premature leaf drop"],
    "treatments": {
      "organic": ["Neem oil spray", "Copper-based fungicide", "Sulfur spray"],
      "chemical": ["Mancozeb", "Captan", "Myclobutanil"],
      "cultural": ["Remove fallen leaves", "Prune for air circulation", "Avoid overhead irrigation"]
    },
    "fertilizer_recommendations": {
      "nitrogen": "Moderate (avoid excess, promotes susceptibility)",
      "potassium": "Increase for disease resistance",
      "calcium": "Apply lime if soil pH < 6.0"
    },
    "irrigation": "Avoid overhead watering; drip irrigation preferred",
    "prevention": ["Plant resistant varieties", "Apply preventive fungicide before rain", "Maintain good sanitation"]
  }
  // ... entries for all 38 classes
}
```

### 6.4 LLM Configuration (Google Gemini — Free Tier)

| Parameter | Value |
|---|---|
| **Model** | `gemini-2.0-flash` (free tier) |
| **Free Tier Limits** | 15 RPM, 1M TPM, 1,500 RPD |
| **Temperature** | 0.3 (factual, not creative) |
| **Max Output Tokens** | 1024 |
| **System Prompt** | See Section 6.5 below |

> **IMPORTANT:** 15 requests per minute is sufficient for a demo/academic project. For the review presentation, this is more than enough.

### 6.5 Agent System Prompt

```
You are CropSage, an expert agricultural advisor AI agent integrated into a smart farming 
disease detection system.

ROLE: You help farmers understand crop diseases detected from leaf images and provide 
actionable, context-aware recommendations.

BEHAVIOR RULES:
1. Always be factual and grounded in agricultural science.
2. Never recommend banned or highly toxic pesticides (WHO Class Ia/Ib).
3. Always suggest organic alternatives first, then chemical options.
4. Consider the farmer's location and weather when making recommendations.
5. Use simple, non-technical language that farmers can understand.
6. Always include preventive measures for the future.
7. If confidence is below 70%, advise the farmer to take another clear photo or consult 
   an agricultural extension officer.
8. Structure your response with clear sections: Diagnosis, Severity, Immediate Actions, 
   Treatment, Fertilizer, Irrigation, Prevention.

TOOLS AVAILABLE:
- disease_knowledge_base: Look up detailed disease information
- weather_api: Get current weather for the farmer's location
- treatment_database: Find recommended treatments
- crop_calendar: Get crop stage and seasonal advice

OUTPUT FORMAT: Always respond in structured markdown with clear headings.
```

---

## 7. TECHNOLOGY STACK

### 7.1 Complete Stack

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| **Language** | Python | 3.10+ | Primary language |
| **ML Framework** | TensorFlow / Keras | 2.15+ | Model training & inference |
| **Data Processing** | NumPy, Pandas | latest | Data manipulation |
| **Image Processing** | Pillow, OpenCV | latest | Image preprocessing |
| **Visualization** | Matplotlib, Seaborn | latest | Training plots, confusion matrix |
| **Web Framework** | Streamlit | 1.30+ | Full-stack web app (UI + backend) |
| **Agentic AI** | LangChain | 0.2+ | Agent orchestration framework |
| **LLM** | Google Gemini API | v2 | Free LLM for recommendations |
| **Weather API** | OpenWeatherMap | Free tier | Environmental data |
| **Version Control** | Git + GitHub | - | Code repository |
| **Model Training** | Local — NVIDIA RTX 4070 (8GB VRAM, CUDA 12.8) | - | Train on local GPU (faster than Colab T4) |
| **Hosting** | Hugging Face Spaces | Free (Streamlit SDK) | Deployment |

### 7.2 Python Dependencies (`requirements.txt`)

```
tensorflow==2.15.0
streamlit>=1.30.0
langchain>=0.2.0
langchain-google-genai>=1.0.0
google-generativeai>=0.5.0
Pillow>=10.0.0
opencv-python-headless>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
requests>=2.31.0
python-dotenv>=1.0.0
plotly>=5.18.0
```

---

## 8. FREE HOSTING STRATEGY (₹0 COST)

### 8.1 Recommended Stack: Hugging Face Spaces (Streamlit)

| Aspect | Detail |
|---|---|
| **Platform** | [Hugging Face Spaces](https://huggingface.co/spaces) |
| **SDK** | Streamlit |
| **Cost** | **Completely FREE** |
| **CPU** | 2 vCPU |
| **RAM** | 16 GB |
| **Storage** | 50 GB (via Git LFS for model files) |
| **Sleep Policy** | Sleeps after 48h of inactivity; wakes on next request (~30s cold start) |
| **Custom Domain** | Yes (free) |
| **Model Size Limit** | Up to 10 GB (our model is ~14 MB — no problem) |

### 8.2 Alternative Hosting Options (Backup)

| Platform | Free Tier | Best For | Limitations |
|---|---|---|---|
| **Streamlit Cloud** | Free for public repos | Streamlit apps | 1 GB RAM, may be slow for ML |
| **Render** | Free web service | Flask/FastAPI apps | 512 MB RAM, spins down after 15 min |
| **Vercel** | Free for frontend | React frontend only | No Python backend |
| **Railway** | $5 free credit/month | Full-stack | Credits expire; not truly free long-term |
| **Google Colab** | Free GPU | Training only | Not suitable for hosting a web app |

### 8.3 Free API Keys Needed

| API | Free Tier Limits | Sign-Up |
|---|---|---|
| **Google Gemini API** | 15 RPM, 1,500 RPD, 1M TPM | [aistudio.google.com](https://aistudio.google.com) |
| **OpenWeatherMap** | 1,000 calls/day | [openweathermap.org](https://openweathermap.org/api) |
| **Hugging Face** | Unlimited for Spaces | [huggingface.co](https://huggingface.co) |

### 8.4 Environment Variables (Secrets)

Set these in Hugging Face Spaces → Settings → Secrets:

```
GOOGLE_API_KEY=your_gemini_api_key
OPENWEATHER_API_KEY=your_openweathermap_key
```

> **NEVER** commit API keys to Git. Always use environment variables / platform secrets.

---

## 9. PROJECT FILE STRUCTURE

```
FYP/
├── dataset/
│   └── plantvillage dataset/
│       ├── color/            # ← We use this (54,305 images)
│       │   ├── Apple___Apple_scab/
│       │   ├── Apple___Black_rot/
│       │   └── ... (38 folders)
│       ├── grayscale/        # Available but not used
│       └── segmented/        # Available but not used
│
├── notebooks/                 # Jupyter notebooks for training
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_model_evaluation.ipynb
│   └── 04_model_export.ipynb
│
├── app/                       # Main application
│   ├── app.py                 # Streamlit main app
│   ├── model/
│   │   ├── crop_disease_model.h5      # Trained model
│   │   ├── class_indices.json         # Class label mapping
│   │   └── model_config.json          # Model metadata
│   ├── agent/
│   │   ├── farming_agent.py           # Agentic AI orchestrator
│   │   ├── tools.py                   # Agent tools (KB lookup, weather, etc.)
│   │   ├── prompts.py                 # System prompts and templates
│   │   └── knowledge_base.json        # Disease + treatment database
│   ├── utils/
│   │   ├── image_processing.py        # Image preprocessing utilities
│   │   ├── prediction.py              # Model inference utilities
│   │   └── visualization.py           # Charts and result display
│   └── config.py                      # App configuration
│
├── models/                    # Saved models directory
│   ├── mobilenetv2_plantvillage.h5
│   ├── mobilenetv2_plantvillage.tflite
│   └── training_history.json
│
├── docs/                      # Documentation
│   ├── IMPLEMENTATION_PLAN.md         # THIS FILE
│   └── literature_survey.md
│
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variable template
├── .gitignore                 # Git ignore rules
├── README.md                  # Project README
└── Agentic AI Smart Farming - Crop Disease Detection.pptx  # Zeroth Review PPT
```

---

## 10. IMPLEMENTATION PHASES & TIMELINE

### Phase 1: Data Preparation & EDA (Week 1–2)

- [ ] Set up Python environment with all dependencies
- [ ] Exploratory Data Analysis (EDA) notebook
  - Class distribution bar chart
  - Sample images per class (grid visualization)
  - Image size/resolution analysis
  - Check for corrupted images
- [ ] Data split: 80/10/10 stratified train/val/test
- [ ] Data augmentation pipeline
- [ ] Save split metadata (CSV with file paths + labels)

### Phase 2: Model Training (Week 2–3)

- [ ] Set up Google Colab notebook with GPU runtime
- [ ] Upload dataset to Google Drive (mount in Colab)
- [ ] Implement MobileNetV2 transfer learning architecture
- [ ] Phase 1 training: Feature extraction (frozen base, 5 epochs)
- [ ] Phase 2 training: Fine-tuning (unfrozen top layers, 15-25 epochs)
- [ ] Monitor training with TensorBoard / matplotlib plots
- [ ] Evaluate on test set: accuracy, precision, recall, F1, confusion matrix
- [ ] Export model as `.h5` and `.tflite`
- [ ] Save `class_indices.json`
- [ ] Target: **≥ 95% test accuracy**

### Phase 3: Knowledge Base & Agent Tools (Week 3–4)

- [ ] Create comprehensive `knowledge_base.json` (all 38 classes)
  - Disease descriptions, symptoms, treatments (organic + chemical)
  - Fertilizer recommendations, irrigation guidance, prevention
- [ ] Implement agent tools:
  - `DiseaseKnowledgeTool`: JSON lookup
  - `WeatherTool`: OpenWeatherMap API integration
  - `TreatmentDatabaseTool`: Treatment query
  - `CropCalendarTool`: Seasonal advice
- [ ] Set up Google Gemini API integration
- [ ] Implement ReAct agent with LangChain
- [ ] Test agent pipeline end-to-end locally

### Phase 4: Web Application (Streamlit) ✅ DONE
- **Framework:** Streamlit for rapid, data-centric UI development.
- **Features:**
  - File uploader (Drag & Drop + Camera input).
  - Model inference visualization (Top 3 Predictions chart & Plotly Speedometer Gauge).
  - **Advanced Weather Dashboard:** Display live temperature, humidity, and wind metric cards for the user's location.
  - **Interactive Map:** Render a geospatial map of the user's location.
  - **Chatbot Interface:** Interactive chat session so the user can ask follow-up questions to the CropSage Agent.
  - **PDF Export:** Allow users to download their treatment plan as a PDF report (with Unicode emoji sanitization).
  - **Premium Agritech Theming:** Sleek Dark Mode with custom CSS gradients, hover animations, and hidden Streamlit branding.
  - **Offline Graceful Degradation:** A local fallback synthesizer that triggers automatically if the cloud AI API hits a rate limit, ensuring 100% uptime.
- [x] Add error handling and input validation
- [x] Style with custom CSS (farmer-friendly, clean design)
- [x] Test locally end-to-end

### Phase 5: Deployment (Week 5–6)

- [ ] Create Hugging Face account and Space
- [ ] Push code to Hugging Face Spaces (Git)
- [ ] Upload model file via Git LFS
- [ ] Configure secrets (API keys)
- [ ] Test deployed app
- [ ] Set up custom README for the Space
- [ ] Create demo video / screenshots

### Phase 6: Documentation & Review Prep (Week 6–7)

- [ ] Update PPT for next review
- [ ] Write project report sections
- [ ] Prepare demo script
- [ ] Performance benchmarking (inference time, accuracy stats)
- [ ] Edge case testing (non-leaf images, blurry images, unknown crops)

---

## 11. RULES & CONSTRAINTS

### 11.1 Hard Constraints (Non-Negotiable)

| Rule | Details |
|---|---|
| **ZERO COST** | No paid services, APIs, or hosting. Everything must be free tier. |
| **PlantVillage Dataset** | Must use the provided PlantVillage dataset (54,305 color images, 38 classes). |
| **Agentic AI** | The system must demonstrate agentic behavior — multi-step reasoning, tool use, not just a simple chatbot wrapper. |
| **Web Interface** | Must have a working web interface accessible via URL. |
| **Python** | Primary language must be Python. |
| **Deep Learning** | Must use deep learning (CNN-based) for classification, not traditional ML. |
| **Academic Integrity** | All code must be original or properly attributed. No plagiarism. |

### 11.2 Technical Constraints

| Constraint | Limit | Impact |
|---|---|---|
| **HF Spaces RAM** | 16 GB | Model + app must fit in RAM. MobileNetV2 (~14 MB) is fine. |
| **HF Spaces CPU** | 2 vCPU (no GPU) | Inference must be fast on CPU. MobileNetV2 is optimized for this. |
| **HF Spaces Storage** | 50 GB | Dataset NOT deployed; only model + app code. |
| **Gemini Free Tier** | 15 RPM, 1,500 RPD | Rate-limit agent calls; add retry logic with backoff. |
| **OpenWeatherMap Free** | 1,000 calls/day | Cache weather data; don't call for every request. |
| **Cold Start** | ~30s on HF Spaces | Show loading spinner; inform user. |
| **Image Upload** | Max 10 MB per image | Validate and resize on upload. |
| **No GPU for Inference** | CPU only on free tier | MobileNetV2 inference: ~100-200ms on CPU (acceptable). |

### 11.3 Design Rules

| Rule | Rationale |
|---|---|
| **Organic treatments first** | Environmentally responsible; farmers prefer low-cost options. |
| **Never recommend WHO Class Ia/Ib pesticides** | Safety — highly hazardous to humans. |
| **Confidence threshold: 70%** | Below 70%, advise farmer to retake photo or consult expert. |
| **Top-3 predictions always shown** | Transparency — let farmer see alternatives. |
| **Simple language** | Target audience: farmers with varying education levels. |
| **Structured output** | Always use headings: Diagnosis → Severity → Actions → Treatment → Prevention. |
| **Graceful degradation** | If Gemini API is down, show CNN prediction + static KB info (no agent). |
| **No user data storage** | Privacy — uploaded images are processed and discarded. |

### 11.4 Code Quality Rules

| Rule | Standard |
|---|---|
| **Code style** | PEP 8 compliant |
| **Docstrings** | Required for all functions and classes |
| **Type hints** | Required for function signatures |
| **Error handling** | Try-except for all API calls, file I/O, model inference |
| **Logging** | Use Python `logging` module, not `print()` |
| **Secrets** | Use `.env` files locally, platform secrets in production |
| **Git commits** | Descriptive commit messages |
| **No hardcoded paths** | Use `os.path` or `pathlib` for all file paths |

---

## 12. RISK MITIGATION

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Gemini API rate limit hit | Medium | Recommendations fail | Implement retry with exponential backoff; fallback to static KB |
| HF Spaces goes down | Low | App unavailable | Have Streamlit Cloud as backup deployment |
| Model overfitting | Medium | Poor real-world performance | Data augmentation, early stopping, dropout, validation monitoring |
| Class imbalance affects accuracy | High | Low recall on minority classes | Class weights, oversampling small classes, augmentation |
| Cold start latency | Certain | 30s wait for first user | Show clear loading message; use HF Spaces "always on" if available |
| Non-leaf images uploaded | High | Wrong prediction | Add pre-validation: check if image contains a leaf (optional confidence gate) |
| API keys exposed | Low | Security breach | Use env vars; add `.env` to `.gitignore`; use HF Secrets |

---

## 13. EVALUATION CRITERIA (for Review Presentations)

### What Evaluators Will Look For:

1. **Technical Depth**: CNN architecture choice justification, training methodology, hyperparameter tuning.
2. **Innovation**: Agentic AI component — how it goes beyond simple classification.
3. **Working Demo**: Live web app with real-time predictions.
4. **Accuracy**: Test set performance metrics (accuracy, F1, confusion matrix).
5. **Practical Usefulness**: Are the recommendations actionable for farmers?
6. **Literature Alignment**: How does this address the identified research gaps?
7. **Code Quality**: Clean, documented, well-structured code.
8. **Presentation**: Clear explanation of methodology and results.

---

## 14. IEEE REFERENCES (from PPT)

1. A. Bonkra et al., "A Systematic Study: Implication of Deep Learning in Plant Disease Detection," IEEE CCET, 2022.
2. V. Balafas et al., "Machine Learning and Deep Learning for Plant Disease Classification and Detection," IEEE Access, vol. 11, 2023.
3. S. R. et al., "Integrated IoT and Machine Learning Solutions for Precision Farming," IEEE I-SMAC, 2024.
4. A. Oad et al., "Plant Leaf Disease Detection Using Ensemble Learning and Explainable AI," IEEE Access, vol. 12, 2024.
5. M. A. Khan et al., "Hybrid Approach of Cotton Disease Detection," IEEE Access, vol. 12, 2024.
6. S. Rekiek et al., "AI-Driven Pest Control and Disease Detection in Smart Farming Systems," Springer AI2SD, 2025.
7. L. Qin et al., "PDD-Agent: Multimodal LLM-Driven AI Agent for Enhanced Plant Disease Diagnosis," IEEE ICIP, 2025.
8. M. A. Arshad et al., "SAGE: Scalable Agentic Grounded Evaluation for Crop Disease Diagnosis," arXiv, 2026.
9. H. Xu et al., "AgriSentinel: Privacy-Enhanced Embedded-LLM Crop Disease Alerting System," ACM IH&MMSec, 2025.
10. A. S. Ibrahim et al., "AI-IoT Based Smart Agriculture Pivot for Plant Diseases Detection and Treatment," Scientific Reports, 2025.
11. S. Kothari et al., "CropGuard: Empowering Agriculture with AI Driven Plant Disease Detection Chatbot," IJISAE, 2024.
12. J. P. Nyakuri et al., "AI and IoT-Powered Edge Device Optimized for Crop Pest and Disease Detection," Scientific Reports, 2025.

---

## 15. QUICK REFERENCE — LLM CONTEXT REFRESH

> **Use this section to quickly remind any LLM about the project.**
>
> ⚠️ **ALSO READ:** `PROJECT_STATE.md` in the same folder — it tracks which phase
> is complete, what was last done, and what to do next. That file is the "save game."

### How to Start Any New LLM Session

Paste this prompt to any new LLM:
```
Read these 2 files before doing anything:
1. c:\Users\chris\Downloads\FYP\IMPLEMENTATION_PLAN.md  (project blueprint)
2. c:\Users\chris\Downloads\FYP\PROJECT_STATE.md         (progress tracker — where I left off)

Then help me continue from where I stopped. Update PROJECT_STATE.md when done.
```

### Project Summary Block
```
PROJECT: Agentic AI Smart Farming — Crop Disease Detection (FYP)
DATASET: PlantVillage, 54,305 color images, 38 classes (14 crops), JPG, 224x224
MODEL: MobileNetV2 transfer learning, ~14 MB, 38-class softmax output
AGENT: LangChain ReAct agent + Google Gemini 2.0 Flash (free tier)
TOOLS: Disease KB (JSON), Weather API (OpenWeatherMap), Treatment DB
FRONTEND: Streamlit web app
HOSTING: Hugging Face Spaces (free, 16GB RAM, 2 vCPU, no GPU)
BUDGET: ₹0 — everything must be free
KEY CONSTRAINT: Agentic AI must do multi-step reasoning (not just chatbot)
TARGET ACCURACY: ≥ 95% on test set
CONFIDENCE THRESHOLD: 70% — below this, advise retake/expert
SAFETY: No WHO Class Ia/Ib pesticides; organic options first
STATE FILE: PROJECT_STATE.md — READ THIS to know current phase & progress
```

---

*Last updated: 2026-09-01*
*This document serves as the single source of truth for the project. Refer to it when resuming work or providing this as context to any LLM assistant.*
