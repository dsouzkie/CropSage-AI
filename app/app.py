import os
import sys
import json
import base64
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from geopy.geocoders import Nominatim
from fpdf import FPDF

# Add the app folder to sys.path to resolve imports correctly
sys.path.append(str(Path(__file__).resolve().parent))

from utils.image_processing import preprocess_image
from utils.prediction import predict, format_class_name

st.set_page_config(page_title="CropSage AI", page_icon="🌿", layout="wide")

# ==========================================
# PREMIUM CUSTOM CSS STYLING
# ==========================================
st.markdown("""
<style>
    /* Main Background & Text */
    .stApp {
        background-color: #121212;
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sleek Headers */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Premium Gradient Button */
    .stButton>button {
        background: linear-gradient(135deg, #4caf50 0%, #2e7d32 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.5);
        border: none;
        color: white;
    }
    
    /* File Uploader Customization */
    .stFileUploader {
        background-color: #1e1e1e;
        border: 2px dashed #333333;
        border-radius: 12px;
        padding: 20px;
        transition: all 0.3s ease;
    }
    .stFileUploader:hover {
        border-color: #4caf50;
        background-color: #252525;
    }
    
    /* Weather Metric Cards */
    [data-testid="stMetric"] {
        background-color: #1e1e1e;
        border: 1px solid #333333;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    [data-testid="stMetricLabel"] {
        color: #9e9e9e !important;
        font-weight: 500;
    }
    [data-testid="stMetricValue"] {
        color: #4caf50 !important;
        font-weight: 700;
    }
    
    /* Chat Message Bubbles */
    .stChatMessage {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 4px solid #4caf50;
    }
    
    /* Expanders (Agent Thoughts) */
    .streamlit-expanderHeader {
        background-color: #1e1e1e !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(76, 175, 80, 0.5);
        color: white;
    }
    
    /* Input Fields */
    .stTextInput>div>div>input {
        border-radius: 6px;
        border: 1px solid #333;
    }
    
    /* Typography */
    h1, h2, h3 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# Constants
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = str(BASE_DIR / "notebooks" / "outputs" / "models" / "crop_disease_model.pth")
CLASS_INDICES_PATH = str(BASE_DIR / "notebooks" / "outputs" / "class_indices.json")

def draw_speedometer(confidence: float, title: str):
    """Draws a plotly gauge chart adapted for Dark Mode."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'color': '#e0e0e0', 'size': 18}},
        number = {'font': {'color': '#4caf50'}, 'suffix': '%', 'valueformat': '.1f'},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#4caf50"},
            'bar': {'color': "#4caf50"},
            'bgcolor': "#333333",
            'steps': [
                {'range': [0, 70], 'color': "#e74c3c"},
                {'range': [70, 90], 'color': "#f39c12"},
                {'range': [90, 100], 'color': "#2ecc71"}
            ],
        }
    ))
    fig.update_layout(
        height=250, 
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="#121212",
        font={'color': "#e0e0e0"}
    )
    return fig

def get_weather_data(location: str):
    """Fetches live weather data for the dashboard."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key: return None
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
        res = requests.get(url).json()
        if res.get("cod") == 200:
            return res
    except:
        pass
    return None

def create_pdf_report(disease, advice_text):
    """Generates a PDF report from the agent's advice."""
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", "B", 20)
    pdf.set_text_color(46, 125, 50)
    pdf.cell(0, 10, "CropSage Treatment Report", ln=True, align="C")
    
    pdf.set_font("Arial", "I", 12)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, f"Diagnosis: {disease}", ln=True, align="C")
    pdf.ln(10)
    
    # Body
    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(0, 0, 0)
    
    # Clean up markdown for PDF (basic cleanup)
    clean_text = advice_text.replace("**", "").replace("*", "-").replace("###", "\n")
    
    # FPDF's default font doesn't support emojis. We safely encode and ignore them.
    clean_text = clean_text.encode('latin-1', 'ignore').decode('latin-1')
    
    pdf.multi_cell(0, 8, clean_text)
    
    return bytes(pdf.output(dest='S'))

def main():
    st.title("🌿 CropSage Agritech Dashboard")
    st.markdown("Upload a leaf photo to receive an instant AI diagnosis, local weather analysis, and a personalized chat session with our agronomy expert.")

    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar Dashboard
    with st.sidebar:
        st.header("⚙️ Farm Profile")
        location = st.text_input("Farm Location", placeholder="e.g., London, UK")
        organic_only = st.checkbox("Organic Farming Only", value=False)
        st.markdown("---")
        
        # Weather Dashboard & Map
        if location:
            weather = get_weather_data(location)
            if weather:
                st.subheader("🌤️ Live Weather")
                colA, colB = st.columns(2)
                colA.metric("Temperature", f"{weather['main']['temp']}°C")
                colB.metric("Humidity", f"{weather['main']['humidity']}%")
                st.metric("Wind Speed", f"{weather['wind']['speed']} m/s")
                
                # Render Map
                try:
                    geolocator = Nominatim(user_agent="cropsage_app")
                    loc = geolocator.geocode(location)
                    if loc:
                        st.map({"lat": [loc.latitude], "lon": [loc.longitude]}, zoom=9, use_container_width=True)
                except:
                    pass
        
        st.markdown("---")
        st.markdown("**CropSage Engine:** MobileNetV2 + Gemini 3.6 Flash")

    # Main UI Layout
    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        st.subheader("1. Plant Scanner")
        upload_type = st.radio("Upload Source:", ["Upload File", "Take Picture"], horizontal=True)
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"]) if upload_type == "Upload File" else st.camera_input("Take a picture")

        if uploaded_file is not None:
            st.image(uploaded_file, caption="Scanned Leaf", use_container_width=True, clamp=True)
            
            if st.button("🔍 Diagnose Disease", type="primary", use_container_width=True):
                with st.spinner("Neural Network processing..."):
                    img_bytes = uploaded_file.getvalue()
                    tensor = preprocess_image(img_bytes)
                    if tensor is None:
                        st.error("Invalid image format.")
                        return
                    
                    try:
                        predictions = predict(tensor, MODEL_PATH, CLASS_INDICES_PATH, top_k=3)
                        top_disease = predictions[0][0]
                        confidence = predictions[0][1] * 100
                        
                        st.session_state['predictions'] = predictions
                        st.session_state['top_disease'] = top_disease
                        st.session_state['confidence'] = confidence
                        
                        # Reset chat history for a new diagnosis
                        st.session_state.messages = []
                        
                        # Trigger the initial agent response
                        agent_prompt = f"Hi CropSage. Your vision model just detected '{top_disease}' on my plant."
                        if location: agent_prompt += f" I am located in {location}."
                        if organic_only: agent_prompt += " I prefer organic methods ONLY."
                        
                        st.session_state.messages.append({"role": "user", "content": agent_prompt})
                        
                    except Exception as e:
                        st.error(f"Prediction Error: {str(e)}")

        # Show Gauge Chart and Top 3 if diagnosed
        if 'confidence' in st.session_state and 'predictions' in st.session_state:
            # Render Speedometer
            formatted_title = format_class_name(st.session_state['top_disease'])
            st.plotly_chart(draw_speedometer(st.session_state['confidence'], formatted_title), use_container_width=True)
            
            # Render Top 3 Predictions
            st.markdown("### Top 3 Predictions")
            from utils.visualization import create_confidence_chart
            st.plotly_chart(create_confidence_chart(st.session_state['predictions']), use_container_width=True)

    with col2:
        st.subheader("2. Agronomy Expert Chat")
        
        # Render Chat History
        for msg in st.session_state.messages:
            # We don't want to show the robotic initial prompt to the user, just their follow-ups.
            if msg["role"] == "user" and "Your vision model just detected" in msg["content"]:
                continue 
            st.chat_message(msg["role"]).write(msg["content"])

        # If we have a pending un-answered initial diagnosis prompt:
        if len(st.session_state.messages) == 1:
            with st.chat_message("assistant"):
                with st.spinner("Analyzing knowledge base & weather..."):
                    try:
                        from agent.farming_agent import get_farming_advice
                        
                        content = get_farming_advice(
                            disease_id=st.session_state['top_disease'],
                            location=location,
                            organic_only=organic_only,
                            chat_history=None
                        )
                        
                        # Parse out <thinking> blocks to display Agent Reasoning
                        parsed_text = ""
                        if isinstance(content, list):
                            for block in content:
                                if isinstance(block, dict) and "text" in block:
                                    parsed_text += block["text"]
                                elif isinstance(block, str):
                                    parsed_text += block
                        else:
                            parsed_text = content
                            
                        # Extract thinking block
                        import re
                        thinking_match = re.search(r'<thinking>(.*?)(?:</thinking>|$)', parsed_text, re.DOTALL)
                        if thinking_match:
                            thought_process = thinking_match.group(1).strip()
                            with st.expander("🧠 View Agent's Thought Process"):
                                st.markdown(f"*{thought_process}*")
                            # Remove the thinking block from the final output
                            cleaned_text = re.sub(r'<thinking>.*?(?:</thinking>|$)', '', parsed_text, flags=re.DOTALL).strip()
                            if cleaned_text:
                                parsed_text = cleaned_text
                            
                        st.write(parsed_text)
                        st.session_state.messages.append({"role": "assistant", "content": parsed_text})
                        st.rerun() # Refresh to show the PDF button
                    except Exception as e:
                        # Prevent infinite auto-retries by removing the trigger message
                        st.session_state.messages.pop()
                        
                        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                            st.error("⚠️ **Rate Limit Reached!** You clicked multiple times while it was broken, so Google has temporarily locked the key. **Please wait exactly 60 seconds**, then click 'Diagnose' again.")
                        else:
                            st.error(f"Agent Error: {str(e)}")
                            
        # Follow-up Chat Input
        if len(st.session_state.messages) > 1:
            # Allow PDF Download of the FIRST assistant response
            initial_advice = st.session_state.messages[1]["content"]
            pdf_bytes = create_pdf_report(st.session_state['top_disease'], initial_advice)
            st.download_button("📄 Download Treatment Plan (PDF)", data=pdf_bytes, file_name="CropSage_Report.pdf", mime="application/pdf")
            
            if user_input := st.chat_input("Ask a follow-up question... (e.g. 'Is it safe for pets?')"):
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.chat_message("user").write(user_input)
                
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            from agent.farming_agent import get_farming_advice
                            content = get_farming_advice(
                                disease_id=st.session_state['top_disease'],
                                location=location,
                                organic_only=organic_only,
                                chat_history=st.session_state.messages
                            )
                            
                            # Parse out <thinking> blocks to display Agent Reasoning
                            parsed_text = ""
                            if isinstance(content, list):
                                for block in content:
                                    if isinstance(block, dict) and "text" in block:
                                        parsed_text += block["text"]
                                    elif isinstance(block, str):
                                        parsed_text += block
                            else:
                                parsed_text = content
                                
                            # Extract thinking block
                            import re
                            thinking_match = re.search(r'<thinking>(.*?)(?:</thinking>|$)', parsed_text, re.DOTALL)
                            if thinking_match:
                                thought_process = thinking_match.group(1).strip()
                                with st.expander("🧠 View Agent's Thought Process"):
                                    st.markdown(f"*{thought_process}*")
                                # Remove the thinking block from the final output
                                cleaned_text = re.sub(r'<thinking>.*?(?:</thinking>|$)', '', parsed_text, flags=re.DOTALL).strip()
                                if cleaned_text:
                                    parsed_text = cleaned_text
                                
                            st.write(parsed_text)
                            st.session_state.messages.append({"role": "assistant", "content": parsed_text})
                        except Exception as e:
                            st.session_state.messages.pop() # Remove the failed user prompt
                            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                                st.error("⚠️ **Rate Limit Reached!** Please wait 60 seconds before sending another message.")
                            else:
                                st.error(f"Agent Error: {str(e)}")

if __name__ == "__main__":
    main()
