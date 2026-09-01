import os
import json
import logging
import requests
from pathlib import Path
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
KNOWLEDGE_BASE_PATH = BASE_DIR / "app" / "agent" / "knowledge_base.json"

try:
    with open(KNOWLEDGE_BASE_PATH, "r") as f:
        KNOWLEDGE_BASE = json.load(f)
except Exception as e:
    logger.error(f"Could not load knowledge base: {e}")
    KNOWLEDGE_BASE = {}


def get_disease_info(disease_id: str) -> str:
    if disease_id in KNOWLEDGE_BASE:
        return json.dumps(KNOWLEDGE_BASE[disease_id], indent=2)
    for key, data in KNOWLEDGE_BASE.items():
        if data.get("disease", "").lower() in disease_id.lower() or disease_id.lower() in key.lower():
            return json.dumps(data, indent=2)
    return "Disease information not found in the local database."


def get_weather_forecast(location: str) -> str:
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key: return "Weather API key missing."
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
        data = requests.get(url).json()
        if data.get("cod") == 200:
            return f"{data['weather'][0]['description']}, {data['main']['temp']}°C, Wind: {data['wind']['speed']}m/s"
        return "Weather data unavailable."
    except:
        return "Weather service offline."


def get_farming_advice(disease_id: str, location: str, organic_only: bool, chat_history: list = None):
    """
    Makes EXACTLY ONE API call to Gemini to save quota.
    It fetches the tool data in Python first, then passes it as context.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0.2,
        max_retries=0
    )

    # 1. Manually fetch the tool data (Costs 0 API quota!)
    disease_data = get_disease_info(disease_id)
    weather_data = get_weather_forecast(location) if location else "No location provided."

    # 2. Build the context for the LLM
    system_prompt = f"""You are CropSage, an expert agronomy AI.
Your goal is to help the farmer treat their plant.

CONTEXT DATA:
- Predicted Disease: {disease_id}
- Database Info: {disease_data}
- Current Weather: {weather_data}
- User Prefers Organic Only: {organic_only}

RULES:
1. You MUST first think through the problem step-by-step. Write your internal reasoning inside a <thinking> block.
2. Inside <thinking>, analyze how the Current Weather impacts spraying, and how the Organic constraint limits your options.
3. After the <thinking> block, you MUST structure your final response to the user with EXACTLY these Markdown headings:
   - ### 🔬 Disease Overview: Briefly explain what this disease is, what plant it affects, and what pathogen/conditions cause it.
   - ### 🛡️ Immediate Treatment Plan: Provide actionable steps (factor in the organic constraint).
   - ### 🌍 Location-Specific Prevention: First, explicitly state the current climate/weather details. Then, recommend preventive measures and explain EXACTLY *why* they are necessary based on these specific weather conditions.
4. If Organic Only is True, do NOT suggest chemical treatments in your final plan.
5. Do not mention your 'context data' to the user in the final output. Just give the advice naturally.
"""

    messages = [SystemMessage(content=system_prompt)]
    
    # Add previous chat history for follow-ups
    if chat_history:
        for msg in chat_history:
            # Skip the initial boilerplate user message so it doesn't confuse the LLM
            if msg["role"] == "user" and "Your vision model just detected" in msg["content"]:
                messages.append(HumanMessage(content=f"I have {disease_id}. What should I do?"))
            elif msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                pass 
    
    # Ensure the latest prompt is always evaluated
    if chat_history:
        messages = [SystemMessage(content=system_prompt)]
        for msg in chat_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                from langchain_core.messages import AIMessage
                messages.append(AIMessage(content=msg["content"]))
    else:
        messages.append(HumanMessage(content=f"My plant was just diagnosed with {disease_id}. What should I do?"))

    # 3. Make ONE single call to Gemini with Graceful Degradation
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        error_msg = str(e).upper()
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "503" in error_msg or "UNAVAILABLE" in error_msg:
            logger.warning(f"Gemini API offline ({e}). Falling back to Local AI Synthesis.")
            # Graceful Fallback Mode: Parse the JSON and build a markdown response manually
            try:
                db_info = json.loads(disease_data)
                disease_name = db_info.get("disease", disease_id.replace("___", " ").replace("_", " "))
                
                fallback_response = f"<thinking>API Rate limit reached. Engaging offline fallback synthesis. Analyzing local knowledge base for {disease_name}. Weather context: {weather_data}. Organic preference: {organic_only}.</thinking>\n\n"
                fallback_response += f"### 🌱 Diagnosis: {disease_name}\n\n"
                
                fallback_response += "#### 🔎 Symptoms\n"
                for sym in db_info.get("symptoms", []):
                    fallback_response += f"- {sym}\n"
                    
                fallback_response += "\n#### 🛡️ Treatment Plan\n"
                if organic_only:
                    for trt in db_info.get("organic_treatment", []):
                        fallback_response += f"- **Organic:** {trt}\n"
                else:
                    for trt in db_info.get("chemical_treatment", []):
                        fallback_response += f"- **Chemical:** {trt}\n"
                    for trt in db_info.get("organic_treatment", []):
                        fallback_response += f"- **Organic:** {trt}\n"
                        
                fallback_response += "\n#### ⛅ Weather Advisory\n"
                fallback_response += f"Current conditions: {weather_data}. Please ensure you only spray when wind speeds are low to avoid drift, and avoid spraying immediately before rain."
                
                return fallback_response
            except:
                return f"<thinking>API offline and JSON parsing failed.</thinking>\n\n**Diagnosis:** {disease_id}\n\n*Our cloud AI is currently at maximum capacity. Please refer to standard agricultural guidelines for this disease.*"
        else:
            raise e


if __name__ == "__main__":
    # Test the agent locally
    agent = get_farming_agent()
    
    test_input = (
        "Hi CropSage! Your vision model just detected 'Tomato___Late_blight' on my leaves. "
        "I live in London, UK. What should I do? I prefer organic methods if possible."
    )
    
    print("\n--- Running Agent Test ---")
    response = agent.invoke({"messages": [{"role": "user", "content": test_input}]})
    print("\n--- Agent Response ---")
    # Langgraph agents return a dict with 'messages'
    print(response["messages"][-1].content)
