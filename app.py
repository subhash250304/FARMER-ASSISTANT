import os
import pickle
import numpy as np
import pandas as pd
import httpx
import openai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import List

# --- 1. App Setup ---
load_dotenv()
app = FastAPI(
    title="Smart Farm Assistant API",
    description="An intelligent API for crop/pesticide recommendations, weather data, and an AI-powered chatbot.",
    version="3.3.0" # Version updated with scaler and robust OpenAI client
)

# --- 2. CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# --- 3. Pydantic Models for Data Validation ---
class CropPredictionRequest(BaseModel):
    N: float = Field(..., example=90.0)
    P: float = Field(..., example=42.0)
    K: float = Field(..., example=43.0)
    temperature: float = Field(..., example=20.87)
    humidity: float = Field(..., example=82.0)
    ph: float = Field(..., example=6.5)
    rainfall: float = Field(..., example=202.9)

class ChatRequest(BaseModel):
    message: str

# --- 4. Load ML Models, Scaler, and Datasets on Startup ---
model = None
scaler = None
pesticide_df = None
openai_client = None
is_openai_configured = False

@app.on_event("startup")
def load_resources():
    global model, scaler, pesticide_df, openai_client, is_openai_configured
    
    # Load ML models and the crucial scaler
    try:
        with open("C:\\Users\\Admin\\Documents\\internship\\ezyZip (1)\\Backend\\Models\\crop_model.pkl", "rb") as f: model = pickle.load(f)
        with open("C:\\Users\Admin\\Documents\\internship\\ezyZip (1)\\Backend\\Scripts\\models\\scaler.pkl", "rb") as f: scaler = pickle.load(f)
        print("✅ Crop model and scaler loaded successfully.")
    except FileNotFoundError:
        print("❌ CRITICAL ERROR: 'crop_model.pkl' or 'scaler.pkl' not found. Please run preprocess.py and train_model.py.")
    
    # Load pesticide data
    try:
        pesticide_df = pd.read_csv("C:\\Users\\Admin\\Documents\\internship\\ezyZip (1)\\Backend\\Data\\pesticide_dataset.csv")
        print("✅ Pesticide dataset loaded successfully.")
    except FileNotFoundError:
        print("⚠️ Warning: Pesticide dataset not found. Pesticide endpoints will fail.")

    # Configure and validate OpenAI Client
    if os.getenv("OPENAI_API_KEY"):
        try:
            openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            openai_client.models.list() # Test call to validate the key
            is_openai_configured = True
            print("✅ OpenAI client configured and API key is valid.")
        except openai.AuthenticationError:
            print("⚠️ WARNING: OpenAI API key is invalid or deactivated. Falling back to simple chatbot.")
            is_openai_configured = False
        except Exception as e:
            print(f"⚠️ WARNING: An error occurred during OpenAI setup: {e}. Falling back to simple chatbot.")
            is_openai_configured = False
    else:
        print("ℹ️ Info: OPENAI_API_KEY not found in .env. Falling back to simple chatbot.")
        is_openai_configured = False

# --- 5. API Endpoints ---
@app.get("/", tags=["General"])
def read_root():
    return {"message": "Welcome to the Smart Farm Assistant API!"}

@app.post("/api/predict", tags=["Services"])
async def predict_crop(request: CropPredictionRequest):
    if not all([model, scaler]):
        raise HTTPException(status_code=503, detail="Model or scaler not loaded.")
    
    features = np.array([[request.N, request.P, request.K, request.temperature, request.humidity, request.ph, request.rainfall]])
    # CRITICAL FIX: Scale the user's input before prediction
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return {"recommended_crop": prediction[0]}

@app.get("/api/weather", tags=["Services"])
async def get_weather(q: str):
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OpenWeather API key not configured.")
    
    url = f"https://api.openweathermap.org/data/2.5/weather?q={q}&appid={api_key}&units=metric"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Error fetching weather data.")
        return response.json()

@app.get("/api/pesticides", tags=["Services"])
async def get_pesticide(crop: str, pest: str):
    if pesticide_df is None: raise HTTPException(status_code=503, detail="Pesticide dataset not available.")
    result = pesticide_df[(pesticide_df['Crop'].str.lower() == crop.lower()) & (pesticide_df['Pest'].str.lower() == pest.lower())]
    if not result.empty: return {"pesticide": result['Pesticide'].values[0]}
    raise HTTPException(status_code=404, detail="No recommendation found.")

@app.get("/api/crops", response_model=dict[str, List[str]], tags=["Services"])
async def get_all_crops():
    if pesticide_df is None: raise HTTPException(status_code=503, detail="Pesticide dataset not available.")
    return {"crops": sorted(pesticide_df['Crop'].unique().tolist())}

@app.get("/api/pests", response_model=dict[str, List[str]], tags=["Services"])
async def get_pests_for_crop(crop: str):
    if pesticide_df is None: raise HTTPException(status_code=503, detail="Pesticide dataset not available.")
    pests = pesticide_df[pesticide_df['Crop'].str.lower() == crop.lower()]['Pest'].unique()
    return {"pests": sorted(pests.tolist())}

@app.post("/api/chat", tags=["Services"])
async def chat(request: ChatRequest):
    if is_openai_configured and openai_client:
        try:
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a friendly Smart Farm Assistant for Andhra Pradesh, India. Provide concise, helpful advice on agriculture."},
                    {"role": "user", "content": request.message}
                ]
            )
            return {"reply": response.choices[0].message.content.strip()}
        except Exception as e:
            print(f"OpenAI call failed, using fallback. Error: {e}")
            return {"reply": get_simple_chatbot_response(request.message)}
    else:
        return {"reply": get_simple_chatbot_response(request.message)}

def get_simple_chatbot_response(message: str) -> str:
    """A simple rule-based chatbot for when the OpenAI API is not available."""
    message = message.lower()
    if "weather" in message:
        return "The AI chatbot is currently unavailable. Please use the Weather widget for forecasts."
    elif "crop" in message:
        return "The AI chatbot is currently unavailable. Please use the Crop Recommendation widget."
    elif "pesticide" in message:
        return "The AI chatbot is currently unavailable. Please use the Pesticide Recommendation widget."
    else:
        return "The advanced AI assistant is temporarily unavailable. Please try asking about 'weather', 'crop', or 'pesticides'."

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)

