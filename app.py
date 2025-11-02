import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load 
import sys
from typing import List

# --- Pydantic Input Model ---
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# --- FastAPI App Setup ---
app = FastAPI(title="IRIS Prediction API (Docker/K8s)", version="1.0")
MODEL_PATH = "artifacts/model.joblib" # The model will be placed here by the Dockerfile
model = None

def load_model():
    """Loads the model from the local file system inside the container."""
    global model
    try:
        # Check if the file exists (it should, as the Dockerfile downloads it)
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
            
        model = load(MODEL_PATH)
        print("Model loaded successfully from local file!")
    except Exception as e:
        print(f"ERROR: Failed to load model. Details: {e}")
        model = None

# Load model on application startup
load_model()

# --- Health Check Endpoint ---
@app.get("/health", status_code=200, tags=["Health"])
def health():
    if model is not None:
        return {"status": "healthy", "model_status": "loaded"}
    else:
        raise HTTPException(status_code=503, detail="Model load failure")

# --- Prediction Endpoint ---
@app.post("/predict", tags=["Prediction"])
def predict(features: IrisFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        input_data = [
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]
        input_array = np.array([input_data])
        
        prediction = model.predict(input_array)
        predicted_class = prediction[0] 

        return {
            "prediction": predicted_class,
            "model_source": "DVC GCS Remote"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")