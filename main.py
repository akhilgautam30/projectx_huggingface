# API/main.py

# main.py (in the root directory)

import sys
import os
from model_utils import predict_personality
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Personality Assessment API is running"}

@app.get("/predict")
async def predict_personality_get(text: str):
    try:
        predictions = predict_personality(text)
        return {"predictions": predictions}
    except NameError:
        return {"error": "predict_personality function not available"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)




""" from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Minej/bert-base-personality")
model = AutoModelForSequenceClassification.from_pretrained("Minej/bert-base-personality")

# Define the personality trait labels
labels = ["Extroversion", "Neuroticism", "Agreeableness", "Conscientiousness", "Openness"]

# Function to predict personality traits
def predict_personality(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)[0]
    probabilities = torch.softmax(outputs, dim=1)
    predictions = [{"trait": label, "score": float(prob)} for label, prob in zip(labels, probabilities[0])]
    return predictions

# Root path handler
@app.get("/")
async def root():
    return {"message": "Personality Assessment API is running"}

@app.get("/predict")
async def predict_personality_get(text: str):
    predictions = predict_personality(text)
    return {"predictions": predictions} """