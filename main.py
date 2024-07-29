# main.py

import sys
import os
from fastapi import FastAPI
from model_utils import load_model_and_weights, single_predict

app = FastAPI()

# Load the model and tokenizer
output_folder = '.'  # Adjust this path as needed
hugging_model = 'roberta-base'
model = load_model_and_weights(hugging_model, output_folder)

# Root path handler for unit test
@app.get("/")
async def root():
    test_text = ("always a problem. My hair is really wet and I should go dry it, but this assignment is what I need to do now. "
                 "I almost slept through my eight o clock class, but I somehow made it. Ok this show keeps getting cheezier and cheezier "
                 "oh dear. I have to cash a check and deposit it so my check book balances, which is something that needs to be done and "
                 "really quickly because I will have to pay extra for all the hot checks I have written- uh oh. My twenty minutes probably "
                 "seems shorter because I am a slower typist than most people. PROPNAME is a psycho whore, I hate hate her. Something shocking "
                 "happens on this show every 0 seconds. I don't think that Days of our lives is a good show, but I seem to be addicted to it "
                 "anyway. PROPNAME is so nice and her and LOCNAME are finally together, but probably not for long because there is")
    predictions = single_predict(model, test_text)
    return {"predictions": predictions}

@app.get("/predict")
async def predict_personality_get(text: str):
    predictions = single_predict(model, text)
    return {"predictions": predictions}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)




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