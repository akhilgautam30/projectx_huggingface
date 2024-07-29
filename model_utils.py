# model_utils.py

import os
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# Define the personality trait labels
traits = ['cAGR', 'cCON', 'cEXT', 'cOPN', 'cNEU']

def load_model_and_weights():
    model_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(traits),
        problem_type="multi_label_classification"
    )

    # Load custom weights
    weights_path = os.path.join(os.getcwd(), 'weights-roberta-base.h5')
    if os.path.exists(weights_path):
        try:
            model.load_weights(weights_path)
            print("Custom weights loaded successfully.")
        except Exception as e:
            print(f"Error loading weights: {str(e)}")
            print("Using default weights.")
    else:
        print(f"Warning: Custom weights file not found at {weights_path}")
        print("Using default weights.")

    return tokenizer, model

# Load the model and tokenizer
tokenizer, model = load_model_and_weights()

def predict_personality(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=512)
    outputs = model(inputs)
    probabilities = tf.nn.sigmoid(outputs.logits)[0]  # Using sigmoid for multi-label
    predictions = [{"trait": trait, "score": float(prob)} for trait, prob in zip(traits, probabilities)]
    return predictions
