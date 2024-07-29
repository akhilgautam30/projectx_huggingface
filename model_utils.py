# model_utils.py

import os
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import numpy as np
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer

# Define the personality trait labels
traits = ['cAGR', 'cCON', 'cEXT', 'cOPN', 'cNEU']

def preprocess(docs):
    stopwrd = set(stopwords.words('english'))
    t = Tokenizer(num_words=20000, filters='!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n')
    t.fit_on_texts(docs)
    encoded_docs = t.texts_to_sequences(docs)
    idx2word = {v: k for k, v in t.word_index.items()}

    def abbreviation_handler(text):
        ln = text.lower()
        ln = ln.replace(r"'t", " not")
        ln = ln.replace(r"'s", " is")
        ln = ln.replace(r"'ll", " will")
        ln = ln.replace(r"'ve", " have")
        ln = ln.replace(r"'re", " are")
        ln = ln.replace(r"'m", " am")
        ln = ln.replace(r"'", " ")
        return ln

    def stopwords_handler(text):
        words = text.split()
        new_words = [w for w in words if w not in stopwrd]
        return ' '.join(new_words)

    def sequence_to_text(listOfSequences):
        tokenized_list = []
        for text in listOfSequences:
            newText = ''
            for num in text:
                newText += idx2word[num] + ' '
            newText = abbreviation_handler(newText)
            newText = stopwords_handler(newText)
            tokenized_list.append(newText)
        return tokenized_list

    newLists = sequence_to_text(encoded_docs)
    return newLists

def tokenize_text(text, hugging_model='roberta-base'):
    clean_text = preprocess(text)
    tokenizer = AutoTokenizer.from_pretrained(hugging_model)
    inputs = tokenizer(clean_text, padding=True, truncation=True, return_tensors='tf')
    x = dict(inputs)
    return x

def single_predict(model, text, traits=['cAGR', 'cCON', 'cEXT', 'cOPN', 'cNEU']):
    traits_scores = dict()
    predicted_labels = dict()
    x = tokenize_text([text])
    logits = model.predict(x, verbose=0).logits
    probs = tf.math.sigmoid(logits).numpy()
    predictions = np.where(probs > 0.5, 1, 0)
    for t, s in zip(traits, probs[0]):
        traits_scores[t] = s
    for t, l in zip(traits, predictions[0]):
        predicted_labels[t] = l
    final_dic = {'probability': traits_scores, 'predicted_label': predicted_labels}
    return final_dic

def load_model_and_weights(hugging_model='roberta-base', output_folder='.'):
    model = TFAutoModelForSequenceClassification.from_pretrained(
        hugging_model, num_labels=len(traits), problem_type="multi_label_classification"
    )
    if len(hugging_model.split('/')) > 1:
        _hugging_model = hugging_model.split('/')[1]
    else:
        _hugging_model = hugging_model.split('/')[0]

    weights_path = os.path.join(output_folder, f'weights-{_hugging_model}.h5')
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
    return model
