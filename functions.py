# Cyberbullying Functions
# functions.py

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import functions
import re
import nltk
import joblib
import pandas as pd
from gensim.models import Word2Vec

nltk.download('wordnet')
nltk.download('punkt')


# Cenverting to lower_case, removing any special characters/symbols and removing extra spaces. 
def clean_text(text):   
    text = text.lower() # Convert to lowercase
    text = re.sub(r'\W', ' ', text) # Remove special characters and symbols
    text = re.sub(r'\s+', ' ', text, flags=re.MULTILINE) # Remove extra spaces
    words = [word for word in text.split() if word.isalnum() and word.isalpha()] # Split the text into words and keep only alphanumeric #
    cleaned_text = ' '.join(words) # Join the words with a space
    return cleaned_text


# Tokenizing and removing stopwords
def tokenizing_and_stopwords(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    text = " ".join(filtered_tokens)
    return text


# lemmatization aims to reduce words to their base or root form.
def lemmatization_and_stopwords(text):
    if not text:
        return ""
    tokens = word_tokenize(text)
    clean_text = []
    lemmatizer = nltk.WordNetLemmatizer()
    for token in tokens:
        if token.lower() not in stopwords.words('english') and len(token) > 3:
            token = lemmatizer.lemmatize(token)
            clean_text.append(token)
    result_text = " ".join(clean_text)
    return result_text