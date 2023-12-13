#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip install streamlit


# In[ ]:


#pip install imbalanced-learn


# In[6]:


import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import re
import nltk
import joblib
import pandas as pd
from gensim.models import Word2Vec


# In[11]:


##### just testing

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

def load_model_and_tokenizer():
    # Load the model
    model = tf.keras.models.load_model('tf2_model')
    # Load the tokenizer
    tokenizer = joblib.load('saved_tokenizer.joblib')
    # Load pre-trained Word2Vec model
    word2vec_model = Word2Vec.load("word2vec.model")

def classify_message(message):
    message = input("Enter the text you want to classify: ")
    # Preprocess the input message
    cleaned_message = functions.clean_text(message)
    print("Cleaned message:", cleaned_message)
    tokenized_message = functions.tokenizing_and_stopwords(cleaned_message)
    print("Tokenized message:", tokenized_message)
    lemmatized_message = functions.lemmatization_and_stopwords(tokenized_message)
    print("Lemmatized message:", lemmatized_message)
    
    # Convert text to sequence
    sequence = tokenizer.texts_to_sequences(lemmatized_message)
    print("Tokenized sequence:", sequence)
    
    # Pad sequence
    padded_sequence = pad_sequences(sequence, maxlen=500) 
    #print("Padded sequence:", padded_sequence)
    
    # Make predictions using the loaded model
    prediction = model.predict(padded_sequence)
    print(prediction)
    
    if prediction[0] > 0.5:
        print("\n--- THIS IS BULLYING!!! --- This message is classified as bullying.")
    else:
        print("\n--- NO BULLYING. --- This message is not classified as bullying.")  
    
    return prediction[0]   


# In[16]:


import streamlit as st

def main():
    # Add custom CSS for the background image
    st.markdown(
        """
        <style>
            body {
                background-image: url('C:/Users/calum/Desktop/Ironhack/Week8/Final Project/streamlit/back.jpg');
                background-size: cover;
                width: 100%;
                height: 100%;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.title("Detect Cyberbullying")

    # User input for message
    message = st.text_area("Enter a message:")

    if st.button("Bullying or not?"):
        
        if message:
            # Classify the message
            result = classify_message(message)

            # Display the result
            if result > 0.5:
                st.error("This is bullying!")
            else:
                st.success("No bullying content.")
        
        else:
            st.warning("Please enter a message to classify.")

    #st.image("cb.png", use_column_width=True)
    
if __name__ == "__main__":
    main()


# In[10]:


#########################################################################################################


# In[ ]:




