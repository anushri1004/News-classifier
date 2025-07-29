# app.py
import streamlit as st
import pickle
import json
import re
import string
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    tfidf, model = pickle.load(f)

# Load categories
with open("categories.json", "r") as f:
    categories = json.load(f)

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Streamlit UI
st.title("📰 News Topic Classifier")
st.write("Enter a news article or paragraph below and classify its topic:")

text_input = st.text_area("Paste news content here...")

if st.button("Classify"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(text_input)
        vect_text = tfidf.transform([cleaned])
        prediction = model.predict(vect_text)[0]
        st.success(f"📢 Predicted Topic: **{categories[prediction]}**")
