import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.utils import pad_sequences
import json
import numpy as np

# --- Load model and tokenizer ---
@st.cache_resource
def load_model_and_tokenizer():
    model = tf.keras.models.load_model("saved_model/final_model.keras")
    with open("saved_model/tokenizer.json", "r", encoding="utf-8") as f:
        tokenizer = tokenizer_from_json(f.read())
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# --- App UI ---
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review below to check whether it's **Positive** or **Negative**.")

user_input = st.text_area("Write your movie review here...")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review first.")
    else:
        seq = tokenizer.texts_to_sequences([user_input])
        pad = pad_sequences(seq, maxlen=200, padding="post", truncating="post")
        pred = model.predict(pad)
        sentiment = "😊 Positive" if pred[0][0] > 0.5 else "😞 Negative"
        st.subheader(sentiment)
        st.write(f"Confidence: {pred[0][0]:.2f}")
