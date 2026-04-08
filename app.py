import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = load_model("models/sentiment_model.keras")

with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

st.title("🎬 Sentiment Analysis System")
st.write("This project predicts whether a review is Positive or Negative.")

review = st.text_area("Enter Review")

if st.button("Predict"):

    review_seq = tokenizer.texts_to_sequences([review])
    review_pad = pad_sequences(review_seq, maxlen=200)

    prediction = model.predict(review_pad)

    if prediction[0][0] >= 0.5:
        st.success(f"Positive Review ({prediction[0][0] * 100:.2f}%)")
    else:
        st.error(f"Negative Review ({(1 - prediction[0][0]) * 100:.2f}%)")