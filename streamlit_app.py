import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

st.title("Sentiment Analyzer")

user_input = st.text_area("Enter text to analyze")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        vect_text = vectorizer.transform([user_input])
        pred_num = model.predict(vect_text)[0]
        pred_label = label_map.get(pred_num, "Unknown")
        st.success(f"Predicted Sentiment: {pred_label}")
