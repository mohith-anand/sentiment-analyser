import streamlit as st
import joblib
import os

# ----------------------------
# Load model and vectorizer
# ----------------------------
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, 'model', 'sentiment_model.pkl')
vector_path = os.path.join(BASE_DIR, 'model', 'tfidf_vectorizer.pkl')

try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vector_path)
except FileNotFoundError:
    st.error("Model files not found! Please place 'sentiment_model.pkl' and 'tfidf_vectorizer.pkl' in the 'models/' folder.")
    st.stop()

# ----------------------------
# Mapping numeric prediction to sentiment label
# ----------------------------
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
color_map = {"Negative": "red", "Neutral": "gray", "Positive": "green"}
emoji_map = {"Negative": "😞", "Neutral": "😐", "Positive": "😄"}

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Sentiment Analyzer", page_icon="📝", layout="centered")

# ----------------------------
# Main UI
# ----------------------------
st.title("📝 Sentiment Analyzer")
st.markdown("Enter text below to predict its sentiment:")

user_input = st.text_area(
    "Your text here:",
    height=150,
    placeholder="Type something like 'I love this product!'"
)

# ----------------------------
# Predict button
# ----------------------------
if st.button("Predict Sentiment"):
    if not user_input.strip():
        st.warning("Please enter some text!")
    else:
        vect_text = vectorizer.transform([user_input])
        pred_num = model.predict(vect_text)[0]
        pred_label = label_map.get(pred_num, "Unknown")

        st.markdown(
            f"<h2 style='color:{color_map[pred_label]}; text-align:center'>{pred_label} {emoji_map[pred_label]}</h2>",
            unsafe_allow_html=True
        )

        if 'history' not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append((user_input, pred_label))

# ----------------------------
# Display session history
# ----------------------------
if 'history' in st.session_state and st.session_state.history:
    st.write("### 🕘 Prediction History")
    for text, label in reversed(st.session_state.history):
        st.write(f"- {text} → **{label}**")
