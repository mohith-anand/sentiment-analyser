import streamlit as st
import os
import requests

# ----------------------------
# Use backend API (do NOT load model locally)
# ----------------------------
# Backend URL should be provided via environment variable `API_URL` or Streamlit secrets.
API_URL = os.environ.get("API_URL") or (st.secrets.get("API_URL") if hasattr(st, "secrets") and "API_URL" in st.secrets else None)

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
        if not API_URL:
            st.error("API_URL not configured. Set the API_URL environment variable to your backend endpoint.")
        else:
            try:
                resp = requests.post(f"{API_URL.rstrip('/')}/predict", json={"text": user_input}, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                pred_label = data.get("label", "Unknown")
                confidence = data.get("confidence")

                st.markdown(
                    f"<h2 style='color:{color_map.get(pred_label, 'black')}; text-align:center'>{pred_label} {emoji_map.get(pred_label, '')}</h2>",
                    unsafe_allow_html=True
                )

                if 'history' not in st.session_state:
                    st.session_state.history = []
                st.session_state.history.append((user_input, pred_label))
            except requests.exceptions.RequestException as e:
                st.error(f"Request to backend failed: {e}")

# ----------------------------
# Display session history
# ----------------------------
if 'history' in st.session_state and st.session_state.history:
    st.write("### 🕘 Prediction History")
    for text, label in reversed(st.session_state.history):
        st.write(f"- {text} → **{label}**")
