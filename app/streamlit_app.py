import streamlit as st
import requests

API_URL = "http://localhost:8000/predict"

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰"
)

st.title("📰 Fake News Detection")

st.write(
    "Paste a news article below and classify it as REAL or FAKE."
)

article = st.text_area(
    "Article Text",
    height=250
)

if st.button("Classify"):

    if not article.strip():
        st.warning("Please enter some text.")
    else:

        payload = {
            "text": article
        }

        response = requests.post(
            API_URL,
            json=payload
        )

        result = response.json()

        label = result["label"]
        confidence = result["confidence"]

        if label == "REAL":
            st.success(f"Prediction: {label}")
        else:
            st.error(f"Prediction: {label}")

        st.progress(float(confidence))

        st.write(
            f"Confidence: {confidence:.2%}"
        )