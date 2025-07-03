import streamlit as st
import numpy as np
import soundfile as sf
import os
import tempfile
from extract_features import extract_features
from utils import load_model

st.set_page_config(page_title="Voice Gender Detector", layout="centered")

st.title("🔊 Voice Gender Detection")
st.write("Upload a short voice clip (WAV format) to predict the speaker's gender.")

uploaded_file = st.file_uploader("Upload Audio File (.wav)", type=["wav"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_audio_path = tmp_file.name

    st.audio(uploaded_file, format='audio/wav')

    features = extract_features(temp_audio_path)
    if features is not None:
        model = load_model("gender_model.pkl")
        prediction = model.predict([features])[0]
        st.success(f"🎯 Predicted Gender: **{prediction.upper()}**")
    else:
        st.error("Failed to extract features from audio file. Try a different file.")
 
