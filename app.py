import streamlit as st
import numpy as np
import tempfile
import os
from extract_features import extract_features
from utils import load_model

st.set_page_config(
    page_title="Voice Gender Detection",
    page_icon="🔊",
    layout="centered"
)

st.title("🔊 Voice Gender Detection")
st.markdown("Upload a short voice clip (WAV format) to predict the speaker's gender.")

# Add sidebar with info
with st.sidebar:
    st.header("About")
    st.write("This app uses machine learning to detect gender from voice samples.")
    st.write("**Features used:**")
    st.write("- MFCC (Mel-frequency cepstral coefficients)")
    st.write("- Audio spectral features")
    
    st.header("Instructions")
    st.write("1. Upload a WAV audio file")
    st.write("2. File should be 1-10 seconds long")
    st.write("3. Clear speech works best")

uploaded_file = st.file_uploader(
    "Upload Audio File (.wav)", 
    type=["wav"],
    help="Upload a WAV file (max 200MB)"
)

if uploaded_file is not None:
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_audio_path = tmp_file.name
    
    # Display audio player
    st.audio(uploaded_file, format='audio/wav')
    
    # Show file info
    file_size = len(uploaded_file.getvalue())
    st.info(f"File size: {file_size / 1024:.1f} KB")
    
    if st.button("🎯 Analyze Voice", type="primary"):
        with st.spinner("Analyzing audio features..."):
            # Extract features
            features = extract_features(temp_audio_path)
            
            if features is not None:
                # Load model and predict
                model = load_model("gender_model.pkl")
                if model is not None:
                    prediction = model.predict([features])[0]
                    
                    # Show result
                    if prediction.lower() == 'male':
                        st.success("🚹 **Predicted Gender: MALE**")
                    else:
                        st.success("🚺 **Predicted Gender: FEMALE**")
                    
                    # Show confidence if available
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba([features])[0]
                        confidence = np.max(proba)
                        st.info(f"Confidence: {confidence:.1%}")
                else:
                    st.error("Could not load the model. Using demo mode.")
                    # Demo prediction
                    demo_prediction = np.random.choice(['male', 'female'])
                    st.warning(f"Demo prediction: {demo_prediction.upper()}")
            else:
                st.error("Failed to extract features. Please try a different audio file.")
    
    # Cleanup
    try:
        os.unlink(temp_audio_path)
    except:
        pass

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
