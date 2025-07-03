import streamlit as st
import numpy as np
import soundfile as sf
import os
import tempfile
from extract_features import extract_features, extract_enhanced_features
from utils import load_model
import librosa

st.set_page_config(page_title="Voice Gender Detector", layout="centered")
st.title("🔊 Voice Gender Detection")
st.write("Upload a short voice clip (WAV format) to predict the speaker's gender.")

# Add debug mode toggle
debug_mode = st.checkbox("Enable Debug Mode", value=False)

uploaded_file = st.file_uploader("Upload Audio File (.wav)", type=["wav"])

if uploaded_file is not None:
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_audio_path = tmp_file.name
    
    if debug_mode:
        st.write(f"Temporary file path: {temp_audio_path}")
        st.write(f"File size: {os.path.getsize(temp_audio_path)} bytes")
    
    # Display audio player
    st.audio(uploaded_file, format='audio/wav')
    
    # Add feature extraction method selection
    feature_method = st.selectbox(
        "Select Feature Extraction Method:",
        ["Basic MFCC", "Enhanced Features"]
    )
    
    if st.button("Analyze Audio"):
        with st.spinner("Extracting features..."):
            if feature_method == "Basic MFCC":
                features = extract_features(temp_audio_path)
            else:
                features = extract_enhanced_features(temp_audio_path)
            
            if debug_mode:
                if features is not None:
                    st.write(f"Features extracted successfully!")
                    st.write(f"Feature vector shape: {features.shape}")
                    st.write(f"Feature vector preview: {features[:10]}...")
                else:
                    st.write("Failed to extract features")
        
        if features is not None:
            with st.spinner("Loading model and making prediction..."):
                try:
                    model = load_model("gender_model.pkl")
                    
                    if debug_mode:
                        st.write("Model loaded successfully")
                    
                    # Make prediction
                    prediction = model.predict([features])[0]
                    
                    # Get prediction probability if available
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba([features])[0]
                        max_proba = np.max(proba)
                        
                        st.success(f"🎯 Predicted Gender: **{prediction.upper()}**")
                        st.info(f"Confidence: {max_proba:.2%}")
                        
                        if debug_mode:
                            # Show probability distribution
                            classes = model.classes_ if hasattr(model, 'classes_') else ['female', 'male']
                            prob_dict = dict(zip(classes, proba))
                            st.write("Probability distribution:")
                            for gender, prob in prob_dict.items():
                                st.write(f"  {gender}: {prob:.2%}")
                    else:
                        st.success(f"🎯 Predicted Gender: **{prediction.upper()}**")
                        
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
                    if debug_mode:
                        import traceback
                        st.text(traceback.format_exc())
        else:
            st.error("Failed to extract features from audio file. Try a different file.")
            if debug_mode:
                st.write("Common issues:")
                st.write("- Audio file is too short (< 0.1 seconds)")
                st.write("- Audio file is corrupted")
                st.write("- Unsupported audio format")
                st.write("- Missing audio processing libraries")
    
    # Cleanup temporary file
    try:
        os.unlink(temp_audio_path)
    except:
        pass

# Add information section
with st.expander("ℹ️ How it works"):
    st.write("""
    This voice gender detection system uses machine learning to analyze audio features:
    
    **Basic MFCC Method:**
    - Extracts 40 Mel-frequency cepstral coefficients (MFCCs)
    - MFCCs capture the spectral characteristics of speech
    - Averages the features across time
    
    **Enhanced Features Method:**
    - MFCC coefficients (mean and standard deviation)
    - Chroma features (pitch class profiles)
    - Spectral centroid (brightness of sound)
    - Zero crossing rate (noisiness indicator)
    - Spectral rolloff (spectral shape)
    
    **Tips for best results:**
    - Use clear, uncompressed WAV files
    - Ensure audio is at least 1-2 seconds long
    - Avoid background noise
    - Use consistent volume levels
    """)

with st.expander("🔧 Troubleshooting"):
    st.write("""
    If you encounter issues:
    
    1. **"Failed to extract features"**: 
       - Check if audio file is valid and not corrupted
       - Ensure file is long enough (> 0.1 seconds)
       - Try a different audio file
    
    2. **"Model not found"**:
       - The system will create a dummy model for testing
       - Train your own model using the provided training code
    
    3. **Enable Debug Mode** to see detailed information about:
       - Feature extraction process
       - Model loading status
       - Prediction confidence scores
    """)
