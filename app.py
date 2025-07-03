import streamlit as st
import numpy as np
import tempfile
import os
from extract_features import extract_features
from utils import load_model, predict_gender_simple

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
    
    st.header("Demo Mode")
    st.info("🔧 Currently running in demo mode. Upload your trained model file to get better predictions.")

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
    st.info(f"📁 File size: {file_size / 1024:.1f} KB")
    
    if st.button("🎯 Analyze Voice", type="primary"):
        with st.spinner("🔍 Analyzing audio features..."):
            # Extract features
            features = extract_features(temp_audio_path)
            
            if features is not None:
                st.success("✅ Features extracted successfully!")
                
                # Load model and predict
                with st.spinner("🤖 Making prediction..."):
                    model = load_model("gender_model.pkl")
                    
                    if model is not None:
                        try:
                            prediction = model.predict([features])[0]
                            
                            # Show result with icons
                            if prediction.lower() == 'male':
                                st.success("🚹 **Predicted Gender: MALE**")
                            else:
                                st.success("🚺 **Predicted Gender: FEMALE**")
                            
                            # Show confidence if available
                            if hasattr(model, 'predict_proba'):
                                proba = model.predict_proba([features])[0]
                                confidence = np.max(proba)
                                st.info(f"📊 Confidence: {confidence:.1%}")
                                
                                # Show probability distribution
                                if hasattr(model, 'classes_'):
                                    classes = model.classes_
                                    for i, class_name in enumerate(classes):
                                        st.write(f"  {class_name}: {proba[i]:.1%}")
                                
                        except Exception as e:
                            st.error(f"❌ Prediction error: {e}")
                            # Fallback to simple prediction
                            simple_pred = predict_gender_simple(features)
                            st.warning(f"🔄 Fallback prediction: {simple_pred.upper()}")
                    else:
                        # Use simple prediction method
                        simple_pred = predict_gender_simple(features)
                        st.warning(f"🔄 Demo prediction: {simple_pred.upper()}")
                        st.info("Upload a trained model for better accuracy!")
                        
            else:
                st.error("❌ Failed to extract features from the audio file.")
                st.write("**Possible issues:**")
                st.write("- Audio file is too short (< 0.1 seconds)")
                st.write("- Audio file is corrupted or in wrong format")
                st.write("- Try a different audio file")
    
    # Cleanup temporary file
    try:
        os.unlink(temp_audio_path)
    except:
        pass

# Add example section
st.markdown("---")
st.header("📋 How it works")
with st.expander("Click to learn more"):
    st.write("""
    **Voice Gender Detection Process:**
    
    1. **Audio Loading**: The system loads your WAV file and extracts the audio signal
    2. **Feature Extraction**: Extracts MFCC (Mel-frequency cepstral coefficients) features
    3. **Preprocessing**: Normalizes and scales the extracted features
    4. **Prediction**: Uses a trained machine learning model to classify the gender
    5. **Result**: Shows the predicted gender with confidence score
    
    **Best Practices:**
    - Use clear, high-quality audio recordings
    - Ensure audio is 1-10 seconds long
    - Avoid background noise
    - Use uncompressed WAV format
    """)

# Add tips section
st.header("💡 Tips for Better Results")
col1, col2 = st.columns(2)

with col1:
    st.write("✅ **Good Audio:**")
    st.write("- Clear speech")
    st.write("- No background noise")
    st.write("- 1-10 seconds duration")
    st.write("- WAV format")

with col2:
    st.write("❌ **Avoid:**")
    st.write("- Very short clips (< 1 sec)")
    st.write("- Noisy recordings")
    st.write("- Compressed audio")
    st.write("- Multiple speakers")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Made with ❤️ using Streamlit | 
    <a href='https://github.com/VidyasagarAlajangi/voice-gender-detection' target='_blank'>
        View on GitHub 🔗
    </a>
    </p>
</div>
""", unsafe_allow_html=True)
