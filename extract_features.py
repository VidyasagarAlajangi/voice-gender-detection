# extract_features.py
import numpy as np
import librosa
import warnings
warnings.filterwarnings('ignore')

def extract_features(audio_path):
    """
    Extract MFCC features from audio file with enhanced error handling
    """
    try:
        print(f"Loading audio file: {audio_path}")
        
        # Load audio file
        audio, sample_rate = librosa.load(audio_path, res_type='kaiser_fast')
        print(f"Audio loaded successfully. Duration: {len(audio)/sample_rate:.2f}s, Sample rate: {sample_rate}")
        
        # Check if audio is too short
        if len(audio) < sample_rate * 0.1:  # Less than 0.1 second
            print("Warning: Audio file is very short")
            return None
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        print(f"MFCC shape: {mfccs.shape}")
        
        # Scale features
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        print(f"Scaled MFCC shape: {mfccs_scaled.shape}")
        
        return mfccs_scaled
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_enhanced_features(audio_path):
    """
    Extract multiple audio features for better gender detection
    """
    try:
        # Load audio
        audio, sample_rate = librosa.load(audio_path, res_type='kaiser_fast')
        
        # Extract various features
        features = []
        
        # MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        features.extend(np.mean(mfccs.T, axis=0))
        features.extend(np.std(mfccs.T, axis=0))
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        features.extend(np.mean(chroma.T, axis=0))
        features.extend(np.std(chroma.T, axis=0))
        
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
        features.extend(np.mean(spectral_centroid.T, axis=0))
        features.extend(np.std(spectral_centroid.T, axis=0))
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        features.extend(np.mean(zcr.T, axis=0))
        features.extend(np.std(zcr.T, axis=0))
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
        features.extend(np.mean(spectral_rolloff.T, axis=0))
        features.extend(np.std(spectral_rolloff.T, axis=0))
        
        return np.array(features)
        
    except Exception as e:
        print(f"Error extracting enhanced features: {e}")
        return None
