import pickle
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

def load_model(model_path):
    """
    Load the trained model with error handling
    """
    try:
        if not os.path.exists(model_path):
            st.warning(f"Model file not found: {model_path}")
            st.info("Using demo model for testing...")
            return create_dummy_model()
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded successfully from {model_path}")
        return model
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Using demo model for testing...")
        return create_dummy_model()

def create_dummy_model():
    """
    Create a dummy model for testing when the real model is not available
    """
    try:
        # Create a simple random forest model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Create dummy training data (this is just for testing)
        X_dummy = np.random.rand(100, 40)  # 100 samples, 40 features (MFCC)
        y_dummy = np.random.choice(['male', 'female'], 100)
        
        # Train the dummy model
        model.fit(X_dummy, y_dummy)
        
        print("Demo model created and trained")
        return model
        
    except Exception as e:
        print(f"Error creating dummy model: {e}")
        return None

def save_model(model, model_path):
    """
    Save the trained model
    """
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

def predict_gender_simple(features):
    """
    Simple gender prediction based on audio features
    Uses basic heuristics when model is not available
    """
    try:
        # Simple heuristic: use spectral centroid and MFCC values
        # This is a very basic approach for demo purposes
        if len(features) >= 5:
            # Use first few MFCC coefficients
            avg_mfcc = np.mean(features[:5])
            
            # Simple threshold-based prediction
            if avg_mfcc > 0:
                return 'female'
            else:
                return 'male'
        else:
            # Random prediction if features are insufficient
            return np.random.choice(['male', 'female'])
    except:
        return np.random.choice(['male', 'female'])
