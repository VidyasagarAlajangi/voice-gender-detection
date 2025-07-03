import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np

def load_model(model_path):
    """
    Load the trained model with error handling
    """
    try:
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            print("Creating a dummy model for testing...")
            return create_dummy_model()
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded successfully from {model_path}")
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Creating a dummy model for testing...")
        return create_dummy_model()

def create_dummy_model():
    """
    Create a dummy model for testing when the real model is not available
    """
    # Create a simple random forest model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    # Create dummy training data (this is just for testing)
    X_dummy = np.random.rand(100, 40)  # 100 samples, 40 features (MFCC)
    y_dummy = np.random.choice(['male', 'female'], 100)
    
    # Train the dummy model
    model.fit(X_dummy, y_dummy)
    
    print("Dummy model created and trained")
    return model

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
