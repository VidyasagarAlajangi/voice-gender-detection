import joblib

def load_model(model_path='gender_model.pkl'):
    return joblib.load(model_path)
 
