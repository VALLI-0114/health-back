# ml/model_loader.py
import joblib
import os

def load_model(base_path):
    model = joblib.load(os.path.join(base_path, "ml/models/disease_model.pkl"))
    encoder = joblib.load(os.path.join(base_path, "ml/models/label_encoder.pkl"))
    return model, encoder
