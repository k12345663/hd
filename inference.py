import joblib
import numpy as np
import pandas as pd
import os

model_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'best_model.pkl')
model = joblib.load(model_path)

feature_order = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']

def predict_risk(patient_dict):
    x_df = pd.DataFrame([patient_dict], columns=feature_order)
    prob = model.predict_proba(x_df)[0][1]
    return float(prob) * 100.0
