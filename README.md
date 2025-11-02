# Cardio Risk AI - Heart Disease Risk Prediction System

A clinical decision-support web application that predicts heart disease risk percentage using machine learning models trained on the UCI Cleveland Heart Disease dataset.

## Features

- **Multi-Model Prediction**: 10 trained ML models including Logistic Regression, Random Forest, Gradient Boosting, XGBoost, CatBoost, SVM, KNN, Neural Networks, and more
- **Risk Assessment**: Probability-based risk prediction (0-100%) with color-coded severity bands
- **Clinical Recommendations**: Cardiologist-style guidance based on risk level
  - Low Risk (<20%): Lifestyle recommendations
  - Moderate Risk (20-50%): Preventive screening tests
  - High Risk (50-75%): Urgent cardiology consultation
  - Critical Risk (>75%): Emergency evaluation
- **Model Leaderboard**: Compare all models by recall, AUROC, accuracy, and other metrics
- **Medical-Grade UI**: Professional interface with clear disclaimers

## Best Model Performance

**SVC (Support Vector Classifier)** - Deployed Model
- Recall (Sensitivity): 82.14%
- AUROC: 92.41%
- Accuracy: 83.33%

## Dataset

UCI Cleveland Heart Disease Dataset (297 samples after cleaning)

**13 Clinical Features:**
- Age, Sex, Chest Pain Type
- Resting Blood Pressure, Serum Cholesterol
- Fasting Blood Sugar, Resting ECG Results
- Maximum Heart Rate, Exercise-Induced Angina
- ST Depression (Oldpeak), ST Segment Slope
- Number of Major Vessels (Fluoroscopy)
- Thalassemia Type

## Project Structure

```
.
├── app/
│   ├── app.py              # Flask application (routes: /, /result, /leaderboard)
│   ├── inference.py        # Model loading and prediction
│   ├── risk_logic.py       # Risk classification and recommendations
│   ├── templates/          # HTML templates
│   │   ├── index.html      # Input form
│   │   ├── result.html     # Risk results and recommendations
│   │   └── leaderboard.html # Model performance comparison
│   └── static/
│       └── styles.css      # Medical-grade UI styling
├── data/
│   ├── heart.csv           # Cleaned dataset
│   ├── best_model.pkl      # Deployed model (SVC)
│   ├── pipeline.pkl        # Preprocessing pipeline
│   └── leaderboard_metrics.csv # All model metrics
├── training/
│   ├── download_data.py    # Dataset download and cleaning
│   └── train_minimal.py    # Training script (10 models)
└── README.md
```

## Running the Application

The Flask app is already running on port 5000. Access it through the Replit webview.

### Training Models (Optional)

To retrain the models:

```bash
python training/download_data.py
python training/train_minimal.py
```

## Technical Stack

- **Backend**: Flask
- **ML Libraries**: scikit-learn, XGBoost, CatBoost, imbalanced-learn
- **Data Processing**: pandas, numpy
- **Model Persistence**: joblib

## Medical Disclaimer

This tool uses machine learning models trained on historical data for **screening purposes only**. It is **not a diagnostic tool** and should **never replace professional medical evaluation**.

If you experience severe chest pain, shortness of breath, or other cardiac symptoms, seek emergency medical attention immediately.

## Model Details

All models use:
- StandardScaler for numerical features
- OneHotEncoder for categorical features
- Stratified train-test split
- Cross-validated predictions
- Calibrated probability outputs

**Training Metrics Tracked:**
- AUROC (Area Under ROC Curve)
- Recall/Sensitivity
- Precision
- Accuracy
- F1 Score
- Brier Score (calibration quality)

## Implementation Notes

- Originally designed for 34 models, reduced to 10 for resource efficiency
- LightGBM skipped due to system library constraints
- BernoulliNB and ComplementNB excluded (incompatible with StandardScaler)
- Models ranked primarily by recall to minimize false negatives (critical in medical screening)
