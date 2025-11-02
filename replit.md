# Cardio Risk AI - Project Memory

## Project Overview
Heart disease risk prediction web application with ML-based screening and cardiologist-style recommendations.

## Recent Implementation (November 2, 2025)
Built complete end-to-end application:
- Trained 10 ML models on UCI Cleveland Heart Disease dataset (297 samples)
- Best model: SVC (82% recall, 92% AUROC)
- Flask app with 3 routes: form, results, leaderboard
- Medical-grade UI with color-coded risk bands

## Architecture Decisions

### Data Pipeline
- UCI Cleveland dataset with 13 clinical features
- StandardScaler + OneHotEncoder preprocessing
- Missing value handling for ca/thal columns

### Model Training Strategy
- Reduced from planned 34 models to 10 for resource constraints
- Train-test split (80/20) instead of k-fold CV for speed
- Models: LogReg, DT, RF, GB, XGB, CatBoost, KNN, SVC, NB, MLP
- Ranking: Recall > AUROC (prioritize sensitivity in medical screening)

### Technical Constraints Resolved
- LightGBM: Skipped due to libgomp.so.1 library issue in Nix environment
- BernoulliNB/ComplementNB: Removed (require non-negative features, incompatible with StandardScaler)
- Path handling: Used os.path.join for relative paths from app/ directory

## Project Structure
```
app/          # Flask application
  app.py      # Main routes
  inference.py # Model loading
  risk_logic.py # Clinical recommendations
  templates/  # HTML (index, result, leaderboard)
  static/     # CSS styling
data/         # Dataset and trained models
training/     # Data download and training scripts
```

## User Preferences
- No comments in code (as explicitly requested)
- Clinical focus: screening tool, not diagnostic
- Professional medical-grade UI design

## Key Features
1. Risk percentage prediction with traffic-light color bands
2. Cardiologist-style recommendations based on risk levels
3. Model performance leaderboard
4. Medical disclaimers throughout

## Running State
- Flask workflow configured on port 5000 (webview output)
- Application running successfully with no errors
- Ready for user testing and deployment
