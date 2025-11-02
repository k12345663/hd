from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
from inference import predict_risk
from risk_logic import classify_band, recommendations

app = Flask(__name__)

metrics_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'leaderboard_metrics.csv')
metrics_df = pd.read_csv(metrics_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    age = float(request.form['age'])
    sex = int(request.form['sex'])
    cp = int(request.form['cp'])
    trestbps = float(request.form['trestbps'])
    chol = float(request.form['chol'])
    fbs = int(request.form['fbs'])
    restecg = int(request.form['restecg'])
    thalach = float(request.form['thalach'])
    exang = int(request.form['exang'])
    oldpeak = float(request.form['oldpeak'])
    slope = int(request.form['slope'])
    ca = int(request.form['ca'])
    thal = int(request.form['thal'])

    patient = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }

    risk_pct = predict_risk(patient)
    band, recs = recommendations(risk_pct, cp, trestbps, chol)

    return render_template(
        'result.html',
        risk_pct=round(risk_pct,2),
        band=band,
        recs=recs
    )

@app.route('/leaderboard')
def leaderboard():
    top_models = metrics_df.sort_values(by=['recall','roc_auc'], ascending=[False,False]).head(10)
    return render_template('leaderboard.html', models=top_models.to_dict('records'))

@app.route('/leaderboard_full')
def leaderboard_full():
    all_models = metrics_df.sort_values(by=['recall','roc_auc'], ascending=[False,False])
    return render_template('leaderboard.html', models=all_models.to_dict('records'), full=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
