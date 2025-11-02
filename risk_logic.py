def classify_band(risk_pct):
    if risk_pct < 20:
        return 'LOW'
    if risk_pct < 50:
        return 'MODERATE'
    if risk_pct < 75:
        return 'HIGH'
    return 'CRITICAL'

def recommendations(risk_pct, chest_pain, bp, chol):
    band = classify_band(risk_pct)
    recs = []
    if band == 'LOW':
        recs.append('Maintain healthy weight, brisk walk 30 min/day')
        recs.append('Annual lipid profile and blood pressure check')
        recs.append('Avoid smoking, manage cholesterol early')
    elif band == 'MODERATE':
        recs.append('Resting ECG and fasting lipid profile soon')
        recs.append('HbA1c screening if diabetic tendency or high fasting sugar')
        recs.append('Treadmill stress test if chest discomfort on exertion')
    elif band == 'HIGH':
        recs.append('Consult a cardiologist within days')
        recs.append('Resting + stress ECG (TMT) and echocardiography')
        recs.append('Troponin test if any active chest pain')
    else:
        recs.append('Seek urgent cardiac evaluation now')
        recs.append('Do ECG immediately and cardiac enzymes (Troponin)')
        recs.append('If chest pain is severe/radiating to left arm or jaw, go to emergency services')
    if bp > 140:
        recs.append('Blood pressure is elevated, discuss antihypertensive management')
    if chol > 240:
        recs.append('Cholesterol is high, ask for full lipid panel and lifestyle modification')
    if chest_pain in [0, 1, 2]:
        recs.append('Describe chest pain pattern to a cardiologist, especially during exertion')
    return band, recs
