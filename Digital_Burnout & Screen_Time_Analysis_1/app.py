from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

xgb_model = joblib.load("saved_models/xgb_model.pkl")
rf_model  = joblib.load("saved_models/rf_model.pkl")
scaler    = joblib.load("saved_models/scaler.pkl")

LABEL_MAP = {0: 'Low', 1: 'Moderate', 2: 'High', 3: 'Extreme'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    features = np.array([[
        float(data.get('age',          25)),
        float(data.get('total_app',     4.0)),
        float(data.get('screen_time',   7.0)),
        int(  data.get('num_apps',      15)),
        float(data.get('social_media',  2.5)),
        float(data.get('productivity',  2.0)),
        float(data.get('gaming',        2.0)),
        int(  data.get('gender_enc',    0)),
        int(  data.get('location_enc',  0))
    ]])

    features_scaled = scaler.transform(features)

    xgb_proba = xgb_model.predict_proba(features_scaled)[0].tolist()
    rf_proba  = rf_model.predict_proba(features_scaled)[0].tolist()

    xgb_class = int(np.argmax(xgb_proba))
    rf_class  = int(np.argmax(rf_proba))

    return jsonify({
        'xgb_label': LABEL_MAP[xgb_class],
        'rf_label':  LABEL_MAP[rf_class],
        'xgb_proba': xgb_proba,
        'rf_proba':  rf_proba
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)