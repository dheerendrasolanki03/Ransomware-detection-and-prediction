from flask import Flask, request, jsonify  # type: ignore
import joblib  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore

app = Flask(__name__)
scaler = joblib.load('models/scaler.pkl')
lgb_model = joblib.load('models/lightgbm_model.pkl')
dl_model = load_model('models/dl_model.h5')
future_model = load_model('models/future_lstm.h5')
autoencoder = load_model('models/autoencoder_lstm.h5', compile=False) 

SEQUENCE_LENGTH = 10

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        features = pd.DataFrame([data])

        features_scaled = scaler.transform(features)

        lgb_pred = int(lgb_model.predict(features_scaled)[0])

        dl_pred = float(dl_model.predict(features_scaled)[0][0])

        sequence_input = np.expand_dims(np.tile(features_scaled, (SEQUENCE_LENGTH, 1)), axis=0)
        future_pred = float(future_model.predict(sequence_input)[0][0])

        anomaly_seq = np.expand_dims(np.tile(features_scaled, (SEQUENCE_LENGTH, 1)), axis=0)
        reconstructed = autoencoder.predict(anomaly_seq)
        reconstruction_error = np.mean(np.power(anomaly_seq - reconstructed, 2))
        
        result = {
            "LightGBM_Prediction": lgb_pred,
            "DL_Prediction": round(dl_pred, 4),
            "Future_LSTM_Prediction": round(future_pred, 4),
            "Anomaly_Score": round(reconstruction_error, 4)
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
