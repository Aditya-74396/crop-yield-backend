from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

MODEL_PATH = "../models/crop_yield_model.pkl"

# Try loading model
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    model = None
    print("Model not found yet. Train it first!", e)


@app.route('/')
def home():
    return jsonify({"message": "Crop Yield Prediction API is running."})


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded."}), 500

    data = request.get_json()
    features = np.array([[ 
        data.get("average_rainfall", 0),
        data.get("pesticides_tonnes", 0),
        data.get("avg_temp", 0),
        
    ]])
    prediction = model.predict(features)[0]
    return jsonify({"predicted_yield": float(prediction)})


if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(debug=True)
