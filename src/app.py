from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'crop_yield_model.pkl')
MODEL_PATH = os.path.abspath(MODEL_PATH)

# Load model
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print("Model not found. Train it first:", e)
    model = None


@app.route('/')
def home():
    return jsonify({"message": "Crop Yield Prediction API running."})


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()

    try:
        # Extract values from frontend
        rainfall = float(data.get("average_rain_fall_mm_per_year") or data.get("rainfall") or 0)
        pesticides = float(data.get("pesticides_tonnes") or data.get("nitrogen") or 0)
        temperature = float(data.get("avg_temp") or data.get("temperature") or 0)

        # Prepare input for model
        features = np.array([[rainfall, pesticides, temperature]])

        # Predict in hectograms per hectare (original unit)
        prediction_hg_ha = model.predict(features)[0]

        # Convert to kg/ha and then quintals/ha
        prediction_kg_ha = prediction_hg_ha * 0.1
        prediction_quintal_ha = prediction_kg_ha / 100

        response = {
            "predicted_yield": round(float(prediction_quintal_ha), 2),
            "unit": "quintals/ha",
            "crop": data.get("cropType") or data.get("crop") or "Unknown",
            "location": f"{data.get('district', 'Unknown')}, {data.get('state', 'Unknown')}",
            "season": data.get("season", "Unknown")
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400


if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(debug=True)
