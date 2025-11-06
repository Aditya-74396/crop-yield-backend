#  Crop Yield Prediction - Backend

This is a Flask-based backend API for predicting crop yield using a trained Linear Regression model.

---

## Project Overview
The API takes agricultural input parameters like rainfall, pesticide usage, temperature, and farm area, and predicts the expected crop yield.
Framework: Flask

Machine Learning: Scikit-learn (Linear Regression)

Data: data/crop_yield.csv

Model saved at: models/crop_yield_model.pkl

## Folder Structure
crop-yield-backend/
│
├── data/
│   └── crop_yield.csv          # Dataset
├── models/
│   └── crop_yield_model.pkl    # Trained ML model
├── src/
│   └── app.py                  # Flask API
├── train_model.py              # Script to train the model
├── requirements.txt            # Python dependencies
└─  README.md

##  Setup & Run Backend

1. Clone the repository.
 ```bash
   git clone https://github.com/Aditya-74396/crop-yield-backend.git
   cd crop-yield-backend
```
2. Create and activate a virtual environment
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate

3. Open terminal inside the folder and run:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
