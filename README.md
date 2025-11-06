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
data/ → Contains the dataset (crop_yield.csv) used to train the model.

models/ → Stores the trained model (crop_yield_model.pkl).

src/ → Contains the backend code (app.py) for the API.

train_model.py → Script to train the crop yield prediction model.

requirements.txt → Lists all Python dependencies.

README.md → Project description, instructions, and API usage.

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

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   
4.  Train the model (optional if the model is already provided in models/)
  ```bash
  python train_model.py
```

5. Run the Flask server
   ```bash
   python src/app.py
   
6. API will be running at
```cpp
https://127.0.0.1:5000/


