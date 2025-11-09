import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os

DATA_PATH = "data/crop_yield.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "crop_yield_model.pkl")

# Load dataset
df = pd.read_csv(DATA_PATH)
print(f" Dataset loaded successfully with {len(df)} rows and {len(df.columns)} columns.")
print("Columns:", df.columns.tolist())

# Features and target
X = df[["average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]]
y = df["hg/ha_yield"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")

# Save model
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f" Model saved at: {MODEL_PATH}")
