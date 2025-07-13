# src/predict.py

import pandas as pd
import joblib
import argparse

# Load model, scaler, and selected features
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Define the 16 selected features (same order as training)
selected_features = [
    'concave points_worst', 'radius_worst', 'texture_worst',
    'concave points_mean', 'compactness_mean', 'area_worst',
    'perimeter_worst', 'concavity_mean', 'symmetry_worst',
    'fractal_dimension_mean', 'compactness_se', 'radius_se',
    'texture_se', 'area_se', 'smoothness_worst', 'symmetry_mean'
]

def predict(input_file):
    data = pd.read_csv(input_file)
    
    # Ensure only the selected features are used
    X = data[selected_features]

    # Apply scaling
    X_scaled = scaler.transform(X)

    # Predict
    predictions = model.predict(X_scaled)

    for i, pred in enumerate(predictions):
        label = "Malignant" if pred == 1 else "Benign"
        print(f"Sample {i+1}: {label}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file")
    args = parser.parse_args()

    predict(args.input)
