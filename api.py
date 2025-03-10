from fastapi import FastAPI
import pickle
import numpy as np
import pandas as pd

app = FastAPI()

# Load trained model
with open("predictive_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load saved feature names
with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

@app.post("/predict/")
def predict(data: dict):
    try:
        # Convert input to DataFrame
        features = pd.DataFrame([data["features"]], columns=["Type", "Air temperature [K]",
                                                              "Process temperature [K]", "Rotational speed [rpm]",
                                                              "Torque [Nm]", "Tool wear [min]"])
        # Ensure One-Hot Encoding consistency
        features = pd.get_dummies(features)

        # Add missing columns (fill with 0)
        for col in feature_columns:
            if col not in features:
                features[col] = 0

        # Reorder columns to match training
        features = features[feature_columns]

        # Make prediction
        prediction = model.predict(features)[0]
        return {"predicted_label": int(prediction)}

    except Exception as e:
        return {"error": str(e)}
