import json
import logging

import joblib
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class PumpFailurePredictor:
    def __init__(self, model_path: str, scaler_path: str, features_path: str):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        with open(features_path) as f:
            self.feature_cols = json.load(f)

        self.threshold = 0.5
        logger.info(f"Модель загружена, признаков: {len(self.feature_cols)}")

    def _prepare_features(self, data: dict) -> np.ndarray:
        df = pd.DataFrame([data])

        df["hours_norm"] = df["Operational_Hours"] / 8760
        df["temp_pressure"] = df["Temperature"] / (df["Pressure"] + 1e-10)
        df["vibration_rpm"] = df["Vibration"] / (df["RPM"] + 1e-10)
        df["flow_temp"] = df["Flow_Rate"] / (df["Temperature"] + 1e-10)
        X = df[self.feature_cols].values
        X_scaled = self.scaler.transform(X)

        return X_scaled

    def predict(self, data: dict) -> dict:
        features = self._prepare_features(data)
        proba = float(self.model.predict_proba(features)[0, 1])
        pred_class = 1 if proba > self.threshold else 0

        if proba < 0.4:
            risk = "LOW"
            status = "Норма"
        elif proba < 0.7:
            risk = "MEDIUM"
            status = "Внимание"
        else:
            risk = "HIGH"
            status = "Опасно"

        return {
            "probability": round(proba, 4),
            "class": pred_class,
            "risk_level": risk,
            "status": status,
        }
