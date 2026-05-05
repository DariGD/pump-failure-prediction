MODEL_CONFIG = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.07,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "use_label_encoder": False,
    "eval_metric": "logloss",
}


TRAIN_CONFIG = {"test_size": 0.2, "random_state": 42}

SMOTE_CONFIG = {"sampling_strategy": 0.1, "random_state": 42}

FEATURE_COLS = [
    "Temperature",
    "Vibration",
    "Pressure",
    "Flow_Rate",
    "RPM",
    "Operational_Hours",
    "hours_norm",
    "temp_pressure",
    "vibration_rpm",
    "flow_temp",
]


MODEL_PATH = "models/xgboost_best.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURES_PATH = "models/features.json"
MODEL_PARAMS_PATH = "models/model_params.json"


API_HOST = "0.0.0.0"
API_PORT = 8000
