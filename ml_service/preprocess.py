import json
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger(__name__)


def load_and_prepare_data(file_path: str, test_size: float = 0.2):
    df = pd.read_csv(file_path)
    logger.info(f"Загружено {len(df)} строк, отказов {df['Failure_Flag'].sum()}")

    data = df.copy()
    data["Pump_ID_original"] = data["Pump_ID"]

    data["hours_norm"] = data["Operational_Hours"] / data["Operational_Hours"].max()
    data["temp_pressure"] = data["Temperature"] / (data["Pressure"] + 1e-10)
    data["vibration_rpm"] = data["Vibration"] / (data["RPM"] + 1e-10)
    data["flow_temp"] = data["Flow_Rate"] / (data["Temperature"] + 1e-10)

    data = pd.get_dummies(data, columns=["Pump_ID"], prefix="pump")
    for i in range(1, 6):
        col = f"pump_{i}"
        if col not in data.columns:
            data[col] = 0

    data = data.sort_values(["Pump_ID_original", "Operational_Hours"])

    pump_cols = [c for c in data.columns if c.startswith("pump_")]
    feature_cols = [
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
    ] + pump_cols

    split_idx = int(len(data) * (1 - test_size))
    train_df = data.iloc[:split_idx]
    test_df = data.iloc[split_idx:]

    X_train = train_df[feature_cols]
    y_train = train_df["Failure_Flag"]
    X_test = test_df[feature_cols]
    y_test = test_df["Failure_Flag"]

    logger.info(f"Train: {X_train.shape}, отказов {y_train.sum()}")
    logger.info(f"Test:  {X_test.shape}, отказов {y_test.sum()}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    with open("../models/features.json", "w") as f:
        json.dump(feature_cols, f)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols
