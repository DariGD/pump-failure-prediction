import json

import joblib
import pandas as pd
import xgboost as xgb
from config import (
    FEATURE_COLS,
    FEATURES_PATH,
    MODEL_CONFIG,
    MODEL_PARAMS_PATH,
    MODEL_PATH,
    SCALER_PATH,
)
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler


try:
    import io
    import os

    import boto3
    from dotenv import load_dotenv

    load_dotenv()

    s3_client = boto3.client(
        service_name="s3",
        endpoint_url=os.getenv("S3_ENDPOINT", "https://storage.yandexcloud.net"),
        aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
    )
    response = s3_client.get_object(
        Bucket=os.getenv("S3_BUCKET", "pump-data-bucket"), Key="dataset_pump.csv"
    )
    data = response["Body"].read()
    df = pd.read_csv(io.BytesIO(data))
    print(f"Данные загружены из S3: {len(df)} записей")
except Exception as e:
    print(f"S3 не доступен ({e}), загрузка из локального файла")
    df = pd.read_csv("data/dataset_pump.csv")
    print(f"Данные загружены из локального файла: {len(df)} записей")


df["hours_norm"] = df["Operational_Hours"] / df["Operational_Hours"].max()
df["temp_pressure"] = df["Temperature"] / (df["Pressure"] + 1e-10)
df["vibration_rpm"] = df["Vibration"] / (df["RPM"] + 1e-10)
df["flow_temp"] = df["Flow_Rate"] / (df["Temperature"] + 1e-10)

df["Pump_ID_original"] = df["Pump_ID"]


def split_by_pump_chronologically(df, feature_cols, test_size=0.2):
    X_train_list, X_test_list = [], []
    y_train_list, y_test_list = [], []

    for pump_id in df["Pump_ID_original"].unique():
        pump_data = df[df["Pump_ID_original"] == pump_id].sort_values("Operational_Hours")

        failure_positions = pump_data[pump_data["Failure_Flag"] == 1].index
        n_failures = len(failure_positions)

        if n_failures >= 2:
            n_test_failures = max(1, int(n_failures * 0.3))
            split_pos = pump_data.index.get_loc(failure_positions[-n_test_failures])
            split_idx = split_pos
        else:
            split_idx = int(len(pump_data) * (1 - test_size))

        split_idx = max(split_idx, 200)
        if len(pump_data) - split_idx < 24:
            split_idx = int(len(pump_data) * (1 - test_size))

        train_data = pump_data.iloc[:split_idx]
        test_data = pump_data.iloc[split_idx:]

        X_train_list.append(train_data[feature_cols])
        X_test_list.append(test_data[feature_cols])
        y_train_list.append(train_data["Failure_Flag"])
        y_test_list.append(test_data["Failure_Flag"])

        print(
            f"Насос {pump_id}: train отказов={train_data['Failure_Flag'].sum()}, "
            f"test отказов={test_data['Failure_Flag'].sum()}"
        )

    X_train = pd.concat(X_train_list, ignore_index=True)
    X_test = pd.concat(X_test_list, ignore_index=True)
    y_train = pd.concat(y_train_list, ignore_index=True)
    y_test = pd.concat(y_test_list, ignore_index=True)

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = split_by_pump_chronologically(df, FEATURE_COLS)

print(f"\nTrain: {X_train.shape}, отказов: {y_train.sum()}")
print(f"Test: {X_test.shape}, отказов: {y_test.sum()}")


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\nscale_pos_weight = {scale_pos_weight:.2f}")

best_xgb = xgb.XGBClassifier(
    n_estimators=MODEL_CONFIG["n_estimators"],
    max_depth=MODEL_CONFIG["max_depth"],
    learning_rate=MODEL_CONFIG["learning_rate"],
    scale_pos_weight=scale_pos_weight,
    subsample=MODEL_CONFIG["subsample"],
    colsample_bytree=MODEL_CONFIG["colsample_bytree"],
    random_state=MODEL_CONFIG["random_state"],
    use_label_encoder=MODEL_CONFIG["use_label_encoder"],
    eval_metric=MODEL_CONFIG["eval_metric"],
)

best_xgb.fit(X_train_scaled, y_train)

joblib.dump(best_xgb, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

with open(FEATURES_PATH, "w") as f:
    json.dump(FEATURE_COLS, f)

model_params = {
    "n_estimators": MODEL_CONFIG["n_estimators"],
    "max_depth": MODEL_CONFIG["max_depth"],
    "learning_rate": MODEL_CONFIG["learning_rate"],
    "scale_pos_weight": float(scale_pos_weight),
    "subsample": MODEL_CONFIG["subsample"],
    "colsample_bytree": MODEL_CONFIG["colsample_bytree"],
    "random_state": MODEL_CONFIG["random_state"],
}
with open(MODEL_PARAMS_PATH, "w") as f:
    json.dump(model_params, f)

print("\n Сохранение в папку models/")
print(
    f"   PR-AUC на тесте: {average_precision_score(y_test, best_xgb.predict_proba(X_test_scaled)[:, 1]):.4f}"
)
