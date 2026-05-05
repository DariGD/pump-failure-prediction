import logging
from typing import Any

import uvicorn
from database import cache_manager, db_manager
from fastapi import FastAPI, HTTPException, Request
from predict import PumpFailurePredictor
from pydantic import BaseModel, Field


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Pump Failure Prediction API",
    description="Прогнозирование отказов насосного оборудования",
    version="1.0.0",
)

predictor = None

MODEL_VERSION = "v1"


class SensorData(BaseModel):
    pump_id: int = Field(..., ge=1, le=5, description="ID насоса (1-5)")
    temperature: float = Field(..., description="Температура (°C)")
    vibration: float = Field(..., description="Вибрация (мм/с)")
    pressure: float = Field(..., description="Давление (бар)")
    flow_rate: float = Field(..., description="Расход (м³/ч)")
    rpm: float = Field(..., description="Обороты (об/мин)")
    operational_hours: float | None = Field(0, description="Часы работы")

    def to_dict(self) -> dict[str, Any]:
        return {
            "Temperature": self.temperature,
            "Vibration": self.vibration,
            "Pressure": self.pressure,
            "Flow_Rate": self.flow_rate,
            "RPM": self.rpm,
            "Operational_Hours": self.operational_hours,
            "Pump_ID": self.pump_id,
        }


class PredictionResponse(BaseModel):
    probability: float = Field(..., description="Вероятность отказа (0-1)")
    class_: int = Field(..., alias="class", description="0-норма, 1-отказ")
    risk_level: str = Field(..., description="LOW/MEDIUM/HIGH")
    status: str = Field(..., description="Норма/Внимание/Опасно")


class AuditLogData(BaseModel):
    user_id: str = Field(..., description="ID пользователя")
    user_role: str = Field(..., description="Роль: operator/technologist/admin")
    action: str = Field(..., description="Действие")
    details: dict | None = Field(None, description="Детали действия")
    ip_address: str | None = Field(None, description="IP-адрес")


@app.on_event("startup")
async def startup_event():
    global predictor
    try:
        predictor = PumpFailurePredictor(
            model_path="../models/xgboost_smote_v1.pkl",
            scaler_path="../models/scaler.pkl",
            features_path="../models/features.json",
        )
        logger.info("Модель загружена")
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        predictor = None

    try:
        await db_manager.connect()
        logger.info("PostgreSQL подключена")
    except Exception as e:
        logger.error(f"Ошибка подключения к PostgreSQL: {e}")

    try:
        await cache_manager.connect()
        logger.info("Redis подключён")
    except Exception as e:
        logger.error(f"Ошибка подключения к Redis: {e}")

    logger.info("API сервис запущен")


@app.on_event("shutdown")
async def shutdown_event():
    await db_manager.close()
    await cache_manager.close()
    logger.info("API сервис остановлен")


@app.get("/health")
async def health_check():
    return {"status": "healthy" if predictor else "degraded", "model_loaded": predictor is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict(data: SensorData, request: Request):
    if not predictor:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    cached = await cache_manager.get_prediction(data.pump_id)
    if cached:
        logger.info(f"Предсказание для насоса {data.pump_id}")
        return cached
    try:
        result = predictor.predict(data.to_dict())

        try:
            await db_manager.save_prediction(
                pump_id=data.pump_id, result=result, model_version=MODEL_VERSION
            )
        except Exception as e:
            logger.error(f"Ошибка сохранения в БД: {e}")

        await cache_manager.set_prediction(data.pump_id, result)

        logger.info(f"Предсказание для насоса {data.pump_id}: {result['probability']}")
        return result

    except Exception as e:
        logger.error(f"Ошибка предсказания: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{pump_id}")
async def get_history(pump_id: int, limit: int = 100):
    try:
        history = await db_manager.get_predictions_history(pump_id, limit)
        return {"pump_id": pump_id, "total": len(history), "history": history}
    except Exception as e:
        logger.error(f"Ошибка получения истории: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Для установки порога классификации и получения его у текущей модели
@app.post("/threshold")
async def set_threshold(threshold: float):
    if not predictor:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    try:
        predictor.set_threshold(threshold)
        logger.info(f"Порог установлен: {threshold}")
        return {"success": True, "threshold": threshold}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/threshold")
async def get_threshold():
    if not predictor:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    return {"threshold": predictor.threshold}


@app.post("/audit")
async def audit_log(data: AuditLogData):
    try:
        await db_manager.log_audit(
            user_id=data.user_id,
            user_role=data.user_role,
            action=data.action,
            details=data.details,
            ip=data.ip_address,
        )
        return {"status": "logged"}
    except Exception as e:
        logger.error(f"Ошибка логирования: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
