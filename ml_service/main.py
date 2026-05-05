import logging
from contextlib import asynccontextmanager
import uvicorn
from database import cache_manager, db_manager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from predict import PumpFailurePredictor
from pydantic import BaseModel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor

    predictor = PumpFailurePredictor(
        model_path="models/xgboost_best.pkl",
        scaler_path="models/scaler.pkl",
        features_path="models/features.json",
    )
    logger.info("Модель загружена")

    await db_manager.connect()
    await cache_manager.connect()
    logger.info("PostgreSQL и Redis подключены")

    yield

    await db_manager.close()
    await cache_manager.close()
    logger.info("Соединения закрыты")


app = FastAPI(
    title="Pump Failure Prediction API",
    description="API для прогнозирования отказов насосного оборудования",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SensorData(BaseModel):
    pump_id: int
    temperature: float
    vibration: float
    pressure: float
    flow_rate: float
    rpm: float
    operational_hours: float = 0


@app.get("/health")
async def health():
    return {"status": "healthy" if predictor else "degraded", "model_loaded": predictor is not None}


@app.post("/predict")
async def predict(data: SensorData):
    if not predictor:
        raise HTTPException(503, "Модель не загружена")

    cached = await cache_manager.get_prediction(data.pump_id)
    if cached:
        logger.info(f"Кэш: предсказание для насоса {data.pump_id}")
        return cached

    input_data = {
        "Temperature": data.temperature,
        "Vibration": data.vibration,
        "Pressure": data.pressure,
        "Flow_Rate": data.flow_rate,
        "RPM": data.rpm,
        "Operational_Hours": data.operational_hours,
        "Pump_ID": data.pump_id,
    }

    result = predictor.predict(input_data)

    try:
        await db_manager.save_prediction(data.pump_id, result)
        logger.info(f"Предсказание для насоса {data.pump_id} сохранено в БД")
    except Exception as e:
        logger.error(f"Ошибка сохранения в БД: {e}")

    await cache_manager.set_prediction(data.pump_id, result)

    return result


@app.get("/history/{pump_id}")
async def get_history(pump_id: int, limit: int = 100):
    """История предсказаний для насоса"""
    try:
        history = await db_manager.get_predictions_history(pump_id, limit)
        return {"pump_id": pump_id, "history": history}
    except Exception as e:
        logger.error(f"Ошибка получения истории: {e}")
        raise HTTPException(500, "Ошибка получения истории")


@app.post("/audit")
async def audit_log(user_id: str, action: str, details: dict = None, ip: str = None):
    try:
        await db_manager.log_audit(user_id, "operator", action, details, ip)
        return {"status": "logged"}
    except Exception as e:
        logger.error(f"Ошибка логирования: {e}")
        raise HTTPException(500, "Ошибка логирования")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
