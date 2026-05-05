import json
import logging
import os
from typing import Any
import asyncpg
import redis.asyncio as redis
from data_loader import DataLoader
from dotenv import load_dotenv


load_dotenv()

logger = logging.getLogger(__name__)


class DatabaseManager:

    def __init__(self):
        self.pool = None
        self._max_retries = 5
        self._retry_delay = 2

    async def init_data_loader(self, redis_client):
        data_loader = DataLoader(
            endpoint_url=os.getenv("S3_ENDPOINT", "https://storage.yandexcloud.net"),
            bucket_name=os.getenv("S3_BUCKET", "pump-data-bucket"),
            access_key=os.getenv("S3_ACCESS_KEY", ""),
            secret_key=os.getenv("S3_SECRET_KEY", ""),
            redis_client=redis_client,
        )

        df = data_loader.load_data()
        if df is not None:
            logger.info(f"Данные загружены: {len(df)} записей")
            return data_loader, df
        else:
            logger.error("Не удалось загрузить данные")
            return None, None

    async def connect(self, retry: bool = True):
        import asyncio

        host = os.getenv("DB_HOST", "postgres")
        port = os.getenv("DB_PORT", "5432")
        database = os.getenv("DB_NAME", "pump_monitoring")
        user = os.getenv("DB_USER", "admin")
        password = os.getenv("DB_PASSWORD", "your_password")

        for attempt in range(self._max_retries):
            try:
                self.pool = await asyncpg.create_pool(
                    host=host,
                    port=port,
                    database=database,
                    user=user,
                    password=password,
                    min_size=1,
                    max_size=10,
                    command_timeout=60,
                )
                logger.info(f"Подключение к PostgreSQL установлено ({host}:{port}/{database})")

                await self._init_tables()
                return

            except Exception as e:
                logger.warning(
                    f"Попытка {attempt + 1}/{self._max_retries} подключения к БД провалена: {e}"
                )
                if retry and attempt < self._max_retries - 1:
                    await asyncio.sleep(self._retry_delay)
                else:
                    logger.error("Не удалось подключиться к PostgreSQL")
                    raise

    async def _init_tables(self):
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    pump_id INTEGER NOT NULL,
                    temperature REAL,
                    vibration REAL,
                    pressure REAL,
                    flow_rate REAL,
                    rpm REAL,
                    operational_hours REAL,
                    probability REAL NOT NULL,
                    prediction_class INTEGER NOT NULL,
                    risk_level VARCHAR(10),
                    status VARCHAR(50),
                    threshold REAL,
                    model_version VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(50),
                    user_role VARCHAR(20),
                    action VARCHAR(100),
                    details JSONB,
                    ip_address VARCHAR(45),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS model_versions (
                    id SERIAL PRIMARY KEY,
                    version VARCHAR(50) UNIQUE NOT NULL,
                    model_path VARCHAR(500),
                    scaler_path VARCHAR(500),
                    features_path VARCHAR(500),
                    pr_auc REAL,
                    recall REAL,
                    is_active BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_predictions_pump_id ON predictions(pump_id);
                CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);
                CREATE INDEX IF NOT EXISTS idx_predictions_risk_level ON predictions(risk_level);
                CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs(created_at);
                CREATE INDEX IF NOT EXISTS idx_model_versions_active ON model_versions(is_active);
            """
            )

        logger.info("Таблицы инициализированы")

    async def save_prediction(
        self,
        pump_id: int,
        result: dict[str, Any],
        sensor_data: dict[str, Any] = None,
        model_version: str = "v1",
    ):
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO predictions (
                    pump_id, temperature, vibration, pressure, flow_rate, rpm, operational_hours,
                    probability, prediction_class, risk_level, status, threshold, model_version
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            """,
                pump_id,
                sensor_data.get("temperature") if sensor_data else None,
                sensor_data.get("vibration") if sensor_data else None,
                sensor_data.get("pressure") if sensor_data else None,
                sensor_data.get("flow_rate") if sensor_data else None,
                sensor_data.get("rpm") if sensor_data else None,
                sensor_data.get("operational_hours") if sensor_data else None,
                result["probability"],
                result["class"],
                result["risk_level"],
                result["status"],
                result.get("threshold", 0.5),
                model_version,
            )
        logger.debug(f"Предсказание для насоса {pump_id} сохранено")

    async def get_predictions_history(
        self, pump_id: int, limit: int = 100, offset: int = 0
    ) -> list[dict]:

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, probability, risk_level, status, threshold, model_version, created_at,
                       temperature, vibration, pressure, flow_rate, rpm
                FROM predictions
                WHERE pump_id = $1
                ORDER BY created_at DESC
                LIMIT $2 OFFSET $3
            """,
                pump_id,
                limit,
                offset,
            )
        return [dict(row) for row in rows]

    async def get_last_prediction(self, pump_id: int) -> dict | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT probability, risk_level, status, created_at
                FROM predictions
                WHERE pump_id = $1
                ORDER BY created_at DESC
                LIMIT 1
            """,
                pump_id,
            )
        return dict(row) if row else None

    async def get_statistics(self, pump_id: int = None) -> dict:
        async with self.pool.acquire() as conn:
            if pump_id:
                row = await conn.fetchrow(
                    """
                    SELECT
                        COUNT(*) as total,
                        AVG(probability) as avg_probability,
                        COUNT(CASE WHEN risk_level = 'HIGH' THEN 1 END) as high_risk_count,
                        COUNT(CASE WHEN risk_level = 'MEDIUM' THEN 1 END) as medium_risk_count,
                        COUNT(CASE WHEN risk_level = 'LOW' THEN 1 END) as low_risk_count
                    FROM predictions
                    WHERE pump_id = $1
                """,
                    pump_id,
                )
            else:
                row = await conn.fetchrow(
                    """
                    SELECT
                        COUNT(*) as total,
                        AVG(probability) as avg_probability,
                        COUNT(CASE WHEN risk_level = 'HIGH' THEN 1 END) as high_risk_count,
                        COUNT(CASE WHEN risk_level = 'MEDIUM' THEN 1 END) as medium_risk_count,
                        COUNT(CASE WHEN risk_level = 'LOW' THEN 1 END) as low_risk_count
                    FROM predictions
                """
                )
        return dict(row) if row else {}

    async def register_model_version(
        self,
        version: str,
        model_path: str,
        scaler_path: str,
        features_path: str,
        pr_auc: float = None,
        recall: float = None,
        is_active: bool = False,
    ):

        async with self.pool.acquire() as conn:
            if is_active:
                await conn.execute("UPDATE model_versions SET is_active = FALSE")

            await conn.execute(
                """
                INSERT INTO model_versions (version, model_path, scaler_path, features_path, pr_auc, recall, is_active)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (version) DO UPDATE SET
                    model_path = EXCLUDED.model_path,
                    scaler_path = EXCLUDED.scaler_path,
                    features_path = EXCLUDED.features_path,
                    pr_auc = EXCLUDED.pr_auc,
                    recall = EXCLUDED.recall,
                    is_active = EXCLUDED.is_active
            """,
                version,
                model_path,
                scaler_path,
                features_path,
                pr_auc,
                recall,
                is_active,
            )
        logger.info(f"Версия модели {version} зарегистрирована (active={is_active})")

    async def get_active_model_version(self) -> dict | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT version, model_path, scaler_path, features_path, pr_auc, recall
                FROM model_versions
                WHERE is_active = TRUE
                LIMIT 1
            """
            )
        return dict(row) if row else None

    async def log_audit(
        self, user_id: str, user_role: str, action: str, details: dict = None, ip: str = None
    ):
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO audit_logs (user_id, user_role, action, details, ip_address)
                VALUES ($1, $2, $3, $4, $5)
            """,
                user_id,
                user_role,
                action,
                json.dumps(details or {}),
                ip,
            )
        logger.info(f"Аудит: {user_id} -> {action}")

    async def get_audit_logs(self, limit: int = 100, user_id: str = None) -> list[dict]:
        async with self.pool.acquire() as conn:
            if user_id:
                rows = await conn.fetch(
                    """
                    SELECT * FROM audit_logs
                    WHERE user_id = $1
                    ORDER BY created_at DESC
                    LIMIT $2
                """,
                    user_id,
                    limit,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT * FROM audit_logs
                    ORDER BY created_at DESC
                    LIMIT $1
                """,
                    limit,
                )
        return [dict(row) for row in rows]

    async def close(self):
        if self.pool:
            await self.pool.close()
            logger.info("Соединение закрыто")


class CacheManager:
    def __init__(self):
        self.redis = None
        self._default_ttl = 3600

    async def connect(self, retry: bool = True):
        import asyncio
        host = os.getenv("REDIS_HOST", "redis")
        port = os.getenv("REDIS_PORT", "6379")

        for attempt in range(5):
            try:
                self.redis = await redis.from_url(f"redis://{host}:{port}", decode_responses=True)
                await self.redis.ping()
                logger.info(f"Подключение к Redis установлено ({host}:{port})")
                return
            except Exception as e:
                logger.warning(f"Попытка {attempt + 1}/5 подключения к Redis failed: {e}")
                if retry and attempt < 4:
                    await asyncio.sleep(2)
                else:
                    logger.error("Не удалось подключиться к Redis")
                    raise

    async def get_prediction(self, pump_id: int) -> dict | None:
        if not self.redis:
            return None
        data = await self.redis.get(f"pump:{pump_id}:prediction")
        return json.loads(data) if data else None

    async def set_prediction(self, pump_id: int, result: dict[str, Any], ttl: int = None):
        if not self.redis:
            return
        ttl = ttl or self._default_ttl
        await self.redis.setex(f"pump:{pump_id}:prediction", ttl, json.dumps(result, default=str))
        logger.debug(f"Кэш обновлён для насоса {pump_id} (TTL={ttl}с)")

    async def get_all_predictions(self) -> dict[int, dict]:
        if not self.redis:
            return {}
        keys = await self.redis.keys("pump:*:prediction")
        result = {}
        for key in keys:
            pump_id = int(key.split(":")[1])
            data = await self.redis.get(key)
            if data:
                result[pump_id] = json.loads(data)
        return result

    async def delete_prediction(self, pump_id: int):
        if self.redis:
            await self.redis.delete(f"pump:{pump_id}:prediction")
            logger.debug(f"Кэш удалён для насоса {pump_id}")

    async def clear_all(self):
        if self.redis:
            keys = await self.redis.keys("pump:*:prediction")
            if keys:
                await self.redis.delete(*keys)
                logger.info(f"Очищено {len(keys)} записей из кэша")

    async def close(self):
        if self.redis:
            await self.redis.close()
            logger.info("Соединение с Redis закрыто")


db_manager = DatabaseManager()
cache_manager = CacheManager()


async def init_connections():
    await db_manager.connect()
    await cache_manager.connect()


async def close_connections():
    await db_manager.close()
    await cache_manager.close()
