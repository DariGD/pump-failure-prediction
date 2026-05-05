import logging

import requests


logger = logging.getLogger(__name__)


class PumpAPIClient:
    def __init__(self, api_url: str = None):
        import os

        self.api_url = api_url or os.getenv("API_URL", "http://ml_service:8000")
        self.timeout = 10

    def predict(self, sensor_data: dict) -> dict:
        try:
            response = requests.post(
                f"{self.api_url}/predict", json=sensor_data, timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Ошибка API: {e}")
            return {"error": True, "probability": 0, "status": "Ошибка"}

    def health_check(self) -> bool:
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
