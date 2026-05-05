import boto3
import pandas as pd
import io
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, 
                 endpoint_url: str,
                 bucket_name: str,
                 access_key: str,
                 secret_key: str,
                 redis_client):
        
        self.bucket_name = bucket_name
        self.redis_client = redis_client
        self.cache_key = 'pump_dataframe'
        self.cache_ttl = 3600
        self.s3_client = boto3.client(
            service_name='s3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )
        logger.info(f"S3 инициализирован: {endpoint_url}")
    
    def load_data(self, force_reload: bool = False):
        if not force_reload:
            cached = self.redis_client.get(self.cache_key)
            if cached:
                try:
                    df = pd.read_json(io.StringIO(cached.decode('utf-8')))
                    logger.info(f"Данные загружены из Redis кэша: {len(df)} записей")
                    return df
                except Exception as e:
                    logger.warning(f"Ошибка чтения кэша: {e}")
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key='pump_data.csv'
            )
            data = response['Body'].read()
            df = pd.read_csv(io.BytesIO(data))
            logger.info(f"Данные загружены из S3: {len(df)} записей")
            
            try:
                self.redis_client.setex(
                    self.cache_key,
                    self.cache_ttl,
                    df.to_json()
                )
                logger.info(f"Данные сохранены в Redis кэш")
            except Exception as e:
                logger.warning(f"Не удалось сохранить кэш: {e}")
            
            return df
            
        except Exception as e:
            logger.error(f"Ошибка загрузки из S3: {e}")
            return None
    
    def get_data_info(self):
        df = self.load_data()
        if df is not None:
            return {
                'rows': len(df),
                'columns': len(df.columns),
                'failures': int(df['Failure_Flag'].sum()) if 'Failure_Flag' in df.columns else None
            }
        return {'error': 'Данные не загружены'}