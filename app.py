import configparser
import datetime
import os
import re
import time
import cv2
import numpy as np
import pytesseract
import requests
import json
from io import BytesIO
from dotenv import load_dotenv
from pdf2image import convert_from_bytes
from pytesseract import Output
from logger import logger

# ----------------------- Загрузка переменных окружения -----------------------
load_dotenv()

# ----------------------- Загрузка конфигурации -----------------------
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.ini")
with open(CONFIG_PATH, encoding="utf-8") as f:
    config_content = os.path.expandvars(f.read())

config = configparser.ConfigParser(interpolation=None)
config.read_string(config_content)

# Параметры API
API_BASE_URL = config.get("API", "base_url", fallback="http://127.0.0.1:8000")
API_AUTH_ENDPOINT = config.get("API", "auth_endpoint", fallback="/core/api/Auth/authenticate")
API_OBJECTS_ENDPOINT = config.get("API", "objects_endpoint", fallback="/core/api/objects/select")
API_DOWNLOAD_ENDPOINT = config.get("API", "download_endpoint", fallback="/core/api/download/{blobId}/{id}")
API_AUTH_USERNAME = config.get("API", "username", fallback="user")
API_AUTH_PASSWORD = config.get("API", "password", fallback="password")

# Параметры обработки
SAVE_IMAGES = config.getboolean("Processing", "save_images")
DPI = config.getint("Processing", "dpi")
USE_ADVANCED_RECOGNITION = config.getboolean("Processing", "use_advanced_recognition", fallback=False)
RESULTS_DIR = config.get("Paths", "results_dir")
POLLING_INTERVAL = config.getint("API", "polling_interval", fallback=60)

# Ключевые слова для отбора технических требований
SELECTION_KEYWORDS = [kw.strip() for kw in config.get("Selection", "keywords", fallback="").split(",") if kw.strip()]

# Ключевые слова для исключения блоков (подписи, штампы)
EXCLUDE_KEYWORDS = [kw.strip() for kw in config.get("Exclusion", "exclude_keywords", fallback="").split(",") if kw.strip()]

# Создаем необходимые папки
os.makedirs(RESULTS_DIR, exist_ok=True)


# ----------------------- API КЛИЕНТ -----------------------
class APIClient:
    def __init__(self, base_url, username, password):
        self.base_url = base_url
        self.username = username
        self.password = password
        self.token = None
        self.session = requests.Session()

    def authenticate(self):
        """Получение токена авторизации"""
        try:
            auth_url = f"{self.base_url}{API_AUTH_ENDPOINT}"
            payload = {
                        "loginName": self.username,
                        "password": self.password,
                        "passwordType": "plainText",
                        "roleID": 0,
                        "accessLevelID": 0
                    }
            
            logger.info(f"Аутентификация на {auth_url}")
            response = self.session.post(auth_url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            self.token = data.get("accessToken")
            
            if self.token:
                self.session.headers.update({"Authorization": f"Bearer {self.token}"})
                logger.info("Аутентификация успешна")
                return True
            else:
                logger.error("Токен не найден в ответе")
                return False
                
        except Exception as e:
            logger.error(f"Ошибка аутентификации: {e}")
            return False

    def get_objects(self):
        """Получение списка объектов"""
        try:
            objects_url = f"{self.base_url}{API_OBJECTS_ENDPOINT}"
            payload = {
                        "objectTypeId": -1,
                        "attributeIdsToSelect": [-2],
                        "conditions": [
                            {
                                "attributeId": 30357, #признак по которому идёт отбор
                                "relationalOperator": "Equal",
                                "logicalOperator": "none",
                                "groupID": 0,
                                "value": "false",     #если true значит технические требования записаны
                                "content": "text"
                            }
                        ]
                    }

            logger.info(f"Запрос списка объектов: {objects_url}")
            
            response = self.session.post(objects_url, json=payload)
            response.raise_for_status()
            
            objects = response.json()
            logger.info(f"Получено объектов: {len(objects) if isinstance(objects, list) else 'неизвестно'}")
            return objects
            
        except Exception as e:
            logger.error(f"Ошибка получения списка объектов: {e}")
            return None

    def download_file(self, blob_id, object_id):
        """Скачивание файла по blobId и id"""
        try:
            download_url = f"{self.base_url}{API_DOWNLOAD_ENDPOINT.format(blobId=blob_id, id=object_id)}"
            logger.info(f"Скачивание файла: {download_url}")
            
            response = self.session.get(download_url)
            response.raise_for_status()
            
            # Проверяем, что это PDF
            content_type = response.headers.get('content-type', '')
            if 'application/pdf' in content_type or response.content[:4] == b'%PDF':
                logger.info(f"Файл успешно скачан, размер: {len(response.content)} байт")
                return response.content
            else:
                logger.warning(f"Скачанный файл не похож на PDF: {content_type}")
                return None
                
        except Exception as e:
            logger.error(f"Ошибка скачивания файла: {e}")
            return None

    def get_pdf_from_test_api(self):
        """Метод для тестового API (для обратной совместимости)"""
        try:
            test_url = f"{self.base_url}/get-pdf"
            response = self.session.get(test_url)
            response.raise_for_status()
            
            logger.info(f"Тестовый PDF скачан, размер: {len(response.content)} байт")
            return response.content
            
        except Exception as e:
            logger.error(f"Ошибка скачивания тестового PDF: {e}")
            return None


if __name__ == "__main__":
    client = APIClient(API_BASE_URL, API_AUTH_USERNAME, API_AUTH_PASSWORD)
    client.authenticate()
    client.get_objects()