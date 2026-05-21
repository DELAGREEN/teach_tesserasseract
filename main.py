# main.py
import configparser
import os
import time
import signal
import sys
from dotenv import load_dotenv
from ocr_processor import OCRProcessor
from api_client import APIClient
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
API_BASE_URL = config.get("API", "base_url")
API_AUTH_ENDPOINT = config.get("API", "authenticate_endpoint")
API_GET_OBJECTS_ID = config.get("API", "get_objects_id")
API_GET_BLOB_ID = config.get("API", "get_blob_id")
API_DOWNLOAD_ENDPOINT = config.get("API", "download_endpoint")
API_SEND_CONTENT = config.get("API", "send_content")
API_AUTH_USERNAME = config.get("API", "username")
API_AUTH_PASSWORD = config.get("API", "password")

# Параметры OCR
SAVE_IMAGES = config.getboolean("Processing", "save_images", fallback=False)
DPI = config.getint("Processing", "dpi", fallback=200)
LANG = config.get("Processing", "lang", fallback="rus+eng")
PSM = config.getint("Processing", "psm", fallback=6)
USE_ADVANCED_RECOGNITION = config.getboolean("Processing", "use_advanced_recognition", fallback=False)
RESULTS_DIR = config.get("Paths", "results_dir", fallback="results")

# Ключевые слова
SELECTION_KEYWORDS = [kw.strip() for kw in config.get("Selection", "keywords", fallback="").split(",") if kw.strip()]
EXCLUDE_KEYWORDS = [kw.strip() for kw in config.get("Exclusion", "exclude_keywords", fallback="").split(",") if kw.strip()]

# Интервал опроса
POLLING_INTERVAL = config.getint("API", "polling_interval", fallback=60)

# Максимальное количество файлов за один цикл (можно поставить 0 для без ограничений)
MAX_FILES_PER_CYCLE = config.getint("Processing", "max_files_per_cycle", fallback=0)

os.makedirs(RESULTS_DIR, exist_ok=True)

# Сборка URL
objects_url = f"{API_BASE_URL}{API_GET_OBJECTS_ID}"
authenticate_url_template  = f"{API_BASE_URL}{API_AUTH_ENDPOINT}"
blob_url_template = f"{API_BASE_URL}{API_GET_BLOB_ID}"
download_url_template = f"{API_BASE_URL}{API_DOWNLOAD_ENDPOINT}"
send_url_template = f"{API_BASE_URL}{API_SEND_CONTENT}"

# Инициализация компонентов
ocr = OCRProcessor(
    dpi=DPI,
    lang=LANG,
    psm=PSM,
    save_images=SAVE_IMAGES,
    save_dir=os.path.join(RESULTS_DIR, "ocr_pages"),
    use_advanced=USE_ADVANCED_RECOGNITION
)

client = APIClient(
    authenticate_url=API_AUTH_ENDPOINT,
    username=API_AUTH_USERNAME,
    password=API_AUTH_PASSWORD,
    ocr_processor=ocr
)

objects_payload = {
    "objectTypeId": -1,
    "attributeIdsToSelect": [-2],
    "conditions": [
        {
            "attributeId": 30357,
            "relationalOperator": "Equal",
            "logicalOperator": "none",
            "groupID": 0,
            "value": "false",
            "content": "text"
        }
    ]
}

# ----------------------- Основной цикл -----------------------
if __name__ == "__main__":
    logger.info("Сервис запущен")
    # Первичная аутентификация
    if not client.authenticate(authenticate_url_template):
        logger.error("Не удалось авторизоваться при старте, выход")
        sys.exit(1)

    def graceful_shutdown(signum, frame):
        logger.info("Получен сигнал завершения, выход")
        sys.exit(0)

    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)

    while True:
        try:
            logger.info("--- Новый цикл опроса ---")
            # Получаем список необработанных объектов
            ids = client.get_objects_id(objects_url, objects_payload)
            if not ids:
                logger.info("Нет объектов для обработки")
            else:
                # Получаем blobId нужных файлов
                client.get_blob_id(blob_url_template, extensions=[".dwg.pdf"])
                # Обрабатываем (можно ограничить количество за цикл)
                max_files = MAX_FILES_PER_CYCLE if MAX_FILES_PER_CYCLE > 0 else None
                client.process_all_files(
                    download_url_template=download_url_template,
                    send_url_template=send_url_template,
                    selection_keywords=SELECTION_KEYWORDS,
                    exclude_keywords=EXCLUDE_KEYWORDS,
                    max_files=max_files
                )
        except KeyboardInterrupt:
            logger.info("Ручная остановка сервиса")
            break
        except Exception as e:
            logger.exception(f"Критическая ошибка в цикле: {e}")
            # Ждем немного перед следующей попыткой, чтобы не заспамить логи
            time.sleep(10)
            continue

        logger.info(f"Ожидание {POLLING_INTERVAL} секунд до следующего опроса...")
        time.sleep(POLLING_INTERVAL)