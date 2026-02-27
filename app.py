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
API_AUTH_USERNAME = config.get("API", "username", fallback="")
API_AUTH_PASSWORD = config.get("API", "password", fallback="")

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
                "username": self.username,
                "password": self.password
            }
            
            logger.info(f"Аутентификация на {auth_url}")
            response = self.session.post(auth_url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            self.token = data.get("token") or data.get("access_token")
            
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
            logger.info(f"Запрос списка объектов: {objects_url}")
            
            response = self.session.get(objects_url)
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


# ----------------------- ФУНКЦИИ ОБРАБОТКИ PDF -----------------------
def extract_image_from_pdf_bytes(pdf_bytes: bytes, page_index=0) -> np.array:
    """Извлекает изображение из PDF байтов"""
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=DPI)
        if not pages:
            logger.error("Не удалось конвертировать PDF: страницы не найдены")
            return None
            
        image = np.array(pages[page_index])
        return image
    except Exception as e:
        logger.error(f"Ошибка конвертации PDF: {e}")
        return None


def preprocess_image(image):
    """Улучшенная предобработка изображения"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(gray, -1, sharpen_kernel)
    binary = cv2.adaptiveThreshold(
        sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    return binary


def preprocess_block_image(image):
    """Специальная предобработка для блоков текста"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((1, 1), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return binary


def detect_text_blocks(image, min_area_ratio=0.005):
    """Обнаруживает все текстовые блоки на изображении"""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    binary = preprocess_image(img_rgb)

    custom_config = r"--psm 6"
    data = pytesseract.image_to_data(
        img_rgb, lang="rus", output_type=Output.DICT, config=custom_config
    )

    mask = np.zeros_like(binary)
    for i in range(len(data["level"])):
        if data["text"][i].strip():
            x, y, w, h = (
                data["left"][i],
                data["top"][i],
                data["width"][i],
                data["height"][i],
            )
            mask[y : y + h, x : x + w] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 30))
    dilated = cv2.dilate(mask, kernel, iterations=3)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        logger.info("Не найдено текстовых областей")
        return []

    min_area = image.shape[0] * image.shape[1] * min_area_ratio
    text_blocks = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            text_blocks.append(
                {
                    "bbox": (x, y, w, h),
                    "area": area,
                    "width": w,
                    "height": h,
                    "aspect_ratio": w / h if h > 0 else 0,
                }
            )

    return text_blocks


def sort_blocks_correctly(text_blocks, image_width):
    """Сортирует блоки: сверху вниз, а при одинаковой высоте - справа налево"""
    sorted_blocks = sorted(
        text_blocks, key=lambda block: (block["bbox"][1], -block["bbox"][0])
    )
    return sorted_blocks


# ---------- ФУНКЦИИ ДЛЯ ИНТЕЛЛЕКТУАЛЬНОГО ОТБОРА БЛОКОВ ----------
def score_text_block(block_text, keywords=None):
    """
    Оценивает текстовый блок на предмет того, является ли он техническими требованиями.
    Возвращает числовую оценку (чем больше, тем вероятнее).
    """
    if not block_text or not block_text.strip():
        return 0

    lines = [line.strip() for line in block_text.split('\n') if line.strip()]
    num_lines = len(lines)
    if num_lines == 0:
        return 0

    # 1. Количество маркированных пунктов (строки, начинающиеся с цифры и точки)
    bullet_points = 0
    for line in lines:
        if line and line[0].isdigit() and '.' in line[:5]:
            bullet_points += 1

    # 2. Поиск ключевых слов
    keyword_matches = 0
    if keywords:
        text_lower = block_text.lower()
        for kw in keywords:
            if kw.lower() in text_lower:
                keyword_matches += 1

    # 3. Средняя длина строки
    total_len = sum(len(line) for line in lines)
    avg_line_len = total_len / num_lines if num_lines else 0

    # Весовые коэффициенты (подобраны эмпирически)
    score = (bullet_points * 5) + (keyword_matches * 8) + (avg_line_len * 0.5) + (num_lines * 2)
    return score


def extract_text_fast(block_image):
    """Быстрое извлечение текста из блока для предварительной оценки (без сложной очистки)"""
    gray = cv2.cvtColor(block_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    custom_config = r"--psm 6 -c tessedit_char_whitelist=0123456789абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,:;-()/ "
    text = pytesseract.image_to_string(binary, lang="rus", config=custom_config).strip()
    return text


def contains_exclusion_words(text, keywords):
    """Проверяет, содержит ли текст хотя бы одно из исключающих слов"""
    if not text or not keywords:
        return False
    text_lower = text.lower()
    for kw in keywords:
        if kw.lower() in text_lower:
            return True
    return False


def select_tt_blocks(text_blocks, image, keywords=None, exclude_keywords=None, threshold_ratio=0.5):
    """
    Отбирает блоки, которые вероятно содержат технические требования,
    исключая блоки со словами из exclude_keywords.
    """
    if not text_blocks:
        return []

    blocks_with_scores = []
    for block in text_blocks:
        x, y, w, h = block['bbox']
        block_img = image[y:y+h, x:x+w]
        if block_img.size == 0:
            continue
        text = extract_text_fast(block_img)
        
        # Проверка на исключающие слова
        if exclude_keywords and contains_exclusion_words(text, exclude_keywords):
            logger.info(f"Блок исключён (слова-исключения): {text[:50]}...")
            continue
        
        score = score_text_block(text, keywords)
        blocks_with_scores.append((block, score, text))

    # Сортируем по убыванию оценки
    blocks_with_scores.sort(key=lambda x: x[1], reverse=True)

    # Логируем топ-5 кандидатов для отладки
    for i, (block, score, text_preview) in enumerate(blocks_with_scores[:5]):
        preview = text_preview[:50].replace('\n', ' ')
        logger.info(f"Кандидат {i+1}: оценка {score:.1f}, текст: {preview}...")

    if not blocks_with_scores:
        return []

    max_score = blocks_with_scores[0][1]
    if max_score == 0:
        return []

    threshold = max_score * threshold_ratio
    selected_blocks = [item[0] for item in blocks_with_scores if item[1] >= threshold]

    # Ограничим количество блоков (например, не более 3, чтобы не захватить лишнего)
    if len(selected_blocks) > 3:
        selected_blocks = selected_blocks[:3]

    logger.info(f"Отобрано блоков с ТТ: {len(selected_blocks)} (порог {threshold:.1f})")
    return selected_blocks


def extract_text_from_block_advanced(image, block_bbox):
    """Улучшенное извлечение текста из блока"""
    x, y, w, h = block_bbox
    block_image = image[y : y + h, x : x + w]

    processed_block = preprocess_block_image(block_image)
    
    whitelist = "0123456789абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,:;-()/\\ "
    
    custom_config = f"--psm 6 --oem 3 -c tessedit_char_whitelist={whitelist} -c preserve_interword_spaces=1"
    
    text = pytesseract.image_to_string(
        processed_block, lang="rus", config=custom_config
    ).strip()

    return text


def extract_text_from_block_simple(image, block_bbox):
    """Простое извлечение текста из блока"""
    x, y, w, h = block_bbox
    block_image = image[y : y + h, x : x + w]
    
    blacklist = "$"
    custom_config = f"--psm 6 -c tessedit_char_blacklist={blacklist} -c preserve_interword_spaces=1"
    
    text = pytesseract.image_to_string(
        block_image, lang="rus", config=custom_config
    ).strip()

    return text


def clean_text(text):
    """Очищает и форматирует распознанный текст"""
    if not text:
        return ""

    lines = [line.strip() for line in text.split("\n") if line.strip()]

    cleaned_lines = []
    current_paragraph = []

    for line in lines:
        if line.endswith((".", ":", ";")) or len(line) < 50:
            if current_paragraph:
                current_paragraph.append(line)
                cleaned_lines.append(" ".join(current_paragraph))
                current_paragraph = []
            else:
                cleaned_lines.append(line)
        else:
            current_paragraph.append(line)

    if current_paragraph:
        cleaned_lines.append(" ".join(current_paragraph))

    return "\n".join(cleaned_lines)


# ---------- ОСНОВНАЯ ФУНКЦИЯ РАСПОЗНАВАНИЯ ----------
def convert_pdf_to_text_advanced(image):
    """Улучшенная функция для распознавания текста в блоках с интеллектуальным отбором ТТ"""
    text_blocks = detect_text_blocks(image)

    if not text_blocks:
        logger.info("Не найдено текстовых блоков")
        return None, "", None

    logger.info(f"Всего найдено блоков: {len(text_blocks)}")

    # Используем интеллектуальный отбор вместо поиска по ширине
    main_blocks = select_tt_blocks(text_blocks, image, SELECTION_KEYWORDS, EXCLUDE_KEYWORDS)

    if not main_blocks:
        logger.info("Не найдено блоков с техническими требованиями")
        return None, "", None

    # Правильно сортируем выбранные блоки
    main_blocks_sorted = sort_blocks_correctly(main_blocks, image.shape[1])

    result_image = image.copy()

    all_text_data = {}
    full_text = ""

    logger.info("\n" + "=" * 60)
    logger.info("РАСПОЗНАВАНИЕ ТЕКСТА ПО БЛОКАМ (ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ)")
    logger.info("=" * 60)

    for i, block_info in enumerate(main_blocks_sorted):
        x, y, w, h = block_info["bbox"]

        logger.info(f"\n--- Блок {i+1} ---")
        logger.info(f"Координаты: ({x}, {y}) размер: {w}x{h}")

        if USE_ADVANCED_RECOGNITION:
            block_text = extract_text_from_block_advanced(image, (x, y, w, h))
        else:
            block_text = extract_text_from_block_simple(image, (x, y, w, h))

        cleaned_text = clean_text(block_text)

        if cleaned_text:
            full_text += cleaned_text + "\n\n"

        all_text_data[f"block_{i+1}"] = {
            "bbox": (x, y, w, h),
            "size": (w, h),
            "text": cleaned_text,
            "position": i + 1,
        }

        logger.info(f"Текст блока {i+1}:")
        logger.info(cleaned_text)

        if SAVE_IMAGES:
            # Рисуем синим прямоугольник для отобранного блока
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (255, 0, 0), 3)
            cv2.putText(
                result_image,
                f"TT Block {i+1}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )

    # Опционально: показываем остальные блоки серым для контекста
    if SAVE_IMAGES:
        all_blocks_sorted = sort_blocks_correctly(text_blocks, image.shape[1])
        for block_info in all_blocks_sorted:
            if block_info not in main_blocks_sorted:
                x, y, w, h = block_info["bbox"]
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (128, 128, 128), 1)

    logger.info("\n" + "=" * 60)
    logger.info("ПОЛНЫЙ ТЕКСТ (только отобранные блоки)")
    logger.info("=" * 60)
    logger.info(full_text)

    return all_text_data, full_text, result_image


def process_pdf_bytes(pdf_bytes, source_info=""):
    """
    Обработка PDF из байтов
    """
    logger.info(f"Начало обработки PDF: {source_info}")

    img = extract_image_from_pdf_bytes(pdf_bytes)
    if img is None:
        logger.error("Не удалось извлечь изображение из PDF")
        return None, None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    text_data, full_text, result_image = convert_pdf_to_text_advanced(img_rgb)

    if full_text and text_data:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Сохраняем текстовый файл
        text_filename = os.path.join(RESULTS_DIR, f"document_{timestamp}.txt")
        with open(text_filename, "w", encoding="utf-8") as f:
            f.write(full_text)
        logger.info(f"Сохранен текстовый файл: {text_filename}")

        if SAVE_IMAGES and result_image is not None:
            outlined_filename = os.path.join(
                RESULTS_DIR, f"document_{timestamp}_blocks_outlined.jpg"
            )
            cv2.imwrite(
                outlined_filename, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            )
            logger.info(f"Сохранено изображение с обводкой блоков: {outlined_filename}")

    return text_data, full_text


# ----------------------- ОСНОВНОЙ ЦИКЛ -----------------------
def process_test_api():
    """Обработка тестового API (один PDF)"""
    logger.info("Запуск обработки тестового API")
    
    client = APIClient(API_BASE_URL, API_AUTH_USERNAME, API_AUTH_PASSWORD)
    
    # Для тестового API аутентификация не нужна
    pdf_bytes = client.get_pdf_from_test_api()
    
    if pdf_bytes:
        return process_pdf_bytes(pdf_bytes, "Тестовый API")
    else:
        logger.error("Не удалось получить PDF из тестового API")
        return None, None


def process_real_api():
    """Обработка реального API с полным циклом"""
    logger.info("Запуск обработки реального API")
    
    client = APIClient(API_BASE_URL, API_AUTH_USERNAME, API_AUTH_PASSWORD)
    
    # 1. Аутентификация
    if not client.authenticate():
        logger.error("Не удалось аутентифицироваться")
        return
    
    # 2. Получение списка объектов
    objects = client.get_objects()
    if not objects:
        logger.error("Не удалось получить список объектов")
        return
    
    # 3. Обработка каждого объекта
    if isinstance(objects, list):
        for obj in objects:
            try:
                # Извлекаем blobId и id из объекта
                # Формат может отличаться в зависимости от API
                blob_id = obj.get("blobId") or obj.get("blob_id") or obj.get("fileId")
                object_id = obj.get("id") or obj.get("objectId")
                
                if not blob_id or not object_id:
                    logger.warning(f"Пропуск объекта: не найдены blobId/id {obj}")
                    continue
                
                logger.info(f"Обработка объекта ID: {object_id}, BlobID: {blob_id}")
                
                # 4. Скачивание файла
                pdf_bytes = client.download_file(blob_id, object_id)
                
                if pdf_bytes:
                    # 5. Обработка PDF
                    process_pdf_bytes(pdf_bytes, f"Объект {object_id}")
                else:
                    logger.warning(f"Не удалось скачать PDF для объекта {object_id}")
                    
            except Exception as e:
                logger.error(f"Ошибка обработки объекта: {e}")
    else:
        logger.warning(f"Неожиданный формат списка объектов: {type(objects)}")


def main():
    """Основная функция"""
    import sys
    
    # Определяем режим работы по аргументу командной строки
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    
    if mode == "test":
        # Тестовый режим
        process_test_api()
    elif mode == "real":
        # Реальный режим с циклическим опросом
        while True:
            try:
                process_real_api()
                logger.info(f"Ожидание {POLLING_INTERVAL} секунд до следующего опроса...")
                time.sleep(POLLING_INTERVAL)
            except KeyboardInterrupt:
                logger.info("Остановка по запросу пользователя")
                break
            except Exception as e:
                logger.error(f"Ошибка в основном цикле: {e}")
                time.sleep(POLLING_INTERVAL)
    else:
        logger.error(f"Неизвестный режим: {mode}. Используйте 'test' или 'real'")


if __name__ == "__main__":
    main()