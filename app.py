import os
import time
import shutil
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import datetime
import re
from pdf2image import convert_from_path
from watchdog.observers import Observer  # Если потребуется, можно заменить на PollingObserver
from watchdog.events import FileSystemEventHandler
from dotenv import load_dotenv
import configparser
import logging
import sys

# ----------------------- Загрузка переменных окружения -----------------------
# Загружаем переменные из файла .env (файл .env должен находиться в той же директории, что и запуск приложения)
load_dotenv()

# ----------------------- Настройка логирования -----------------------
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ----------------------- Загрузка конфигурации с подстановкой переменных -----------------------
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.ini')
# Читаем содержимое config.ini как текст и выполняем замену переменных (например, ${MONITORED_DIR})
with open(CONFIG_PATH, encoding="utf-8") as f:
    config_content = os.path.expandvars(f.read())

# Отключаем встроенную интерполяцию, поскольку замена уже выполнена
config = configparser.ConfigParser(interpolation=None)
config.read_string(config_content)

# Получаем пути (значения подставлены функцией os.path.expandvars)
MONITORED_DIR = config.get('Paths', 'monitored_dir')
RESULTS_DIR   = config.get('Paths', 'results_dir')
TRESH_DIR     = config.get('Paths', 'tresh_dir')
PROCESSED_DIR = config.get('Paths', 'processed_dir')

# Параметры обработки
SAVE_IMAGES = config.getboolean('Processing', 'save_images')
DPI = config.getint('Processing', 'dpi')

# Параметры watchdog
SLEEP_DELAY = config.getfloat('Watchdog', 'sleep_delay')

# Создаем необходимые папки, если их нет
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TRESH_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ----------------------- ФУНКЦИИ -----------------------
def preprocess_image(image):
    """
    Предобработка изображения:
      - Перевод в оттенки серого,
      - Гауссово размытие,
      - Повышение резкости,
      - Адаптивное бинаризование.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    sharpen_kernel = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])
    sharpened = cv2.filter2D(gray, -1, sharpen_kernel)
    binary = cv2.adaptiveThreshold(sharpened, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV,
                                   11, 2)
    return binary

def process_pdf(file_path):
    """
    Обработка PDF-файла:
      - Конвертирует первую страницу PDF в изображение.
      - Определяет наибольшую область с текстом и обводит её на полном изображении.
      - Извлекает область для распознавания, на ней обводит найденные строки,
        распознаёт текст и сохраняет результаты.
    """
    logger.info(f"Начало обработки файла: {file_path}")

    # Конвертация первой страницы PDF в изображение (используем DPI из конфигурации)
    pages = convert_from_path(file_path, dpi=DPI)
    if not pages:
        logger.error(f"Не удалось конвертировать PDF: {file_path}")
        return
    img = np.array(pages[0])
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Копия полного изображения для сохранения с обведенной областью
    full_outlined_image = img_rgb.copy()

    # Предобработка для выделения текстовых областей
    binary = preprocess_image(img_rgb)
    data = pytesseract.image_to_data(binary, lang="rus",
                                     output_type=Output.DICT,
                                     config="--psm 6")

    # Получаем координаты распознанных блоков
    boxes = [(data['left'][i], data['top'][i], data['width'][i], data['height'][i])
             for i in range(len(data['level']))
             if data['text'][i].strip()]

    mask = np.zeros_like(binary)
    for (x, y, w, h) in boxes:
        mask[y:y+h, x:x+w] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 60))
    dilated = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        logger.error(f"Не найдены контуры в файле {file_path}.")
        return
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)

    # Обводим найденную область на полном изображении
    cv2.rectangle(full_outlined_image, (x, y), (x+w, y+h), (0, 255, 0), 3)

    # Извлекаем область для распознавания
    largest_text_area = img_rgb[y:y+h, x:x+w]
    binary_text = preprocess_image(largest_text_area)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 20))
    dilated_small = cv2.dilate(binary_text, kernel_small, iterations=1)
    contours_small, _ = cv2.findContours(dilated_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Сортируем найденные контуры по координате Y
    contours_small = sorted(contours_small, key=lambda cnt: cv2.boundingRect(cnt)[1])

    recognized_text_lines = []
    # Копия области для обведения найденных строк
    outlined_area_image = largest_text_area.copy()

    for contour in contours_small:
        cx, cy, cw, ch = cv2.boundingRect(contour)
        text_area = largest_text_area[cy:cy+ch, cx:cx+cw]
        processed_text = preprocess_image(text_area)
        text = pytesseract.image_to_string(processed_text, lang="rus", config="--psm 6").strip()
        if text:
            recognized_text_lines.append(text)
            if SAVE_IMAGES:
                cv2.rectangle(outlined_area_image, (cx, cy), (cx+cw, cy+ch), (0, 255, 0), 2)

    recognized_text = "\n".join(recognized_text_lines)

    # Формирование имен файлов результатов на основе исходного имени и времени обработки
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if SAVE_IMAGES:
        # Сохраняем полное изображение с обведенной областью
        full_outlined_filename = os.path.join(RESULTS_DIR, f"{base_name}_{timestamp}_full_outlined.jpg")
        cv2.imwrite(full_outlined_filename, cv2.cvtColor(full_outlined_image, cv2.COLOR_RGB2BGR))

        # Сохраняем область с обведенными строками
        outlined_area_filename = os.path.join(RESULTS_DIR, f"{base_name}_{timestamp}_outlined.jpg")
        cv2.imwrite(outlined_area_filename, cv2.cvtColor(outlined_area_image, cv2.COLOR_RGB2BGR))

        # Сохраняем оригинальную (вырезанную) область без обводки
        cropped_area_filename = os.path.join(RESULTS_DIR, f"{base_name}_{timestamp}_cropped.jpg")
        cv2.imwrite(cropped_area_filename, cv2.cvtColor(largest_text_area, cv2.COLOR_RGB2BGR))

    text_filename = os.path.join(RESULTS_DIR, f"{base_name}_{timestamp}.txt")
    with open(text_filename, "w", encoding="utf-8") as f:
        f.write(recognized_text)

    logger.info(f"Обработан файл: {file_path}")
    if SAVE_IMAGES:
        logger.info(
            f"Сохранены изображения:\n  Полное с обводкой: {full_outlined_filename}\n"
            f"  Область с обводкой: {outlined_area_filename}\n"
            f"  Оригинальная область: {cropped_area_filename}"
        )
    logger.info(f"Сохранен текстовый файл: {text_filename}")

def process_existing_files():
    """
    Обрабатывает файлы, уже находящиеся в папке MONITORED_DIR.
    Если файл имеет расширение .dwg.pdf – обрабатывает его и перемещает в PROCESSED_DIR,
    остальные перемещает в TRESH_DIR.
    """
    for file in os.listdir(MONITORED_DIR):
        full_path = os.path.join(MONITORED_DIR, file)
        # Пропускаем каталоги (включая results, tresh, processed)
        if os.path.isdir(full_path):
            continue
        file_lower = file.lower()
        if file_lower.endswith(".dwg.pdf"):
            try:
                process_pdf(full_path)
                dest_path = os.path.join(PROCESSED_DIR, file)
                shutil.move(full_path, dest_path)
                logger.info(f"Файл {file} обработан и перемещён в папку processed")
            except Exception as e:
                logger.error(f"Ошибка при обработке {full_path}: {e}")
        else:
            try:
                dest_path = os.path.join(TRESH_DIR, file)
                shutil.move(full_path, dest_path)
                logger.info(f"Файл {file} перемещён в папку tresh")
            except Exception as e:
                logger.error(f"Ошибка при перемещении {file} в папку tresh: {e}")

# ----------------------- WATCHDOG -----------------------
class PDFEventHandler(FileSystemEventHandler):
    """Обработчик событий для мониторинга создания файлов в указанной папке."""
    def on_created(self, event):
        # Игнорируем события для папок
        if event.is_directory:
            return

        # Задержка согласно параметру из конфигурации
        time.sleep(SLEEP_DELAY)

        file_path = event.src_path
        file_name = os.path.basename(file_path)
        file_lower = file_name.lower()

        if file_lower.endswith(".dwg.pdf"):
            try:
                process_pdf(file_path)
                dest_path = os.path.join(PROCESSED_DIR, file_name)
                shutil.move(file_path, dest_path)
                logger.info(f"Файл {file_name} обработан и перемещён в папку processed")
            except Exception as e:
                logger.error(f"Ошибка при обработке {file_path}: {e}")
        else:
            try:
                dest_path = os.path.join(TRESH_DIR, file_name)
                shutil.move(file_path, dest_path)
                logger.info(f"Файл {file_name} перемещён в папку tresh")
            except Exception as e:
                logger.error(f"Ошибка при перемещении {file_name} в папку tresh: {e}")

if __name__ == "__main__":
    # Сначала обрабатываем все уже лежащие в папке файлы
    process_existing_files()

    # Запуск watchdog для мониторинга новых файлов
    event_handler = PDFEventHandler()
    observer = Observer()
    observer.schedule(event_handler, MONITORED_DIR, recursive=False)
    #Если указать recursive=True, то любые изменения, происходящие не только в указанной директории, 
    #но и в её подкаталогах, также будут отслеживаться. 
    observer.start()
    logger.info(f"Мониторинг папки {MONITORED_DIR} запущен...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logger.info("Остановка мониторинга по KeyboardInterrupt...")
    observer.join()
