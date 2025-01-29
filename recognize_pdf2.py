from pdf2image import convert_from_path
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import datetime

start = datetime.datetime.now()

def convert_pdf_to_text(path):
    pages = convert_from_path(path, dpi=300)
    img = np.array(pages[0])
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    # Используем Tesseract для получения данных о тексте
    data = pytesseract.image_to_data(img_rgb, lang="rus", output_type=Output.DICT)

    # Собираем координаты всех боксов текста
    boxes = [(data['left'][i], data['top'][i], data['width'][i], data['height'][i])
             for i in range(len(data['level'])) if data['text'][i].strip()]

    # Создаем пустое изображение для маски
    mask = np.zeros_like(gray)
    for (x, y, w, h) in boxes:
        mask[y:y+h, x:x+w] = 255

    # Применяем морфологические операции для объединения близко расположенных текстовых блоков
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 60))
    dilated = cv2.dilate(mask, kernel, iterations=1)

    # Находим контуры объединенных блоков
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Находим самую большую область текста
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    largest_text_area = img_rgb[y:y+h, x:x+w]

    # Бинаризуем выделенную область
    gray = cv2.cvtColor(largest_text_area, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Разбиваем область на более мелкие сегменты
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 30))
    dilated_small = cv2.dilate(thresh, kernel_small, iterations=1)
    contours, _ = cv2.findContours(dilated_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Распознаем текст в каждой области
    recognized_text = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        text_area = largest_text_area[y:y+h, x:x+w]
        gray_text = cv2.cvtColor(text_area, cv2.COLOR_BGR2GRAY)
        _, binary_text = cv2.threshold(gray_text, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(binary_text, lang="rus").strip()
        if text:
            recognized_text.append(text)
            # Отрисовываем прямоугольные области текста
            cv2.rectangle(largest_text_area, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Выводим распознанный текст
    for i, text in enumerate(recognized_text):
        print(f"Область текста {i+1}:")
        print(text)

    cv2.imshow('Largest Text Area with Boxes', largest_text_area)
    cv2.waitKey(0)

path = '/home/nzxt/rep/teach_tesserasseract/pdf/РНАТ.723351.069_Сегмент.dwg.pdf'
convert_pdf_to_text(path)

end = datetime.datetime.now()
print(end-start)
