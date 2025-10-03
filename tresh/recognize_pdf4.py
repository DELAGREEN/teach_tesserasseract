from pdf2image import convert_from_path
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import datetime
import re
from icecream import ic as icprint

start = datetime.datetime.now()

def extract_image(pdf_file: str, page_index) -> np.array:
    pages = convert_from_path(pdf_file, dpi=300)
    image = np.array(pages[page_index])
    return image

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(gray, -1, sharpen_kernel)
    binary = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return binary

def group_contours(contours, threshold_y=10, threshold_x=50):
    """Группирует контуры, которые находятся близко по вертикали и горизонтали"""
    if not contours:
        return []
    
    # Получаем bounding boxes для всех контуров
    boxes = [cv2.boundingRect(c) for c in contours]
    
    # Сортируем по Y координате
    boxes.sort(key=lambda x: x[1])
    
    groups = []
    current_group = [boxes[0]]
    
    for box in boxes[1:]:
        x, y, w, h = box
        last_box = current_group[-1]
        last_x, last_y, last_w, last_h = last_box
        
        # Проверяем, находятся ли боксы близко по вертикали и перекрываются по горизонтали
        vertical_close = abs(y - (last_y + last_h)) < threshold_y
        horizontal_overlap = not (x > last_x + last_w + threshold_x or last_x > x + w + threshold_x)
        
        if vertical_close and horizontal_overlap:
            current_group.append(box)
        else:
            groups.append(current_group)
            current_group = [box]
    
    if current_group:
        groups.append(current_group)
    
    return groups

def convert_pdf_to_text(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    binary = preprocess_image(img_rgb)

    # Используем более точные настройки для Tesseract
    custom_config = r'--psm 6'
    data = pytesseract.image_to_data(img_rgb, lang="rus", output_type=Output.DICT, config=custom_config)
    
    # Создаем маску для текстовых областей
    mask = np.zeros_like(binary)
    for i in range(len(data['level'])):
        if data['text'][i].strip():
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            mask[y:y+h, x:x+w] = 255
    
    # Находим основную текстовую область
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 60))
    dilated = cv2.dilate(mask, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Не найдено текстовых областей")
        return
    
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    largest_text_area = img_rgb[y:y+h, x:x+w]
    
    # Обрабатываем основную текстовую область для выделения блоков
    binary_text_area = preprocess_image(largest_text_area)
    
    # Используем морфологические операции для объединения близких текстовых элементов
    kernel_block = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 15))
    dilated_blocks = cv2.dilate(binary_text_area, kernel_block, iterations=2)
    
    contours, _ = cv2.findContours(dilated_blocks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Группируем контуры
    contour_groups = group_contours(contours)
    
    # Извлекаем текст из сгруппированных областей
    grouped_text = {}
    
    for i, group in enumerate(contour_groups):
        # Объединяем bounding boxes группы
        x_min = min(box[0] for box in group)
        y_min = min(box[1] for box in group)
        x_max = max(box[0] + box[2] for box in group)
        y_max = max(box[1] + box[3] for box in group)
        
        # Извлекаем область текста
        text_area = largest_text_area[y_min:y_max, x_min:x_max]
        
        # Распознаем текст
        text = pytesseract.image_to_string(text_area, lang="rus", config=custom_config).strip()
        
        if text:
            # Рисуем bounding box для визуализации
            cv2.rectangle(largest_text_area, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Пытаемся найти номер группы в начале текста
            match = re.match(r'^(\d+)[\.\s]*(.*)', text)
            if match:
                number, content = match.groups()
                if number in grouped_text:
                    grouped_text[number].append(content.strip())
                else:
                    grouped_text[number] = [content.strip()]
            else:
                # Если нет номера, используем индекс группы
                group_key = f"group_{i}"
                grouped_text[group_key] = [text]
    
    # Выводим результаты
    for key, texts in sorted(grouped_text.items(), 
                           key=lambda x: int(x[0]) if x[0].isdigit() else float('inf')):
        print(f"Группа {key}:")
        for text in texts:
            print(f"  {text}")
        print("-" * 50)
    
    # Показываем результат
    cv2.imshow('Text Areas', largest_text_area)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Альтернативный подход - используем данные напрямую от Tesseract
def convert_pdf_to_text_v2(image):
    """Альтернативный подход с использованием данных Tesseract напрямую"""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    #custom_config = r'--psm 6 -c preserve_interword_spaces=1'
    custom_config = r'--psm 6 -c preserve_interword_spaces=1'
    data = pytesseract.image_to_data(img_rgb, lang="rus", output_type=Output.DICT, config=custom_config)
    
    # Группируем слова по строкам и блокам
    current_block = []
    blocks = []
    
    for i in range(len(data['level'])):
        if data['text'][i].strip():
            block_num = data['block_num'][i]
            line_num = data['line_num'][i]
            word_num = data['word_num'][i]
            text = data['text'][i]
            conf = data['conf'][i]
            
            # Фильтруем по уверенности распознавания
            if conf > 30:
                current_block.append({
                    'text': text,
                    'block': block_num,
                    'line': line_num,
                    'word': word_num,
                    'bbox': (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                })
    
    # Группируем по блокам и строкам
    if current_block:
        # Сортируем по блокам, строкам и словам
        current_block.sort(key=lambda x: (x['block'], x['line'], x['word']))
        
        current_line = []
        lines = []
        last_line = -1
        
        for item in current_block:
            if item['line'] != last_line and current_line:
                lines.append(' '.join(current_line))
                current_line = []
            
            current_line.append(item['text'])
            last_line = item['line']
            
            # Рисуем bounding box для визуализации
            x, y, w, h = item['bbox']
            cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Выводим результат
        print("Распознанный текст (по строкам):")
        for i, line in enumerate(lines, 1):
            print(f"Строка {i}: {line}")
    
    cv2.imshow('Text Detection', img_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

path = '/home/user/rep/teach_tesserasseract/pdf/processed/К0810-2055.01.02_184820331.dwg.pdf'
image = extract_image(path, 0)

# оба подхода
#print("=== Подход 1: Морфологическая обработка ===")
convert_pdf_to_text(image)

#print("\n=== Подход 2: Прямая работа с Tesseract ===")
convert_pdf_to_text_v2(image)

print(image.shape)

path = '/home/user/rep/teach_tesserasseract/pdf/processed/65869914_Чертеж-РНАТ.dwg.pdf'
image = extract_image(path, 0)
print(image.shape)

convert_pdf_to_text(image)


convert_pdf_to_text_v2(image)



end = datetime.datetime.now()
print(f"Время выполнения: {end-start}")

