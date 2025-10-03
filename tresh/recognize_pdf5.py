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
    
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes.sort(key=lambda x: x[1])
    
    groups = []
    current_group = [boxes[0]]
    
    for box in boxes[1:]:
        x, y, w, h = box
        last_box = current_group[-1]
        last_x, last_y, last_w, last_h = last_box
        
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

def detect_text_blocks(image, min_area_ratio=0.01):
    """Обнаруживает все текстовые блоки на изображении"""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    binary = preprocess_image(img_rgb)

    # Используем Tesseract для создания маски текстовых областей
    custom_config = r'--psm 6'
    data = pytesseract.image_to_data(img_rgb, lang="rus", output_type=Output.DICT, config=custom_config)
    
    # Создаем маску для текстовых областей
    mask = np.zeros_like(binary)
    for i in range(len(data['level'])):
        if data['text'][i].strip():
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            mask[y:y+h, x:x+w] = 255
    
    # Морфологические операции для объединения текстовых областей
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 30))
    dilated = cv2.dilate(mask, kernel, iterations=3)
    
    # Находим все контуры текстовых блоков
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Не найдено текстовых областей")
        return []
    
    # Фильтруем контуры по площади
    min_area = image.shape[0] * image.shape[1] * min_area_ratio
    text_blocks = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            text_blocks.append((x, y, w, h))
    
    # Сортируем блоки слева направо
    text_blocks.sort(key=lambda block: block[0])
    
    return text_blocks

def process_text_block(block_image, block_num):
    """Обрабатывает отдельный текстовый блок"""
    img_rgb = cv2.cvtColor(block_image, cv2.COLOR_BGR2RGB)
    binary_block = preprocess_image(block_image)
    
    # Морфологические операции для выделения текстовых групп внутри блока
    kernel_block = cv2.getStructuringElement(cv2.MORPH_RECT, (200, 30))
    dilated_blocks = cv2.dilate(binary_block, kernel_block, iterations=2)
    
    contours, _ = cv2.findContours(dilated_blocks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Группируем контуры внутри блока
    contour_groups = group_contours(contours, threshold_y=5, threshold_x=20)
    
    # Извлекаем текст из сгруппированных областей
    grouped_text = {}
    
    for i, group in enumerate(contour_groups):
        # Объединяем bounding boxes группы
        x_min = min(box[0] for box in group)
        y_min = min(box[1] for box in group)
        x_max = max(box[0] + box[2] for box in group)
        y_max = max(box[1] + box[3] for box in group)
        
        # Извлекаем область текста
        text_area = block_image[y_min:y_max, x_min:x_max]
        
        # Распознаем текст
        custom_config = r'--psm 6'
        text = pytesseract.image_to_string(text_area, lang="rus", config=custom_config).strip()
        
        if text:
            # Рисуем bounding box для визуализации
            cv2.rectangle(block_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Пытаемся найти номер группы в начале текста
            match = re.match(r'^(\d+)[\.\s]*(.*)', text)
            if match:
                number, content = match.groups()
                grouped_text[number] = content.strip()
            else:
                # Если нет номера, используем индекс группы
                group_key = f"group_{i}"
                grouped_text[group_key] = text
    
    return block_image, grouped_text

def convert_pdf_to_text_multiple_blocks(image):
    """Основная функция для обработки нескольких текстовых блоков"""
    # Обнаруживаем все текстовые блоки
    text_blocks = detect_text_blocks(image)
    
    if not text_blocks:
        print("Не найдено текстовых блоков")
        return
    
    print(f"Найдено текстовых блоков: {len(text_blocks)}")
    
    # Создаем копию изображения для визуализации
    result_image = image.copy()
    
    all_text_data = {}
    
    # Обрабатываем каждый блок отдельно
    for i, (x, y, w, h) in enumerate(text_blocks):
        print(f"\n--- Обработка блока {i+1} ---")
        print(f"Координаты: x={x}, y={y}, w={w}, h={h}")
        
        # Вырезаем блок
        block_image = image[y:y+h, x:x+w]
        
        # Обрабатываем блок
        processed_block, block_text = process_text_block(block_image, i+1)
        
        # Вставляем обработанный блок в результат
        result_image[y:y+h, x:x+w] = processed_block
        
        # Рисуем bounding box вокруг всего блока
        cv2.rectangle(result_image, (x, y), (x+w, y+h), (255, 0, 0), 3)
        cv2.putText(result_image, f'Block {i+1}', (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Сохраняем результаты
        all_text_data[f"block_{i+1}"] = block_text
        
        # Выводим распознанный текст из блока
        print(f"Текст блока {i+1}:")
        for key, text in block_text.items():
            print(f"  {key}: {text}")
    
    # Показываем результат с всеми блоками
    cv2.imshow('All Text Blocks', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return all_text_data

def convert_pdf_to_text_v2_multiple_blocks(image):
    """Альтернативный подход для нескольких блоков с использованием данных Tesseract"""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    custom_config = r'--psm 6 -c preserve_interword_spaces=1'
    data = pytesseract.image_to_data(img_rgb, lang="rus", output_type=Output.DICT, config=custom_config)
    
    # Группируем слова по блокам Tesseract
    blocks_data = {}
    
    for i in range(len(data['level'])):
        if data['text'][i].strip() and data['conf'][i] > 30:
            block_num = data['block_num'][i]
            line_num = data['line_num'][i]
            word_num = data['word_num'][i]
            text = data['text'][i]
            
            if block_num not in blocks_data:
                blocks_data[block_num] = {}
            
            if line_num not in blocks_data[block_num]:
                blocks_data[block_num][line_num] = []
            
            blocks_data[block_num][line_num].append({
                'text': text,
                'word_num': word_num,
                'bbox': (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            })
    
    # Визуализируем и выводим результаты
    result_image = image.copy()
    
    print("Распознанный текст (по блокам Tesseract):")
    for block_num in sorted(blocks_data.keys()):
        print(f"\n--- Блок {block_num} ---")
        
        # Собираем текст блока
        block_text = []
        for line_num in sorted(blocks_data[block_num].keys()):
            # Сортируем слова в строке
            line_words = sorted(blocks_data[block_num][line_num], key=lambda x: x['word_num'])
            line_text = ' '.join(word['text'] for word in line_words)
            block_text.append(line_text)
            
            # Рисуем bounding box для всей строки
            if line_words:
                x_coords = [word['bbox'][0] for word in line_words]
                y_coords = [word['bbox'][1] for word in line_words]
                x_end_coords = [word['bbox'][0] + word['bbox'][2] for word in line_words]
                y_end_coords = [word['bbox'][1] + word['bbox'][3] for word in line_words]
                
                x_min = min(x_coords)
                y_min = min(y_coords)
                x_max = max(x_end_coords)
                y_max = max(y_end_coords)
                
                cv2.rectangle(result_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Выводим текст блока
        for i, line in enumerate(block_text, 1):
            print(f"Строка {i}: {line}")
    
    cv2.imshow('Text Detection - Multiple Blocks', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Пример использования
path = '/home/user/rep/teach_tesserasseract/pdf/processed/КЛАБ.302231.008СБ_180567923.dwg.pdf'
image = extract_image(path, 0)

print("=== Подход 1: Обработка нескольких текстовых блоков ===")
text_data = convert_pdf_to_text_multiple_blocks(image)

print("\n=== Подход 2: Прямая работа с Tesseract (множественные блоки) ===")
convert_pdf_to_text_v2_multiple_blocks(image)

end = datetime.datetime.now()
print(f"Время выполнения: {end-start}")