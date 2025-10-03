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

def detect_text_blocks(image, min_area_ratio=0.005):
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
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 30)) #Ширина Высота
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
            text_blocks.append({
                'bbox': (x, y, w, h),
                'area': area,
                'width': w,
                'height': h,
                'aspect_ratio': w / h if h > 0 else 0
            })
    
    return text_blocks

def sort_blocks_correctly(text_blocks, image_width):
    """
    Сортирует блоки: сверху вниз, а при одинаковой высоте - справа налево
    """
    # Сортируем сначала по Y (сверху вниз), затем по X (справа налево)
    sorted_blocks = sorted(text_blocks, 
                          key=lambda block: (block['bbox'][1], -block['bbox'][0]))
    
    return sorted_blocks

def find_main_blocks_by_width(text_blocks, width_tolerance=0.3):
    """
    Находит основной блок и все блоки с похожей шириной
    """
    if not text_blocks:
        return [], None
    
    # Находим самый большой блок по площади
    largest_block = max(text_blocks, key=lambda x: x['area'])
    
    print(f"Самый большой блок:")
    print(f"  Размер: {largest_block['width']}x{largest_block['height']}")
    print(f"  Площадь: {largest_block['area']:.0f}")
    
    # Определяем допустимый диапазон ширины
    target_width = largest_block['width']
    min_width = target_width * (1 - width_tolerance)
    max_width = target_width * (1 + width_tolerance)
    
    print(f"Допустимый диапазон ширины: {min_width:.0f} - {max_width:.0f}")
    
    # Ищем все блоки с похожей шириной
    matching_blocks = []
    for block in text_blocks:
        if min_width <= block['width'] <= max_width:
            matching_blocks.append(block)
    
    print(f"Найдено блоков с похожей шириной: {len(matching_blocks)} из {len(text_blocks)}")
    
    return matching_blocks, largest_block

def extract_text_from_block(image, block_bbox):
    """Извлекает текст из блока без группировки"""
    x, y, w, h = block_bbox
    block_image = image[y:y+h, x:x+w]
    
    # Используем Tesseract для распознавания всего текста в блоке
    custom_config = r'--psm 6 -c preserve_interword_spaces=1'
    text = pytesseract.image_to_string(block_image, lang="rus", config=custom_config).strip()
    
    return text

def convert_pdf_to_text_simple(image, width_tolerance=0.3):
    """Упрощенная функция для распознавания текста в блоках"""
    # Обнаруживаем все текстовые блоки
    text_blocks = detect_text_blocks(image)
    
    if not text_blocks:
        print("Не найдено текстовых блоков")
        return
    
    print(f"Всего найдено блоков: {len(text_blocks)}")
    
    # Находим основные блоки по ширине
    main_blocks, largest_block = find_main_blocks_by_width(text_blocks, width_tolerance)
    
    if not main_blocks:
        print("Не найдено подходящих блоков")
        return
    
    # Правильно сортируем основные блоки: сверху вниз, справа налево
    main_blocks_sorted = sort_blocks_correctly(main_blocks, image.shape[1])
    
    # Создаем копию изображения для визуализации
    result_image = image.copy()
    
    all_text_data = {}
    full_text = ""
    
    print("\n" + "="*60)
    print("РАСПОЗНАВАНИЕ ТЕКСТА ПО БЛОКАМ")
    print("="*60)
    
    # Обрабатываем только выбранные блоки в правильном порядке
    for i, block_info in enumerate(main_blocks_sorted):
        x, y, w, h = block_info['bbox']
        
        print(f"\n--- Блок {i+1} ---")
        print(f"Координаты: ({x}, {y}) размер: {w}x{h}")
        
        # Извлекаем текст из блока
        block_text = extract_text_from_block(image, (x, y, w, h))
        
        # Очищаем и форматируем текст
        cleaned_text = clean_text(block_text)
        
        # Добавляем к полному тексту
        if cleaned_text:
            full_text += cleaned_text + "\n\n"
        
        # Сохраняем результаты
        all_text_data[f"block_{i+1}"] = {
            'bbox': (x, y, w, h),
            'size': (w, h),
            'text': cleaned_text,
            'position': i + 1
        }
        
        # Выводим распознанный текст
        print(f"Текст блока {i+1}:")
        print(cleaned_text)
        
        # Визуализация
        # Рисуем bounding box вокруг основного блока (синий)
        cv2.rectangle(result_image, (x, y), (x+w, y+h), (255, 0, 0), 3)
        cv2.putText(result_image, f'Block {i+1}', (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Рисуем bounding boxes для всех остальных блоков (серым)
    all_blocks_sorted = sort_blocks_correctly(text_blocks, image.shape[1])
    for block_info in all_blocks_sorted:
        if block_info not in main_blocks_sorted:
            x, y, w, h = block_info['bbox']
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (128, 128, 128), 1)
    
    # Выводим полный объединенный текст
    print("\n" + "="*60)
    print("ПОЛНЫЙ ТЕКСТ (все блоки последовательно)")
    print("="*60)
    print(full_text)
    
    # Показываем результат
    cv2.imshow('Text Blocks - Simple Recognition', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return all_text_data, full_text

def clean_text(text):
    """Очищает и форматирует распознанный текст"""
    if not text:
        return ""
    
    # Удаляем лишние пустые строки
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Объединяем строки в логические абзацы
    cleaned_lines = []
    current_paragraph = []
    
    for line in lines:
        # Если строка заканчивается на точку или двоеточие, начинаем новый абзац
        if line.endswith(('.', ':', ';')) or len(line) < 50:
            if current_paragraph:
                current_paragraph.append(line)
                cleaned_lines.append(' '.join(current_paragraph))
                current_paragraph = []
            else:
                cleaned_lines.append(line)
        else:
            current_paragraph.append(line)
    
    # Добавляем последний абзац, если он есть
    if current_paragraph:
        cleaned_lines.append(' '.join(current_paragraph))
    
    return '\n'.join(cleaned_lines)

def print_blocks_summary(text_blocks, main_blocks, image_width):
    """Выводит краткую информацию о блоках"""
    print("\n" + "="*60)
    print("СВОДКА ПО БЛОКАМ")
    print("="*60)
    
    # Сортируем все блоки правильно
    all_blocks_sorted = sort_blocks_correctly(text_blocks, image_width)
    main_blocks_sorted = sort_blocks_correctly(main_blocks, image_width)
    
    print(f"Всего блоков: {len(text_blocks)}")
    print(f"Основных блоков: {len(main_blocks)}")
    
    print("\nОсновные блоки (порядок обработки):")
    for i, block in enumerate(main_blocks_sorted):
        x, y, w, h = block['bbox']
        print(f"{i+1:2d}. Позиция: ({x:4d}, {y:4d}) Размер: {w:4d}x{h:4d}")

# Пример использования
path = '/home/user/rep/teach_tesserasseract/pdf/processed/КЛАБ.302231.008СБ_180567923.dwg.pdf'
image = extract_image(path, 0)

print("=== ПРОСТОЕ РАСПОЗНАВАНИЕ ТЕКСТА ПО БЛОКАМ ===")
text_data, full_text = convert_pdf_to_text_simple(image, width_tolerance=0.3)

# Сводка по блокам
text_blocks = detect_text_blocks(image)
main_blocks, largest_block = find_main_blocks_by_width(text_blocks, width_tolerance=0.3)
print_blocks_summary(text_blocks, main_blocks, image.shape[1])

end = datetime.datetime.now()
print(f"\nВремя выполнения: {end-start}")