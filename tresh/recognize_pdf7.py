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

def find_main_blocks_by_width(text_blocks, width_tolerance=0.3, min_text_length=10):
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
    print(f"  Соотношение сторон: {largest_block['aspect_ratio']:.2f}")
    
    # Определяем допустимый диапазон ширины
    target_width = largest_block['width']
    min_width = target_width * (1 - width_tolerance)
    max_width = target_width * (1 + width_tolerance)
    
    print(f"\nДопустимый диапазон ширины: {min_width:.0f} - {max_width:.0f} (основа: {target_width:.0f})")
    
    # Ищем все блоки с похожей шириной
    matching_blocks = []
    for block in text_blocks:
        if min_width <= block['width'] <= max_width:
            matching_blocks.append(block)
    
    print(f"\nНайдено блоков с похожей шириной: {len(matching_blocks)} из {len(text_blocks)}")
    
    return matching_blocks, largest_block

def analyze_block_content(image, block_bbox):
    """Анализирует содержание блока"""
    x, y, w, h = block_bbox
    block_image = image[y:y+h, x:x+w]
    
    # Распознаем текст в блоке
    custom_config = r'--psm 6'
    text = pytesseract.image_to_string(block_image, lang="rus", config=custom_config).strip()
    
    # Анализируем содержание
    content_info = {
        'text_length': len(text),
        'line_count': len(text.split('\n')),
        'has_numbers': bool(re.search(r'\d', text)),
        'has_letters': bool(re.search(r'[а-яА-Яa-zA-Z]', text)),
        'text_preview': text[:100] + "..." if len(text) > 100 else text
    }
    
    return content_info

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

def convert_pdf_to_text_selected_blocks(image, width_tolerance=0.3):
    """Основная функция для обработки только выбранных блоков"""
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
    
    # Обрабатываем только выбранные блоки в правильном порядке
    for i, block_info in enumerate(main_blocks_sorted):
        x, y, w, h = block_info['bbox']
        
        print(f"\n--- Обработка основного блока {i+1} ---")
        print(f"Координаты: x={x}, y={y}, w={w}, h={h}")
        print(f"Размер: {w}x{h}, Площадь: {block_info['area']:.0f}")
        
        # Анализируем содержание блока
        content_info = analyze_block_content(image, (x, y, w, h))
        print(f"Содержание: {content_info['text_preview']}")
        print(f"Длина текста: {content_info['text_length']} символов")
        
        # Вырезаем блок
        block_image = image[y:y+h, x:x+w]
        
        # Обрабатываем блок
        processed_block, block_text = process_text_block(block_image, i+1)
        
        # Вставляем обработанный блок в результат
        result_image[y:y+h, x:x+w] = processed_block
        
        # Рисуем bounding box вокруг основного блока (синий)
        cv2.rectangle(result_image, (x, y), (x+w, y+h), (255, 0, 0), 3)
        cv2.putText(result_image, f'Block {i+1}', (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Сохраняем результаты
        all_text_data[f"block_{i+1}"] = {
            'bbox': (x, y, w, h),
            'size': (w, h),
            'area': block_info['area'],
            'content': block_text,
            'content_info': content_info,
            'position': i + 1
        }
        
        # Выводим распознанный текст из блока
        print(f"Распознанный текст блока {i+1}:")
        for key, text in block_text.items():
            print(f"  {key}: {text}")
    
    # Рисуем bounding boxes для всех остальных блоков (серым)
    all_blocks_sorted = sort_blocks_correctly(text_blocks, image.shape[1])
    for block_info in all_blocks_sorted:
        if block_info not in main_blocks_sorted:
            x, y, w, h = block_info['bbox']
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (128, 128, 128), 1)
            cv2.putText(result_image, 'Other', (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
    
    # Показываем результат
    cv2.imshow('Selected Text Blocks - Correct Order', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return all_text_data

def print_detailed_block_info(text_blocks, main_blocks, image_width):
    """Выводит подробную информацию о всех блоках в правильном порядке"""
    print("\n" + "="*60)
    print("ДЕТАЛЬНАЯ ИНФОРМАЦИЯ О БЛОКАХ (сверху вниз, справа налево)")
    print("="*60)
    
    # Сортируем все блоки правильно
    all_blocks_sorted = sort_blocks_correctly(text_blocks, image_width)
    main_blocks_sorted = sort_blocks_correctly(main_blocks, image_width)
    
    print(f"\nВсего блоков: {len(text_blocks)}")
    print(f"Основных блоков: {len(main_blocks)}")
    
    print("\nВсе блоки (сверху вниз, справа налево):")
    for i, block in enumerate(all_blocks_sorted):
        x, y, w, h = block['bbox']
        status = "ОСНОВНОЙ" if block in main_blocks_sorted else "вспомогательный"
        position = main_blocks_sorted.index(block) + 1 if block in main_blocks_sorted else "-"
        print(f"{i+1:2d}. [{status}] Pos: ({x:4d},{y:4d}) Size: {w:4d}x{h:4d} "
              f"Area: {block['area']:7.0f} MainPos: {position}")
    
    if main_blocks_sorted:
        print(f"\nОсновные блоки (правильный порядок):")
        for i, block in enumerate(main_blocks_sorted):
            x, y, w, h = block['bbox']
            print(f"{i+1:2d}. Pos: ({x:4d},{y:4d}) Size: {w:4d}x{h:4d} "
                  f"Area: {block['area']:7.0f}")

# Пример использования
path = '/home/user/rep/teach_tesserasseract/pdf/processed/КЛАБ.302231.008СБ_180567923.dwg.pdf'
image = extract_image(path, 0)

print("=== ОБРАБОТКА ОСНОВНЫХ ТЕКСТОВЫХ БЛОКОВ (правильный порядок) ===")
text_data = convert_pdf_to_text_selected_blocks(image, width_tolerance=0.3)

# Дополнительная информация о блоках
text_blocks = detect_text_blocks(image)
main_blocks, largest_block = find_main_blocks_by_width(text_blocks, width_tolerance=0.3)
print_detailed_block_info(text_blocks, main_blocks, image.shape[1])

end = datetime.datetime.now()
print(f"\nВремя выполнения: {end-start}")