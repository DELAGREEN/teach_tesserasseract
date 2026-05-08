import os
os.environ["QT_QPA_PLATFORM"] = "xcb"   # убирает предупреждение Wayland

from pdf2image import convert_from_path
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import datetime
import tkinter as tk
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
    binary = cv2.adaptiveThreshold(
        sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    return binary


import math

def is_likely_drawing(roi, 
                      line_density_threshold=0.05,      # плотность горизонтальных/вертикальных линий
                      contour_area_threshold=0.3,       # мин. доля площади контуров от bbox
                      min_contour_vertices=4,           # мин. вершин для замкнутой фигуры
                      hough_line_threshold=0.02,        # доля пикселей на линиях (любых)
                      corner_response_threshold=0.1):   # доля угловых точек
    """
    Определяет, является ли блок чертежом (содержит замкнутые контуры, наклонные линии, дуги).
    Возвращает True, если хотя бы один критерий превышает порог.
    """
    if roi.size == 0:
        return False

    # 1. Бинаризация
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = binary.shape
    area_total = h * w
    if area_total == 0:
        return False

    # ------------------- 2. Контурный анализ (замкнутые фигуры) -------------------
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_contour_area = 0
    closed_shape_count = 0

    for cnt in contours:
        cnt_area = cv2.contourArea(cnt)
        if cnt_area > 50:  # игнорируем шум
            large_contour_area += cnt_area
            # Аппроксимируем контур многоугольником
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            vertices = len(approx)
            # Замкнутая фигура: если количество вершин >= min_contour_vertices и контур замкнут
            if vertices >= min_contour_vertices and cv2.isContourConvex(approx):
                closed_shape_count += 1

    # Доля площади контуров от всей области
    contour_ratio = large_contour_area / area_total if area_total > 0 else 0

    # ------------------- 3. Линии под любым углом (Hough) -------------------
    # Скелетизация (утончение) для лучшего выделения линий
    skeleton = cv2.ximgproc.thinning(binary) if hasattr(cv2, 'ximgproc') else binary
    lines = cv2.HoughLinesP(skeleton, rho=1, theta=np.pi/180, threshold=30,
                            minLineLength=max(5, min(w,h)//10),
                            maxLineGap=3)
    line_pixels = set()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # рисуем линию на временной маске, чтобы посчитать уникальные пиксели
            cv2.line(skeleton, (x1, y1), (x2, y2), 255, 2)  # используем skeleton как маску
            # более точный подсчёт: через алгоритм Брезенхэма, но для простоты:
            # просто посчитаем ненулевые пиксели в новой маске
    # После отрисовки всех линий считаем ненулевые пиксели (они уже на skeleton)
    hough_pixels = cv2.countNonZero(skeleton) if lines is not None else 0
    hough_ratio = hough_pixels / area_total if area_total > 0 else 0

    # ------------------- 4. Угловые точки (Harris) -------------------
    # Для цветного изображения преобразуем в float32
    gray_float = np.float32(gray)
    dst = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)
    # Порог: 0.01 * max(response)
    corner_thresh = 0.01 * dst.max()
    corners = np.argwhere(dst > corner_thresh)
    corner_count = len(corners)
    corner_ratio = corner_count / area_total if area_total > 0 else 0

    # ------------------- 5. (Опционально) Проверка на наличие текста -------------------
    # Если распознаваемый текст занимает меньше 20% площади блока – подозрительно
    try:
        text_data = pytesseract.image_to_data(roi, lang='rus', output_type=Output.DICT, config='--psm 6')
        text_pixels = 0
        for i in range(len(text_data['level'])):
            if text_data['text'][i].strip():
                x, y, tw, th = text_data['left'][i], text_data['top'][i], text_data['width'][i], text_data['height'][i]
                text_pixels += tw * th
        text_ratio = text_pixels / area_total
    except:
        text_ratio = 0.0

    # ------------------- Решающее правило -------------------
    # Блок является чертежом, если:
    #   - много замкнутых контуров (contour_ratio > contour_area_threshold)
    #   - много линий по Хафу (hough_ratio > hough_line_threshold)
    #   - много угловых точек (corner_ratio > corner_response_threshold)
    #   - при этом текст покрывает менее 20% площади (доп. условие)
    drawing_conditions = [
        contour_ratio > contour_area_threshold,
        hough_ratio > hough_line_threshold,
        corner_ratio > corner_response_threshold,
        (contour_ratio > 0.15 and text_ratio < 0.2)  # если мало текста, но есть контуры
    ]

    # Также проверяем оригинальные горизонтальные/вертикальные линии
    # (можно оставить старую логику или вызвать её как часть условий)
    # Для краткости я вызову функцию, которую вы уже написали, передав порог 0.05
    # (избегаем дублирования кода – вы можете добавить этот вызов, если нужно)
    
    return any(drawing_conditions)


# Обновлённая функция detect_text_blocks с новым порогом
def detect_text_blocks(image, min_area_ratio=0.005, drawing_threshold=0.05):
    """
    Обнаруживает текстовые блоки, отфильтровывая чертежи по расширенным критериям.
    Параметр drawing_threshold зарезервирован для обратной совместимости.
    """
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
        return []

    min_area = image.shape[0] * image.shape[1] * min_area_ratio
    text_blocks = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            roi = image[y:y+h, x:x+w]
            # Если это чертёж – пропускаем
            if is_likely_drawing(roi):
                continue
            text_blocks.append({
                "bbox": (x, y, w, h),
                "area": area,
                "width": w,
                "height": h,
                "aspect_ratio": w / h if h > 0 else 0,
            })
    return text_blocks


def sort_blocks_correctly(text_blocks, image_width):
    return sorted(text_blocks, key=lambda block: (block['bbox'][1], -block['bbox'][0]))


def find_main_blocks_by_width(text_blocks, width_tolerance=0.3):
    if not text_blocks:
        return [], None

    largest_block = max(text_blocks, key=lambda x: x['area'])
    print(f"Самый большой блок: {largest_block['width']}x{largest_block['height']} пл.{largest_block['area']:.0f}")

    target_width = largest_block['width']
    min_width = target_width * (1 - width_tolerance)
    max_width = target_width * (1 + width_tolerance)

    matching_blocks = [b for b in text_blocks if min_width <= b['width'] <= max_width]
    print(f"Найдено блоков с похожей шириной: {len(matching_blocks)} из {len(text_blocks)}")
    return matching_blocks, largest_block


def extract_text_from_block(image, block_bbox):
    x, y, w, h = block_bbox
    block_image = image[y:y+h, x:x+w]
    custom_config = r'--psm 6 -c preserve_interword_spaces=1'
    text = pytesseract.image_to_string(block_image, lang="rus", config=custom_config).strip()
    return text


def clean_text(text):
    if not text:
        return ""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    cleaned_lines = []
    current_paragraph = []
    for line in lines:
        if line.endswith(('.', ':', ';')) or len(line) < 50:
            if current_paragraph:
                current_paragraph.append(line)
                cleaned_lines.append(' '.join(current_paragraph))
                current_paragraph = []
            else:
                cleaned_lines.append(line)
        else:
            current_paragraph.append(line)
    if current_paragraph:
        cleaned_lines.append(' '.join(current_paragraph))
    return '\n'.join(cleaned_lines)


def convert_pdf_to_text_simple(image, width_tolerance=0.3, show_image=False):
    """
    Упрощенная функция для распознавания текста в блоках.
    Параметр show_image: True – показать окно с визуализацией, False – только распознавание (для сервера).
    """
    text_blocks = detect_text_blocks(image)
    if not text_blocks:
        print("Не найдено текстовых блоков")
        return

    print(f"Всего найдено блоков: {len(text_blocks)}")
    main_blocks, _ = find_main_blocks_by_width(text_blocks, width_tolerance)
    if not main_blocks:
        print("Не найдено подходящих блоков")
        return

    main_blocks_sorted = sort_blocks_correctly(main_blocks, image.shape[1])

    # Создаём копию изображения только если нужен показ (экономия памяти на сервере)
    result_image = image.copy() if show_image else None
    all_text_data = {}
    full_text = ""

    print("\n" + "="*60)
    print("РАСПОЗНАВАНИЕ ТЕКСТА ПО БЛОКАМ")
    print("="*60)

    for i, block_info in enumerate(main_blocks_sorted):
        x, y, w, h = block_info['bbox']
        print(f"\n--- Блок {i+1} --- ({x},{y}) {w}x{h}")
        block_text = extract_text_from_block(image, (x, y, w, h))
        cleaned_text = clean_text(block_text)
        if cleaned_text:
            full_text += cleaned_text + "\n\n"

        all_text_data[f"block_{i+1}"] = {
            'bbox': (x, y, w, h),
            'size': (w, h),
            'text': cleaned_text,
            'position': i+1
        }
        print(cleaned_text)

        # Рисуем рамки только при show_image
        if show_image:
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (255,0,0), 3)
            cv2.putText(result_image, f'Block {i+1}', (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    # Показать остальные блоки серым (если нужен показ)
    if show_image:
        all_blocks_sorted = sort_blocks_correctly(text_blocks, image.shape[1])
        for block_info in all_blocks_sorted:
            if block_info not in main_blocks_sorted:
                x, y, w, h = block_info['bbox']
                cv2.rectangle(result_image, (x, y), (x+w, y+h), (128,128,128), 1)

    print("\n" + "="*60)
    print("ПОЛНЫЙ ТЕКСТ")
    print("="*60)
    print(full_text)

    # Показ изображения только если явно запрошен
    if show_image:
        try:
            import tkinter as tk
            root = tk.Tk()
            screen_w = root.winfo_screenwidth()
            screen_h = root.winfo_screenheight()
            root.destroy()
        except:
            screen_w, screen_h = 1920, 1080

        h_img, w_img = result_image.shape[:2]
        scale = min(screen_w / w_img, screen_h / h_img, 1.0)
        if scale < 1.0:
            new_w = int(w_img * scale)
            new_h = int(h_img * scale)
            result_image = cv2.resize(result_image, (new_w, new_h))

        cv2.imshow('Text Blocks - Simple Recognition', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return all_text_data, full_text

def print_blocks_summary(text_blocks, main_blocks, image_width):
    print("\n" + "="*60)
    print("СВОДКА ПО БЛОКАМ")
    print("="*60)
    main_sorted = sort_blocks_correctly(main_blocks, image_width)
    print(f"Всего блоков: {len(text_blocks)}")
    print(f"Основных блоков: {len(main_blocks)}")
    for i, b in enumerate(main_sorted):
        x, y, w, h = b['bbox']
        print(f"{i+1:2d}. ({x:4d}, {y:4d}) {w:4d}x{h:4d}")


# ========== ЗАПУСК ==========
if __name__ == "__main__":
    path = '/home/icebook7/rep/teach_tesserasseract/pdf/РНАТ.713141.724_182778663.dwg.pdf'
    image = extract_image(path, 0)

    # Для сервера: show_image=False (окно не появится)
    # Для отладки на локальной машине: show_image=True
    SHOW_PREVIEW = True   # <-- поменять на True, если нужен визуальный вывод

    print("=== ПРОСТОЕ РАСПОЗНАВАНИЕ ТЕКСТА ПО БЛОКАМ ===")
    text_data, full_text = convert_pdf_to_text_simple(image, width_tolerance=0.3, show_image=SHOW_PREVIEW)

    text_blocks = detect_text_blocks(image)
    main_blocks, _ = find_main_blocks_by_width(text_blocks, 0.3)
    print_blocks_summary(text_blocks, main_blocks, image.shape[1])

    end = datetime.datetime.now()
    print(f"\nВремя выполнения: {end-start}")