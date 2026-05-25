import os
import sys
import datetime
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from pdf2image import convert_from_bytes
from logger import logger

os.environ["QT_QPA_PLATFORM"] = "xcb"


class OCRProcessor:
    def __init__(self, dpi=300, lang="rus", psm=6, char_whitelist="", min_area_ratio=0.0005, width_tolerance=0.3,
                 save_images=False, save_dir=None, use_advanced=True,
                 kernel_width=50, kernel_height=30, dilation_iterations=3,
                 debug=False, production_mode=True):
        self.dpi = dpi
        self.lang = lang
        self.psm = psm
        self.min_area_ratio = min_area_ratio
        self.width_tolerance = width_tolerance
        self.save_images = save_images
        self.save_dir = save_dir or "ocr_debug"
        self.use_advanced = use_advanced
        self.kernel_width = kernel_width
        self.kernel_height = kernel_height
        self.dilation_iterations = dilation_iterations
        self.debug = debug
        self.production_mode = production_mode
        self.char_whitelist = char_whitelist
        if self.save_images or not self.production_mode:
            os.makedirs(self.save_dir, exist_ok=True)

    def pdf_to_images(self, pdf_bytes):
        try:
            pil_images = convert_from_bytes(pdf_bytes, dpi=self.dpi)
            images = [np.array(img) for img in pil_images]
            logger.debug(f"PDF преобразован в {len(images)} изображений")
            return images
        except Exception as e:
            logger.error(f"Ошибка конвертации PDF: {e}")
            return []

    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(gray, -1, sharpen_kernel)
        binary = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        return binary

    def is_likely_drawing(self, roi, block_info=None):
        """Возвращает True, если блок содержит много линий/контуров (чертёж)."""
        if roi.size == 0:
            return False
        h, w = roi.shape[:2]
        area_total = h * w
        if area_total == 0:
            return False

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large_contour_area = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 50)
        contour_ratio = large_contour_area / area_total

        lines = cv2.HoughLinesP(binary, rho=1, theta=np.pi/180, threshold=30,
                                minLineLength=max(5, min(w, h)//10), maxLineGap=3)
        line_mask = np.zeros_like(binary)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
        hough_ratio = cv2.countNonZero(line_mask) / area_total

        gray_float = np.float32(gray)
        dst = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)
        dst = cv2.dilate(dst, None)
        corners = np.argwhere(dst > 0.01 * dst.max())
        corner_ratio = len(corners) / area_total

        reasons = []
        if contour_ratio > 0.3:
            reasons.append(f"contour_ratio={contour_ratio:.3f}")
        if hough_ratio > 0.02:
            reasons.append(f"hough_ratio={hough_ratio:.3f}")
        if corner_ratio > 0.1:
            reasons.append(f"corner_ratio={corner_ratio:.3f}")
        if contour_ratio > 0.15 and hough_ratio > 0.01:
            reasons.append(f"contour+hough ({contour_ratio:.3f}, {hough_ratio:.3f})")

        if reasons:
            if self.debug and block_info:
                print(f"  Блок {block_info} отброшен как чертёж: {', '.join(reasons)}")
            return True
        return False

    def detect_text_blocks(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        binary = self.preprocess_image(img_rgb)

        data = pytesseract.image_to_data(img_rgb, lang=self.lang, output_type=Output.DICT,
                                         config=f'--psm {self.psm}')

        mask = np.zeros_like(binary)
        for i in range(len(data["level"])):
            if data["text"][i].strip():
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                mask[y:y+h, x:x+w] = 255

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernel_width, self.kernel_height))
        dilated = cv2.dilate(mask, kernel, iterations=self.dilation_iterations)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        min_area = image.shape[0] * image.shape[1] * self.min_area_ratio
        if self.debug:
            print(f"min_area = {min_area:.0f} (изображение {image.shape[1]}x{image.shape[0]})")
        text_blocks = []
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            if area < min_area:
                if self.debug:
                    print(f"  Блок #{idx} ({w}x{h}) отброшен: площадь {area:.0f} < {min_area:.0f}")
                continue
            roi = image[y:y+h, x:x+w]
            if self.use_advanced and self.is_likely_drawing(roi, block_info=f"#{idx} ({w}x{h})"):
                continue
            text_blocks.append({
                "bbox": (x, y, w, h),
                "area": area,
                "width": w,
                "height": h,
                "aspect_ratio": w / h if h > 0 else 0,
            })
        if self.debug:
            print(f"Найдено {len(text_blocks)} текстовых блоков из {len(contours)} контуров")
        return text_blocks

    def sort_blocks_correctly(self, text_blocks):
        """Сортировка блоков сверху вниз, при равном Y — справа налево."""
        return sorted(text_blocks, key=lambda block: (block['bbox'][1], -block['bbox'][0]))

    def find_main_blocks_by_width(self, text_blocks, width_tolerance=None):
        if not text_blocks:
            return [], None
        if width_tolerance is None:
            width_tolerance = self.width_tolerance
        widths = [b['width'] for b in text_blocks]
        median_width = np.median(widths)
        min_width = median_width * (1 - width_tolerance)
        max_width = median_width * (1 + width_tolerance)
        if self.debug:
            print(f"Медианная ширина блоков: {median_width:.0f}, допуск: {min_width:.0f}-{max_width:.0f}")
            for i, b in enumerate(text_blocks):
                status = "✓" if min_width <= b['width'] <= max_width else "✗"
                print(f"  Блок {i}: {b['width']}x{b['height']} {status}")
        matching_blocks = [b for b in text_blocks if min_width <= b['width'] <= max_width]
        if self.debug:
            print(f"Основных блоков: {len(matching_blocks)} из {len(text_blocks)}")
        return matching_blocks, None

    def extract_text_from_block(self, image, block_bbox):
        x, y, w, h = block_bbox
        block_image = image[y:y+h, x:x+w]
        config = r'--psm 6 -c preserve_interword_spaces=1'
        if self.char_whitelist:
            config += f' -c tessedit_char_whitelist="{self.char_whitelist}"'
        return pytesseract.image_to_string(block_image, lang=self.lang, config=config).strip()

    def clean_text(self, text):
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

    def extract_text(self, pdf_bytes, file_name):
        images = self.pdf_to_images(pdf_bytes)
        if not images:
            return ""
        full_text = []
        for page_idx, image in enumerate(images):
            if self.debug:
                print(f"\n--- Страница {page_idx+1} ---")
            text_blocks = self.detect_text_blocks(image)
            if not text_blocks:
                continue
            main_blocks, _ = self.find_main_blocks_by_width(text_blocks)
            if not main_blocks:
                main_blocks = text_blocks
            sorted_blocks = self.sort_blocks_correctly(main_blocks)
            page_text = []
            for block_info in sorted_blocks:
                block_text = self.extract_text_from_block(image, block_info['bbox'])
                cleaned = self.clean_text(block_text)
                if cleaned:
                    page_text.append(cleaned)
            if page_text:
                full_text.append('\n'.join(page_text))

            # Сохраняем размеченное изображение, если не продакшн
            if not self.production_mode or self.save_images:
                result_image = image.copy()
                for i, block in enumerate(sorted_blocks):
                    x, y, w, h = block['bbox']
                    cv2.rectangle(result_image, (x, y), (x+w, y+h), (255, 0, 0), 3)
                    cv2.putText(result_image, f'{i+1}', (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
                save_path = os.path.join(self.save_dir, 
                                         f"{file_name}_{page_idx+1}_{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')}.png")
                cv2.imwrite(save_path, result_image)
                logger.debug(f"Сохранено размеченное изображение: {save_path}")

        return '\n\n'.join(full_text)


# ----------------------- АВТОНОМНЫЙ ЗАПУСК -----------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OCR Processor для извлечения текста из PDF")
    parser.add_argument("pdf_path", nargs="?", default=None, help="Путь к PDF-файлу")
    parser.add_argument("--debug", action="store_true", help="Включить отладочный вывод")
    parser.add_argument("--min-area-ratio", type=float, default=0.0005, help="Минимальная доля площади блока")
    parser.add_argument("--kernel-width", type=int, default=50, help="Ширина ядра морфологии")
    parser.add_argument("--kernel-height", type=int, default=30, help="Высота ядра морфологии")
    parser.add_argument("--dilations", type=int, default=3, help="Число итераций расширения")
    parser.add_argument("--width-tolerance", type=float, default=0.3, help="Допуск по ширине для основного столбца")
    parser.add_argument("--font-scale", type=float, default=2.8, help="Масштаб шрифта номеров блоков")
    parser.add_argument("--no-advanced", action="store_true", help="Отключить фильтрацию чертежей")
    args = parser.parse_args()

    default_pdf = "/home/user/rep/teach_tesserasseract/pdf/processed/КЛАБ.302231.008СБ_180567923.dwg.pdf"
    pdf_path = args.pdf_path
    if not pdf_path:
        if os.path.exists(default_pdf):
            pdf_path = default_pdf
            print(f"Используется путь по умолчанию: {pdf_path}")
        else:
            pdf_path = input("Введите путь к PDF-файлу: ").strip()

    if not pdf_path or not os.path.exists(pdf_path):
        print(f"❌ Файл не найден: {pdf_path}")
        sys.exit(1)

    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()

    # Автономный запуск – не продакшн, поэтому production_mode=False
    processor = OCRProcessor(
        dpi=300, lang="rus", psm=6,
        min_area_ratio=args.min_area_ratio,
        width_tolerance=args.width_tolerance,
        save_images=True, save_dir="results/ocr_debug",
        use_advanced=not args.no_advanced,
        kernel_width=args.kernel_width,
        kernel_height=args.kernel_height,
        dilation_iterations=args.dilations,
        debug=args.debug,
        production_mode=False
    )

    images = processor.pdf_to_images(pdf_bytes)
    if not images:
        print("❌ Не удалось извлечь изображения")
        sys.exit(1)

    image = images[0]
    print(f"Обрабатывается страница 1 из {len(images)}...")

    text_blocks = processor.detect_text_blocks(image)
    if not text_blocks:
        print("❌ Текстовые блоки не найдены.")
        sys.exit(0)

    main_blocks, _ = processor.find_main_blocks_by_width(text_blocks)
    if not main_blocks:
        print("Не удалось выделить основные блоки, используются все")
        main_blocks = text_blocks

    sorted_main = processor.sort_blocks_correctly(main_blocks)
    full_text = ""
    print("\nРАСПОЗНАННЫЙ ТЕКСТ:\n")
    for i, block in enumerate(sorted_main):
        text = processor.extract_text_from_block(image, block['bbox'])
        cleaned = processor.clean_text(text)
        if cleaned:
            full_text += cleaned + "\n\n"
        print(f"--- Блок {i+1} ({block['width']}x{block['height']}) ---")
        print(cleaned)
        print()

    # Дополнительная визуализация с крупными номерами (сохраняется поверх того, что уже сделал extract_text)
    result_image = image.copy()
    font_scale = args.font_scale
    for i, block in enumerate(sorted_main):
        x, y, w, h = block['bbox']
        cv2.rectangle(result_image, (x, y), (x+w, y+h), (255, 0, 0), 3)
        cv2.putText(result_image, f'{i+1}', (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), max(2, int(font_scale*2)))

    all_sorted = processor.sort_blocks_correctly(text_blocks)
    for block in all_sorted:
        if block not in sorted_main:
            x, y, w, h = block['bbox']
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (128, 128, 128), 1)

    os.makedirs("results", exist_ok=True)
    cv2.imwrite(os.path.join("results", "annotated_blocks.png"), result_image)
    with open(os.path.join("results", "recognized_text.txt"), 'w', encoding='utf-8') as f:
        f.write(full_text)
    print("✅ Результаты сохранены в results/")

    try:
        import tkinter as tk
        root = tk.Tk()
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        h_img, w_img = result_image.shape[:2]
        scale = min(sw/w_img, sh/h_img, 1.0)
        if scale < 1.0:
            result_image = cv2.resize(result_image, (int(w_img*scale), int(h_img*scale)))
        cv2.imshow('Text Blocks', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        pass