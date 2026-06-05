import os
import sys
import datetime
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from pdf2image import convert_from_bytes
from logger import logger
import re

os.environ["QT_QPA_PLATFORM"] = "xcb"


class OCRProcessor:
    def __init__(self, dpi=300, lang="rus", psm=6, char_whitelist="", min_area_ratio=0.0005, width_tolerance=0.3,
                 save_images=False, save_dir=None, use_advanced=True,
                 kernel_width=50, kernel_height=30, dilation_iterations=3,
                 debug=False, production_mode=True, frac_height_ratio=2.0, word_merge_kernel_width=15):
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
        self.frac_height_ratio = frac_height_ratio
        self.word_merge_kernel_width = word_merge_kernel_width

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

        # === Проверка на наличие текста ===
        has_text = False
        try:
            # Быстрый OCR с ограничением символов (только русские, цифры, основные знаки)
            config = '--psm 7 -c tessedit_char_whitelist=АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789.,:;()-'
            ocr_result = pytesseract.image_to_string(roi, lang=self.lang, config=config).strip()
            # Если нашлось хотя бы 3 символа (буквы/цифры) — считаем, что текст есть
            if len(re.search(r'[А-Яа-я0-9]', ocr_result)) >= 3:
                has_text = True
        except:
            pass
        if has_text:
            if self.debug and block_info:
                print(f"  Блок {block_info} оставлен как текст (найдены символы)")
            return False   # не чертёж

        reasons = []
        if contour_ratio > 0.3:
            reasons.append(f"contour_ratio={contour_ratio:.3f}")
        if hough_ratio > 0.02 and (contour_ratio > 0.05 or corner_ratio > 0.05):
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
    
    def _find_fraction_line(self, word_roi):
        """Ищет горизонтальную линию внутри ROI, возвращает Y-координату центра линии или None."""
        gray = cv2.cvtColor(word_roi, cv2.COLOR_BGR2GRAY)
        # Бинаризация, чтобы линия стала белой
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Морфологически выделяем длинные горизонтальные линии
        # Ядро: ширина = почти весь ROI, высота = 1
        line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (word_roi.shape[1] // 3, 1))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, line_kernel)

        # Находим контуры линий
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Ищем самый длинный контур (по ширине)
        best_cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(best_cnt)
        if w < word_roi.shape[1] * 0.3:  # линия должна занимать хотя бы 30% ширины слова
            return None
        return y + h // 2

    def filter_drawing_from_roi(self, roi):
        """
        Удаляет из ROI всё, что похоже на чертёж, оставляя только текст.
        Возвращает очищенное изображение (BGR).
        """
        if roi.size == 0:
            return roi
        # Бинаризация
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Поиск компонент связности
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # Создаём маску для закрашивания "чертёжных" пикселей
        drawing_mask = np.zeros_like(binary)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                        stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            aspect_ratio = max(w, h) / (min(w, h) if min(w, h) > 0 else 1)
            
            # Эвристики "чертёжного" элемента:
            # - очень длинная тонкая линия
            # - площадь мала, но вытянутость большая
            # - НЕ содержит текста (быстрая проверка mini-OCR на фрагменте)
            is_line = (aspect_ratio > 10) and (area < 200)  # пороги настраиваются
            is_possible_drawing = (aspect_ratio > 5) and (area < 500)
            
            if is_line or is_possible_drawing:
                # Дополнительно убедимся, что это не фрагмент буквы
                roi_fragment = roi[y:y+h, x:x+w]
                if self._is_text_fragment(roi_fragment):
                    continue  # оставляем
                # Закрашиваем компоненту на маске
                drawing_mask[labels == i] = 255
        
        # Применяем маску: заменяем пиксели, помеченные как чертёж, на белый цвет
        cleaned = roi.copy()
        cleaned[drawing_mask > 0] = (255, 255, 255)  # белый фон
        return cleaned

    def _is_text_fragment(self, fragment):
        """Быстрая проверка, содержит ли фрагмент хотя бы один символ текста."""
        if fragment.size == 0:
            return False
        try:
            config = '--psm 7 -c tessedit_char_whitelist=0123456789АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя.,:;()"'
            res = pytesseract.image_to_string(fragment, lang=self.lang, config=config).strip()
            return len(res) > 0
        except:
            return False
        
    def handle_fractions_in_block(self, block_image):
        """
        Возвращает (очищенное_изображение, текст_дроби_или_None).
        Если дробь найдена, она вырезается из изображения и возвращается её текстовое представление.
        """
        gray = cv2.cvtColor(block_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Ищем горизонтальные линии (длинные, сплошные)
        lines = cv2.HoughLinesP(binary, rho=1, theta=np.pi/180, threshold=30,
                                minLineLength=block_image.shape[1] * 0.4,  # минимум 40% ширины блока
                                maxLineGap=3)
        if lines is None:
            return block_image, None  # дробей нет
        
        # Группируем близкие по Y линии (одна дробь может состоять из нескольких отрезков)
        line_ys = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Убедимся, что линия почти горизонтальна
            if abs(y2 - y1) > 5:
                continue
            line_ys.append((min(y1, y2) + max(y1, y2)) // 2)
        
        if not line_ys:
            return block_image, None
        
        # Выбираем самую длинную (по протяжённости) или среднюю Y
        y_center = int(np.median(line_ys))
        
        # Проверяем, что выше и ниже есть текст (быстрый OCR)
        top_part = block_image[0:y_center, :]
        bot_part = block_image[y_center:, :]
        
        top_text = pytesseract.image_to_string(top_part, lang=self.lang, config='--psm 7').strip()
        bot_text = pytesseract.image_to_string(bot_part, lang=self.lang, config='--psm 7').strip()
        
        if top_text and bot_text:
            fraction_str = f"{top_text}/{bot_text}"
            # Вырезаем область дроби из изображения (закрашиваем белым)
            cleaned = block_image.copy()
            # Расширяем область удаления чуть выше и ниже линии, чтобы убрать остатки
            margin = 2
            cleaned[max(0, y_center-margin):min(block_image.shape[0], y_center+margin), :] = (255, 255, 255)
            return cleaned, fraction_str
        
        return block_image, None
    
    def get_merged_word_boxes(self, roi):
        """Возвращает bounding box'ы слов после горизонтального склеивания."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Получаем сырые слова от Tesseract
        data = pytesseract.image_to_data(gray, lang=self.lang,
                                        output_type=Output.DICT,
                                        config='--psm 6')
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        for i in range(len(data['level'])):
            text = data['text'][i].strip()
            if not text:
                continue
            w, h = data['width'][i], data['height'][i]
            if w <= 0 or h <= 0:
                continue
            x, y = data['left'][i], data['top'][i]
            mask[y:y+h, x:x+w] = 255

        # Горизонтальное склеивание (только по X, не по Y)
        merge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                                (self.word_merge_kernel_width, 1))
        dilated = cv2.dilate(mask, merge_kernel, iterations=1)

        # Находим контуры объединённых слов
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        words = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 5 or h < 5:          # отсекаем артефакты
                continue
            words.append({'x': x, 'y': y, 'w': w, 'h': h})
        return words

    def analyze_block_for_fractions(self, block_image, frac_height_ratio=None,
                                min_text_len=2, debug_words=False, debug_image_path=None):
        if frac_height_ratio is None:
            frac_height_ratio = self.frac_height_ratio

        # Получаем объединённые слова
        words = self.get_merged_word_boxes(block_image)
        if not words:
            return block_image, [], []

        # Медианная высота
        heights = [wd['h'] for wd in words]
        median_h = np.median(heights)
        if median_h < 5:
            return block_image, [], []

        # Разделяем на высокие и нормальные
        high_words = [wd for wd in words if wd['h'] >= median_h * frac_height_ratio]
        normal_words = [wd for wd in words if wd['h'] < median_h * frac_height_ratio]

        # Отладка — сохранение картинки со словами
        if debug_words and debug_image_path:
            debug_img = block_image.copy()
            for wd in words:
                color = (0, 255, 0) if wd['h'] < median_h * frac_height_ratio else (0, 0, 255)
                cv2.rectangle(debug_img, (wd['x'], wd['y']),
                            (wd['x']+wd['w'], wd['y']+wd['h']), color, 1)
                cv2.putText(debug_img, f"{wd['h']}", (wd['x'], wd['y']-2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.imwrite(debug_image_path, debug_img)
            print(f"[DEBUG] Слова блока сохранены: {debug_image_path}")

        # Обработка высоких слов как и раньше
        fraction_texts = []
        high_word_boxes = []
        if high_words:
            cleaned_img = block_image.copy()
            for wd in high_words:
                x, y, w, h = wd['x'], wd['y'], wd['w'], wd['h']
                # Вырежем чуть больше, чтобы не обрезать края
                x1 = max(0, x - 2)
                y1 = max(0, y - 2)
                x2 = min(block_image.shape[1], x + w + 2)
                y2 = min(block_image.shape[0], y + h + 2)
                word_roi = block_image[y1:y2, x1:x2]

                # Ищем горизонтальную линию (можно через морфологию или Hough)
                line_y = self._find_fraction_line(word_roi)
                if line_y is None:
                    # Если линия не найдена, делим пополам (грубо)
                    line_y = word_roi.shape[0] // 2

                top_part = word_roi[0:line_y, :]
                bot_part = word_roi[line_y:, :]

                # Распознаём числитель и знаменатель
                num_text = pytesseract.image_to_string(top_part, lang=self.lang, config='--psm 7').strip()
                den_text = pytesseract.image_to_string(bot_part, lang=self.lang, config='--psm 7').strip()
                # Очищаем от мусора (оставляем буквы, цифры, точку, запятую, минус)
                import re
                num_clean = re.sub(r'[^\w.,\-]', '', num_text)
                den_clean = re.sub(r'[^\w.,\-]', '', den_text)

                if num_clean and den_clean:
                    fraction_texts.append(f"{num_clean}/{den_clean}")
                    cleaned_img[y1:y2, x1:x2] = (255, 255, 255)
                    high_word_boxes.append((wd['x'], wd['y'], wd['w'], wd['h']))
                return cleaned_img, fraction_texts, high_word_boxes

            return block_image, [], []


    def _analyze_by_word_height(self, block_image, frac_height_ratio=1.8, min_text_len=2):
        """
        Гибридный детектор дроби:
        1) пытается найти по аномальной высоте слов,
        2) если не получилось – ищет горизонтальную линию морфологически.
        Возвращает: (cleaned_image, fraction_texts, high_word_boxes)
        """
        # -------- попытка 1: анализ высоты слов (быстрый) --------
        cleaned, frac_texts, boxes = self._analyze_by_word_height(block_image, frac_height_ratio, min_text_len)
        if frac_texts:
            return cleaned, frac_texts, boxes

        # -------- попытка 2: поиск горизонтальной линии --------
        gray = cv2.cvtColor(block_image, cv2.COLOR_BGR2GRAY)
        # бинаризуем с адаптивным порогом, чтобы линия точно была чёрной
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)

        # Морфологическое выделение длинных горизонтальных линий
        line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (block_image.shape[1]//4, 1))
        # открытие уберёт всё, что короче 1/4 ширины и тоньше 1 пикселя
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, line_kernel)

        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return block_image, [], []

        # Ищем самый длинный контур, ширина которого > 40% ширины блока
        min_line_width = block_image.shape[1] * 0.4
        best_line = None
        best_len = 0
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > min_line_width and h < 5:  # линия тонкая
                if w > best_len:
                    best_len = w
                    best_line = (x, y, w, h)

        if best_line is None:
            return block_image, [], []

        # Извлекаем Y-центр линии
        xl, yl, wl, hl = best_line
        y_center = yl + hl//2

        # Проверяем, что выше и ниже есть хоть какие-то пиксели
        top_part = block_image[0:y_center, :]
        bot_part = block_image[y_center:, :]
        if top_part.size == 0 or bot_part.size == 0:
            return block_image, [], []

        # Пробуем распознать числитель и знаменатель
        num_text = pytesseract.image_to_string(top_part, lang=self.lang, config='--psm 7').strip()
        den_text = pytesseract.image_to_string(bot_part, lang=self.lang, config='--psm 7').strip()
        # Оставляем только цифры/буквы/точки
        import re
        num_clean = re.sub(r'[^\w.,\-]', '', num_text)
        den_clean = re.sub(r'[^\w.,\-]', '', den_text)

        if num_clean and den_clean:
            # Закрашиваем линию и немного пространства вокруг неё
            cleaned_img = block_image.copy()
            margin = 2
            y1 = max(0, y_center - margin)
            y2 = min(block_image.shape[0], y_center + margin)
            cleaned_img[y1:y2, :] = (255, 255, 255)
            # Создаём фиктивный box для отрисовки (охватывает верх и низ)
            h_total = block_image.shape[0]
            box = (0, 0, block_image.shape[1], h_total)  # весь блок
            return cleaned_img, [f"{num_clean}/{den_clean}"], [box]

        return block_image, [], []

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
        logger.debug(f"min_area = {min_area:.0f} (изображение {image.shape[1]}x{image.shape[0]})")
        text_blocks = []
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            if area < min_area:
                logger.debug(f"  Блок #{idx} ({w}x{h}) отброшен: площадь {area:.0f} < {min_area:.0f}")
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

    def sort_blocks_by_columns(self, text_blocks, x_tolerance=None):
        """
        Сортировка блоков для чертежей: справа налево по колонкам,
        внутри колонки — сверху вниз.
        """
        if not text_blocks:
            return []
        if x_tolerance is None:
            """
            x_tolerance – допуск по горизонтали (ось X) для объединения блоков в одну колонку.
            Если разница X-координат двух блоков <= x_tolerance, они считаются лежащими
            в одной вертикальной зоне и сортируются внутри неё сверху вниз.
            """
            # TODO(DELAGREEN): в перспективе перенести в config
            x_tolerance = getattr(self, 'x_tolerance', 50)   # можно добавить в __init__

        # Сортируем все блоки по X убыванию
        sorted_by_x = sorted(text_blocks, key=lambda b: b['bbox'][0], reverse=True)

        columns = []
        current_column = [sorted_by_x[0]]
        for block in sorted_by_x[1:]:
            # если X текущего блока близок к последнему в колонке — та же колонка
            if abs(block['bbox'][0] - current_column[-1]['bbox'][0]) <= x_tolerance:
                current_column.append(block)
            else:
                # завершили колонку: сортируем внутри по Y
                current_column.sort(key=lambda b: b['bbox'][1])
                columns.append(current_column)
                current_column = [block]
        # последнюю колонку тоже сортируем по Y
        current_column.sort(key=lambda b: b['bbox'][1])
        columns.append(current_column)

        # Объединяем колонки в плоский список (порядок колонок — справа налево)
        result = []
        for col in columns:
            result.extend(col)
        return result

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

    def extract_text_from_block(self, image, block_bbox, debug_words=False):
        x, y, w, h = block_bbox
        block_image = image[y:y+h, x:x+w].copy()

        # Анализируем дроби (с опциональной отладкой слов)
        debug_path = None
        if debug_words and self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            debug_path = os.path.join(self.save_dir, f"words_{x}_{y}.png")

        cleaned_image, fraction_texts, fraction_boxes = self.analyze_block_for_fractions(
            block_image,
            frac_height_ratio=self.frac_height_ratio,  # добавь параметр в __init__ (по умолчанию 2.0)
            debug_words=debug_words,
            debug_image_path=debug_path
        )

        # Очищаем от чертёжных линий (твой метод)
        cleaned_image = self.filter_drawing_from_roi(cleaned_image)

        # Основной OCR
        config = r'--psm 6 -c preserve_interword_spaces=1'
        if self.char_whitelist:
            config += f' -c tessedit_char_whitelist="{self.char_whitelist}"'
        main_text = pytesseract.image_to_string(cleaned_image, lang=self.lang, config=config).strip()

        # Собираем итоговый текст
        final_text = main_text
        if fraction_texts:
            for frac in fraction_texts:
                final_text += f" {frac}"

        has_fraction = bool(fraction_texts)
        return final_text, has_fraction, fraction_boxes

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
            sorted_blocks = self.sort_blocks_by_columns(main_blocks, 50)
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
                # В методе extract_text, в цикле по sorted_blocks:
                for i, block in enumerate(sorted_blocks):
                    block_text, has_fraction, fraction_boxes = self.extract_text_from_block(
                        image, block['bbox'], debug_words=self.debug
                    )
                    cleaned = self.clean_text(block_text)
                    if cleaned:
                        page_text.append(cleaned)

                    x_blk, y_blk, w_blk, h_blk = block['bbox']

                    # Цвет основного блока: розовый, если есть дроби, иначе синий
                    blk_color = (255, 0, 255) if has_fraction else (255, 0, 0)  # BGR
                    cv2.rectangle(result_image, (x_blk, y_blk), (x_blk + w_blk, y_blk + h_blk), blk_color, 3)
                    label = f'{i+1}'
                    if has_fraction:
                        label += ' (F)'
                    cv2.putText(result_image, label, (x_blk, y_blk - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, blk_color, 3)

                    # Рисуем розовые прямоугольники вокруг найденных дробей
                    for (fx, fy, fw, fh) in fraction_boxes:
                        abs_x = x_blk + fx
                        abs_y = y_blk + fy
                        cv2.rectangle(result_image, (abs_x, abs_y), (abs_x + fw, abs_y + fh), (255, 0, 255), 2)
                save_path = os.path.join(self.save_dir, 
                                         f"{file_name}_{page_idx+1}_{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')}.png")
                cv2.imwrite(save_path, result_image)
                logger.debug(f"Сохранено размеченное изображение: {save_path}")

        return '\n\n'.join(full_text)


# ----------------------- АВТОНОМНЫЙ ЗАПУСК -----------------------
if __name__ == "__main__":
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(description="Визуализация фильтрации блоков OCR")
    parser.add_argument("pdf_path", nargs="?", default=None, help="Путь к PDF-файлу")
    parser.add_argument("--debug", action="store_true", help="Включить отладочный вывод в консоль")
    parser.add_argument("--min-area-ratio", type=float, default=None, help="Переопределить min_area_ratio из конфига")
    parser.add_argument("--kernel-width", type=int, default=None)
    parser.add_argument("--kernel-height", type=int, default=None)
    parser.add_argument("--dilations", type=int, default=None)
    parser.add_argument("--width-tolerance", type=float, default=None)
    parser.add_argument("--no-advanced", action="store_true", help="Отключить фильтрацию чертежей (для сравнения)")
    args = parser.parse_args()

    # Загрузка конфигурации (как у вас)
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.ini")
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config_content = os.path.expandvars(f.read())
    import configparser
    config = configparser.ConfigParser(interpolation=None)
    config.read_string(config_content)

    def get_config(section, key, fallback=None, type=str):
        env_key = f"{section}_{key}".upper()
        env_value = os.environ.get(env_key)
        if env_value is not None:
            if type == bool:
                return env_value.lower() in ('true', '1', 'yes')
            return type(env_value)
        if type == bool:
            return config.getboolean(section, key, fallback=fallback)
        elif type == int:
            return config.getint(section, key, fallback=fallback)
        elif type == float:
            return config.getfloat(section, key, fallback=fallback)
        else:
            return config.get(section, key, fallback=fallback)

    def override_if_set(arg_value, current):
        return current if arg_value is None else arg_value

    # Параметры из конфига / командной строки
    DPI = get_config("OCR", "dpi", fallback=300, type=int)
    MIN_AREA_RATIO = get_config("OCR", "min_area_ratio", fallback=0.0005, type=float)
    WIDTH_TOLERANCE = get_config("OCR", "width_tolerance", fallback=0.3, type=float)
    KERNEL_WIDTH = get_config("OCR", "kernel_width", fallback=20, type=int)
    KERNEL_HEIGHT = get_config("OCR", "kernel_height", fallback=30, type=int)
    DILATION_ITERATIONS = get_config("OCR", "dilation_iterations", fallback=3, type=int)
    USE_ADVANCED = get_config("OCR", "use_advanced_recognition", fallback=False, type=bool)
    DEBUG = get_config("Processing", "debug", fallback=False, type=bool)

    # Переопределение из аргументов
    MIN_AREA_RATIO = override_if_set(args.min_area_ratio, MIN_AREA_RATIO)
    KERNEL_WIDTH = override_if_set(args.kernel_width, KERNEL_WIDTH)
    KERNEL_HEIGHT = override_if_set(args.kernel_height, KERNEL_HEIGHT)
    DILATION_ITERATIONS = override_if_set(args.dilations, DILATION_ITERATIONS)
    WIDTH_TOLERANCE = override_if_set(args.width_tolerance, WIDTH_TOLERANCE)
    DEBUG = override_if_set(args.debug, DEBUG)

    # Подготовка PDF
    default_pdf = "/home/user/rep/teach_tesseract_new/teach_tesserasseract/test_server/mock_files/103/К7856-12014(0).dwg.pdf"
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

    # Создаём два процессора:
    # 1. "штатный" – для определения, прошёл бы блок фильтры.
    # 2. "сырой" – для извлечения всех контуров без фильтрации.
    processor_normal = OCRProcessor(
        dpi=DPI, lang="rus", psm=6,
        min_area_ratio=MIN_AREA_RATIO,
        width_tolerance=WIDTH_TOLERANCE,
        save_images=False,
        use_advanced=USE_ADVANCED,
        kernel_width=KERNEL_WIDTH,
        kernel_height=KERNEL_HEIGHT,
        dilation_iterations=DILATION_ITERATIONS,
        debug=False,
        production_mode=False,
        word_merge_kernel_width=15
    )

    # Для сбора всех контуров: min_area_ratio = 0, use_advanced = False
    processor_raw = OCRProcessor(
        dpi=DPI, lang="rus", psm=6,
        min_area_ratio=0.0,
        width_tolerance=WIDTH_TOLERANCE,
        save_images=False,
        use_advanced=False,
        kernel_width=KERNEL_WIDTH,
        kernel_height=KERNEL_HEIGHT,
        dilation_iterations=DILATION_ITERATIONS,
        debug=False,
        production_mode=False
    )

    images = processor_raw.pdf_to_images(pdf_bytes)
    if not images:
        print("❌ Не удалось извлечь изображения")
        sys.exit(1)

    # Для каждой страницы анализируем и показываем
    for page_idx, image in enumerate(images):
        print(f"\n====== Страница {page_idx+1} ======")
        # Получаем ВСЕ контуры с сырым процессором
        all_blocks = processor_raw.detect_text_blocks(image)  # вернёт все, т.к. min_area=0 и без фильтра чертежей

        # Вычисляем реальное min_area, которое используется штатным процессором
        min_area = image.shape[0] * image.shape[1] * MIN_AREA_RATIO

        # Разделим блоки на категории и соберём причины
        accepted_blocks = []
        rejected_area_blocks = []   # (bbox, area)
        rejected_drawing_blocks = [] # (bbox, reasons)

        for block in all_blocks:
            x, y, w, h = block['bbox']
            area = block['area']
            if area < min_area:
                rejected_area_blocks.append(((x, y, w, h), area))
                continue
            # Проверяем, отбросил бы его is_likely_drawing штатного процессора
            roi = image[y:y+h, x:x+w]
            # Используем тот же метод, но без изменения кода класса
            is_drawing = processor_normal.is_likely_drawing(roi, block_info=None)
            if is_drawing:
                # Получим причины (дублируем логику is_likely_drawing, чтобы не менять оригинал)
                # Этот блок кода повторяет вычисления is_likely_drawing для получения причин
                h_roi, w_roi = roi.shape[:2]
                area_total = h_roi * w_roi
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                large_contour_area = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 50)
                contour_ratio = large_contour_area / area_total
                lines = cv2.HoughLinesP(binary, rho=1, theta=np.pi/180, threshold=30,
                                        minLineLength=max(5, min(w_roi, h_roi)//10), maxLineGap=3)
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
                    reasons.append(f"контуры={contour_ratio:.3f}")
                if hough_ratio > 0.02:
                    reasons.append(f"линии={hough_ratio:.3f}")
                if corner_ratio > 0.1:
                    reasons.append(f"углы={corner_ratio:.3f}")
                if contour_ratio > 0.15 and hough_ratio > 0.01:
                    reasons.append(f"конт+лин ({contour_ratio:.3f}, {hough_ratio:.3f})")
                if not reasons:
                    reasons.append("неизвестная причина")  # на всякий случай
                rejected_drawing_blocks.append(((x, y, w, h), reasons))
            else:
                accepted_blocks.append((x, y, w, h))

        # Вывод статистики в консоль
        print(f"Всего контуров: {len(all_blocks)}")
        print(f"Принято (текст): {len(accepted_blocks)}")
        print(f"Отброшено по площади (< {min_area:.0f}): {len(rejected_area_blocks)}")
        print(f"Отброшено как чертёж: {len(rejected_drawing_blocks)}")
        if rejected_drawing_blocks and DEBUG:
            for bbox, reasons in rejected_drawing_blocks:
                print(f"  Блок {bbox}: {', '.join(reasons)}")

        # Визуализация на копии изображения
        vis_image = image.copy()
        # Принятые – зеленый
        # В цикле по accepted_blocks:
        for (x, y, w, h) in accepted_blocks:
            cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 255, 0), 3)
            roi = image[y:y+h, x:x+w]
            words = processor_normal.get_merged_word_boxes(roi)
            if words:
                heights = [wd['h'] for wd in words]
                median_h = np.median(heights)
                for wd in words:
                    abs_x = x + wd['x']
                    abs_y = y + wd['y']
                    color = (0, 0, 255) if wd['h'] >= median_h * processor_normal.frac_height_ratio else (0, 255, 0)
                    cv2.rectangle(vis_image, (abs_x, abs_y), (abs_x + wd['w'], abs_y + wd['h']), color, 1)
                    cv2.putText(vis_image, f"{wd['h']}", (abs_x, abs_y - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


        # Отброшенные по площади – серый
        for ((x, y, w, h), area) in rejected_area_blocks:
            cv2.rectangle(vis_image, (x, y), (x+w, y+h), (128, 128, 128), 2)
            cv2.putText(vis_image, f"area={area:.0f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (128, 128, 128), 1)
        # Отброшенные как чертёж – красный, подписываем первую причину
        for ((x, y, w, h), reasons) in rejected_drawing_blocks:
            cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 0, 255), 3)
            text = reasons[0] if reasons else "drawing"
            cv2.putText(vis_image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 1)

        # Масштабируем, чтобы влезло в экран
        try:
            import tkinter as tk
            root = tk.Tk()
            sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
            root.destroy()
            h_img, w_img = vis_image.shape[:2]
            scale = min(sw/w_img, sh/h_img, 1.0)
            if scale < 1.0:
                vis_image = cv2.resize(vis_image, (int(w_img*scale), int(h_img*scale)))
        except:
            pass

        #cv2.imshow(f'Фильтрация блоков – Страница {page_idx+1}', vis_image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        cv2.imwrite("debug_filtered.png", vis_image)