import datetime
import os
import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_bytes
from logger import logger

# Здесь импортируем константы, которые будут передаваться при создании экземпляра,
# но не тянем их из config.ini глобально – класс должен быть независимым.

class OCRProcessor:
    """
    Класс для извлечения и фильтрации текста из PDF-байтов с помощью Tesseract OCR.
    """
    def __init__(self, dpi=300, lang="rus+eng", psm=6, save_images=False, save_dir=None, use_advanced=False):
        self.dpi = dpi
        self.lang = lang
        self.psm = psm
        self.save_images = save_images
        self.save_dir = save_dir or "ocr_debug"
        self.use_advanced = use_advanced
        if self.save_images:
            os.makedirs(self.save_dir, exist_ok=True)

    def pdf_to_images(self, pdf_bytes):
        """
        Конвертирует PDF-байты в список PIL Image.
        """
        try:
            images = convert_from_bytes(pdf_bytes, dpi=self.dpi)
            logger.debug(f"PDF преобразован в {len(images)} изображений")
            return images
        except Exception as e:
            logger.error(f"Ошибка конвертации PDF: {e}")
            return []

    def image_to_text(self, image, page_num=0):
        """
        Распознавание текста на одном изображении.
        page_num используется для сохранения отладочных картинок.
        """
        try:
            if self.use_advanced:
                image = self._preprocess_image(image)

            if self.save_images:
                fname = f"page_{page_num}_{datetime.datetime.now().strftime('%H%M%S%f')}.png"
                image.save(os.path.join(self.save_dir, fname))

            text = pytesseract.image_to_string(
                image,
                lang=self.lang,
                config=f'--psm {self.psm}'
            )
            return text
        except Exception as e:
            logger.error(f"Ошибка OCR на странице {page_num}: {e}")
            return ""

    def _preprocess_image(self, image):
        """
        Дополнительная предобработка изображения (улучшение качества).
        """
        import numpy as np
        import cv2
        from PIL import Image
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        img_cv = cv2.equalizeHist(img_cv)
        _, img_cv = cv2.threshold(img_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(img_cv)

    def extract_text(self, pdf_bytes):
        """
        Основной метод: pdf_bytes -> полный текст со всех страниц.
        """
        images = self.pdf_to_images(pdf_bytes)
        if not images:
            return ""

        full_text = []
        for i, img in enumerate(images):
            text = self.image_to_text(img, page_num=i)
            # Базовая очистка: удаляем лишние пробелы и пустые строки
            cleaned = '\n'.join(line.strip() for line in text.splitlines() if line.strip())
            if cleaned:
                full_text.append(cleaned)
        return '\n'.join(full_text)

    def filter_technical_requirements(self, text, selection_keywords=None, exclude_keywords=None):
        """
        Фильтрация текста по ключевым словам.
        Оставляет только строки, содержащие selection_keywords,
        и удаляет строки, содержащие exclude_keywords.
        """
        if not text:
            return ""

        if selection_keywords is None:
            selection_keywords = []
        if exclude_keywords is None:
            exclude_keywords = []

        lines = text.splitlines()
        filtered = []
        for line in lines:
            if exclude_keywords and any(kw.lower() in line.lower() for kw in exclude_keywords):
                continue
            if selection_keywords:
                if any(kw.lower() in line.lower() for kw in selection_keywords):
                    filtered.append(line)
            else:
                filtered.append(line)
        return '\n'.join(filtered)