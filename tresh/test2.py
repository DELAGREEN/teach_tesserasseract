import cv2
import numpy as np
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBox, LTFigure
import fitz  # PyMuPDF
from pdf2image import  convert_from_path

def parse_pdf_layout(pdf_path):
    pages = list(extract_pages(pdf_path))
    return pages

def get_area(path_to_pdf, x0, y0, x1, y1):
    doc = fitz.open(path_to_pdf)
    page = doc[0]  # первая страница
    # Прямоугольник с координатами, полученными из PDFMiner
    rect = fitz.Rect(x0, y0, x1, y1)
    # Извлечение текста внутри прямоугольника
    text = page.get_text(clip=rect)
    print(text)

def visualize_pdf_blocks(pdf_path, page_number=0):
    pages = parse_pdf_layout(pdf_path)
    page = pages[page_number]

    # Создаем черный холст для отображения (будем рисовать на нем)
    # Размер холста равен размеру страницы PDF
    img = np.ones((int(page.height), int(page.width), 3), dtype=np.uint8) * 255

    pages_s = convert_from_path(pdf_path, dpi=300)
    img_s = np.array(pages_s[0])
    img_rgb_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)
    # Копия полного изображения для сохранения с обведенной областью
    full_outlined_image = img_rgb_s.copy()

    scale_x = img_rgb_s.shape[1] / page.width
    scale_y = img_rgb_s.shape[0] / page.height

    for element in page:
        if isinstance(element, LTTextBox):
            (x0, y0, x1, y1) = element.bbox
            # Преобразуем координаты
            x0_scaled = int(x0 * scale_x)
            x1_scaled = int(x1 * scale_x)
            y0_scaled = int((page.height - y0) * scale_y)
            y1_scaled = int((page.height - y1) * scale_y)

            # Рисуем прямоугольник
            cv2.rectangle(full_outlined_image, (x0_scaled, y1_scaled), (x1_scaled, y0_scaled), (0, 255, 0), 2)
            cv2.putText(full_outlined_image, 'Text', (x0_scaled, y1_scaled - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        elif isinstance(element, LTFigure):
            (x0, y0, x1, y1) = element.bbox
            x0_scaled = int(x0 * scale_x)
            x1_scaled = int(x1 * scale_x)
            y0_scaled = int((page.height - y0) * scale_y)
            y1_scaled = int((page.height - y1) * scale_y)

            get_area(pdf_path, x0, y0, x1, y1)

            cv2.rectangle(full_outlined_image, (x0_scaled, y1_scaled), (x1_scaled, y0_scaled), (255, 0, 0), 2)
            cv2.putText(full_outlined_image, 'Figure', (x0_scaled, y1_scaled - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)


    # Показываем результат с помощью OpenCV
    cv2.imshow("PDF Blocks", full_outlined_image)
    cv2.waitKey(0)  # Ждем нажатия клавиши для закрытия окна
    cv2.destroyAllWindows()

# Пример вызова:
visualize_pdf_blocks("/home/user/Desktop/Примеры чертежей/Компас/А3/РНАТ.741136.117.cdw.PDF")
