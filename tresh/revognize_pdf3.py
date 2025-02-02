from pdf2image import convert_from_path
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import datetime
import re
from icecream import ic as icprint

start = datetime.datetime.now()

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(gray, -1, sharpen_kernel)
    binary = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return binary

def convert_pdf_to_text(path):
    pages = convert_from_path(path, dpi=300)
    img = np.array(pages[0])
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    binary = preprocess_image(img_rgb)

    data = pytesseract.image_to_data(binary, lang="rus", output_type=Output.DICT, config="--psm 6")
    
    boxes = [(data['left'][i], data['top'][i], data['width'][i], data['height'][i])
             for i in range(len(data['level'])) if data['text'][i].strip()]
    
    mask = np.zeros_like(binary)
    for (x, y, w, h) in boxes:
        mask[y:y+h, x:x+w] = 255
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 60))
    dilated = cv2.dilate(mask, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    largest_text_area = img_rgb[y:y+h, x:x+w]
    binary_text = preprocess_image(largest_text_area)
    
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 35))
    dilated_small = cv2.dilate(binary_text, kernel_small, iterations=1)
    contours, _ = cv2.findContours(dilated_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    grouped_text = {}
    last_number = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        text_area = largest_text_area[y:y+h, x:x+w]
        processed_text = preprocess_image(text_area)
        text = pytesseract.image_to_string(processed_text, lang="rus", config="--psm 6").strip()
        if text:
            cv2.rectangle(largest_text_area, (x, y), (x+w, y+h), (0, 255, 0), 2)
            match = re.match(r'^(\d+)\s+(.*)', text)
            match2 = re.match( r'\A-\s.+\;|.|%.', text)
            #icprint(match)
            icprint(match2)
            if match:
                number, content = match.groups()
                last_number = number
                if number in grouped_text:
                    grouped_text[number].append(text)
                else:
                    grouped_text[number] = [text]
            elif match2:
                if last_number:
                    grouped_text[last_number].append(text)
                else:
                    if "misc" in grouped_text:
                        grouped_text["misc"].append(text)
                    else:
                        grouped_text["misc"] = [text]

    for key, texts in sorted(grouped_text.items(), key=lambda x: int(x[0]) if x[0].isdigit() else float('inf'), reverse=True):
        print(f"Группа {key}:")
        print("\n".join(texts))
    
    cv2.imshow('Largest Text Area with Boxes', largest_text_area)
    cv2.waitKey(0)

path = '/home/nzxt/rep/teach_tesserasseract/pdf/РНАТ.305155.008СБ_182425323.dwg.pdf'
convert_pdf_to_text(path)

end = datetime.datetime.now()
print(end-start)
