from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBox, LTTextLine, LTChar, LTImage, LTFigure

def explore_pdf_structure(pdf_path):
    for page_number, page_layout in enumerate(extract_pages(pdf_path), start=1):
        print(f"\n=== Страница {page_number} ===")
        for element in page_layout:
            print(f"  [{type(element).__name__}]")
            
            # Если это текстовый блок, выводим подробнее
            if isinstance(element, LTTextBox):
                for text_line in element:
                    print(f"    [Line] {text_line.get_text().strip()}")
                    for char in text_line:
                        if isinstance(char, LTChar):
                            print(f"      [Char] '{char.get_text()}' шрифт={char.fontname} размер={char.size}")
            
            elif isinstance(element, LTImage):
                print(f"    [Image] name={element.name}, bbox={element.bbox}")
            
            elif isinstance(element, LTFigure):
                print(f"    [Figure] bbox={element.bbox}, содержит вложенные элементы")
            
            else:
                print(f"    [Unknown] bbox={element.bbox if hasattr(element, 'bbox') else 'нет bbox'}")

# Пример использования
explore_pdf_structure("/home/user/Desktop/Примеры чертежей/Компас/А3/РНАТ.741136.117.cdw.PDF")
