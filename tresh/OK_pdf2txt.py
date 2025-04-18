#pip install pymupdf
import fitz  # pymupdf
import icecream

doc = fitz.open("/home/user/Desktop/Примеры чертежей/Компас/А4/РНАТ.715111.384_Палец.cdw.PDF")
all_text = ""
for page in doc:
    all_text += page.get_text()
icecream.ic(all_text)
