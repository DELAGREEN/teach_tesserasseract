FROM python:3.10-slim

# Обновляем пакеты и устанавливаем необходимые утилиты:
# - poppler-utils для работы с PDF
# - tesseract-ocr и tesseract-ocr-rus для OCR (русский язык)
# - libgl1-mesa-glx для libGL (необходим для OpenCV)
RUN apt-get update && \
    apt-get install -y poppler-utils tesseract-ocr tesseract-ocr-rus libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]
