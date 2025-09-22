FROM registry.astralinux.ru/library/astra/ubi18-python311:latest
# Добавляем репозитории AstraLinux
#RUN echo "deb https://dl.astralinux.ru/astra/stable/orel/repository orel main contrib non-free" >> /etc/apt/sources.list

# Пример установки утилит
# Обновляем пакеты и устанавливаем необходимые утилиты:
# - poppler-utils для работы с PDF
# - tesseract-ocr и tesseract-ocr-rus для OCR (русский язык)
# - libgl1-mesa-glx для libGL (необходим для OpenCV)
RUN apt-get update && \
    apt-get install -y poppler-utils \
    wget \ 
    tesseract-ocr \ 
    tesseract-ocr-rus \
    # Альтернатива для libgl1-mesa-glx 
    #libgl1-mesa-glx \
    # GLib library
    #libglib2.0-0 \
    #Для создания вирт окружения
    python3-venv \ 
#    && pip3 --upgrade \
    && rm -rf /var/lib/apt/lists/*

# Скачиваем и устанавливаем пакеты Mesa напрямую
#RUN cd /tmp && \
#    wget https://dl.astralinux.ru/astra/stable/orel/repository/pool/main/m/mesa/libgl1-mesa-glx_20.1.7-1+astra1_amd64.deb && \
#    wget https://dl.astralinux.ru/astra/stable/orel/repository/pool/main/m/mesa/libglapi-mesa_20.1.7-1+astra1_amd64.deb && \
#    wget https://dl.astralinux.ru/astra/stable/orel/repository/pool/main/m/mesa/libgl1-mesa-dri_20.1.7-1+astra1_amd64.deb && \
#    wget https://dl.astralinux.ru/astra/stable/orel/repository/pool/main/m/mesa/libglx-mesa0_20.1.7-1+astra1_amd64.deb && \
#    dpkg -i *.deb || apt-get install -f -y && \
#    rm -f *.deb

WORKDIR /app

COPY requirements.txt .

# Создаем виртуальное окружение и устанавливаем зависимости в одной команде
RUN python3 -m venv venv && \
    venv/bin/pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["venv/bin/python", "app.py"]
#CMD ["tail", "-f", "/dev/null"]