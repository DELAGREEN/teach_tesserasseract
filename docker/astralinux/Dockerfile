FROM registry.astralinux.ru/library/astra/ubi18-python311:latest

# Добавляем репозитории AstraLinux
RUN echo "deb https://dl.astralinux.ru/astra/stable/orel/repository orel main contrib non-free" >> /etc/apt/sources.list

# Устанавливаем необходимые утилиты
RUN apt-get update && \
    apt-get install -y poppler-utils \
    tesseract-ocr \ 
    tesseract-ocr-rus \
    libgl1-mesa-glx \
    python3-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Создаем пользователя
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Копируем requirements.txt ДО переключения пользователя
COPY requirements.txt .

# Создаем виртуальное окружение и устанавливаем зависимости как ROOT
RUN python3 -m venv venv && \
    venv/bin/python -m ensurepip --upgrade && \
    venv/bin/pip install --no-cache-dir -r requirements.txt

# Создаем файл лога (теперь у appuser есть права на запись)
RUN touch /app/log.log && chmod 644 /app/log.log

# Меняем владельца всей директории /app на appuser
RUN chown -R appuser:appuser /app

# Теперь переключаемся на непривилегированного пользователя
USER appuser

# Копируем остальные файлы с правильными правами
COPY --chown=appuser:appuser app.py .
COPY --chown=appuser:appuser logger.py .
COPY --chown=appuser:appuser config.ini .

# Запускаем приложение
#CMD ["venv/bin/python", "app.py"]
CMD ["tail", "-f", "/dev/null"]