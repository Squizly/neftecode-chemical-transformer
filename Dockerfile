# Используем легковесный официальный образ Python 3.10
FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем список зависимостей и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код, настройки и скрипты
COPY src/ /app/src/
COPY settings.py /app/
COPY inference.py /app/
COPY train.py /app/

# Копируем баш-скрипт оркестрации
COPY entrypoint.sh /app/

# Делаем скрипт исполняемым и удаляем Windows-символы возврата каретки (CRLF -> LF)
RUN sed -i 's/\r$//' /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Точка входа — наш bash-скрипт
ENTRYPOINT ["/app/entrypoint.sh"]

# Команда по умолчанию (если не передать аргументов)
CMD ["inference"]