FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Копируем все файлы из текущей папки в контейнер
# (Сюда попадут: STransformer_078316.pth, test.csv и твой скрипт)
COPY . .

# Команда для запуска предсказаний. 
# Предполагается, что код из ноутбука ты перенес в файл inference.py
CMD ["python", "inference.py"]