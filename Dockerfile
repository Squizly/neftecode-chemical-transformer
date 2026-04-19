FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y libxrender1 libxext6 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/
COPY settings.py /app/
COPY inference.py /app/

COPY weights/ /app/weights/

# Исправляем команду запуска (нужно расширение .py)
CMD ["python", "inference.py"]