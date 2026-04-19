#!/bin/bash
set -e 

MODE=$1

if [ "$MODE" = "train" ]; then
    echo "================================================="
    echo "🚀 ЗАПУСК ПОЛНОГО ЦИКЛА: ОБУЧЕНИЕ + ИНФЕРЕНС"
    echo "================================================="
    
    export WEIGHTS_DIR="weights/from_train"
    
    # Сначала обучаем
    python train.py
    
    # Затем делаем предсказания на новых весах
    python inference.py

elif [ "$MODE" = "inference" ]; then
    echo "================================================="
    echo "🔮 ЗАПУСК ТОЛЬКО ИНФЕРЕНСА (НА ГОТОВОЙ МОДЕЛИ)"
    echo "================================================="
    
    # Путь к вашей лучшей заготовленной модели
    export WEIGHTS_DIR="weights/wd-40"
    
    # Сразу делаем предсказания
    python inference.py

else
    echo "❌ Ошибка: Неизвестный режим '$MODE'. Используйте 'train' или 'inference'."
    exit 1
fi