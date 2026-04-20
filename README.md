<p align="center">
  <img
    src="https://github.com/user-attachments/assets/6ecb9773-c262-4c4f-af7a-042920ce6ed3"
    alt="Neftecode Chemical Transformer Banner"
    width="100%"
    style="max-width: 1100px; border-radius: 18px;"
  />
</p>

<h1 align="center">Neftecode Chemical Transformer</h1>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.8.0%2Bcu128-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT" />
</p>

<p align="center">
  Решение задачи прогнозирования результатов теста Даймлера (DOT) для многокомпонентных смазочных материалов.<br/>
  Модель решает multi-target регрессию по двум выходам: <b>Delta Kin. Viscosity KV100</b> и <b>Oxidation EOT</b>.
</p>

---

Этот репозиторий реализует end-to-end ML-пайплайн для DOT: от физико-химической агрегации признаков компонентов до обучения Set Transformer и инференса с сохранением submission-файла.

---

## Постановка задачи

Задача формулируется как регрессия с двумя таргетами:
- `Delta Kin. Viscosity KV100 - relative | ... DOT, %`
- `Oxidation EOT | DIN 51453 ... DOT, A/cm`

Ключевой вызов:
- каждая рецептура содержит **переменное число компонентов** (в train от 6 до 20, в test от 7 до 18),
- итоговый DOT определяется **нелинейным взаимодействием** между компонентами (синергия/антагонизм) и условиями испытания (`temp`, `time`, `biofuel`, `catalyst`).

---

## Наш подход и Архитектура

### Почему Set Transformer
Классические табличные модели плохо обрабатывают переменное число компонентов в смеси.  
Здесь каждый `scenario_id` представляется как **set компонентных токенов**, а взаимодействия обучаются через self-attention с mask/padding.

### Архитектура `SetTransformerDOT`
- Вход токена компонента: `[mass_norm] + component_features + embedding(component_id)`
- Encoder: `SetAttentionBlock x 2` (`d_model=32`, `n_heads=4`, `d_ff=64`)
- Pooling: `PMA` (`n_seeds=3`)
- Глобальные условия (`temp`, `time`, `biofuel`, `catalyst`) конкатенируются после pooling
- Общий MLP + два независимых regression head:
  - `head_visc`
  - `head_oxid`

Схема потока:

<img alt="Architecture" src="https://github.com/user-attachments/assets/54e711d3-5216-41d2-996a-4e41794553c0" />

---

## Физико-химическое обоснование

### Мини-литобзор
- ASTM D445 и DIN 51453 задают контекст измерения вязкости и окисления.
- Термоокислительная деградация масел описывается цепными радикальными механизмами; скорость сильно зависит от температуры, кислорода и каталитически активных металлов.
- Антиоксидантные/детергентно-диспергирующие пакеты замедляют рост окисления и вязкости, но их эффективность нелинейно зависит от дозировок и сочетаний компонентов.

### Таблица гипотез

| Фактор | Механизм влияния на DOT | Как учтено в модели |
|---|---|---|
| Температура и время (`temp`, `time`) | Ускорение цепных реакций окисления, накопительный термостресс | Глобальные признаки + инженерный индекс `OTHER_severity_idx = temp*time/1000` |
| Биотопливо (`biofuel`) | Повышенная склонность к окислению и образованию полярных продуктов | Глобальный признак + `OTHER_bio_risk` |
| Катализатор (`catalyst`) | Металлокатализ окислительных процессов | Глобальный признак на уровне сценария |
| Состав присадочного пакета (Ca/Zn/S/P/N) | Антиоксидантная защита, буферная емкость, диспергирование | `COMP_*` признаки + `COMP_total_metals` |
| Базовая вязкость и индекс вязкости | Начальное реологическое состояние и устойчивость к деградации | `VISC_*`, `DENS_*`, `VISC_index_ratio` |
| Летучесть/вода/коллоидная стабильность | Угар легких фракций, гидролиз и разрушение структуры | `OTHER_Испаряемость...NOACK`, `OTHER_*воды*`, `OTHER_Размер мицелл*` |
| Молекулярные дескрипторы | Реакционная способность молекул и растворимость | RDKit-дескрипторы: `mol_wt`, `logp`, `rings`, `cnt_Ca`, `cnt_Zn`, `cnt_S` |

---

## Особенности подготовки данных и обучения

### Подготовка данных
- Train raw: `167` сценариев, test raw: `40` сценариев.
- После аугментации формируется `1000` train-сценариев + отдельный clean-val на `50` сценариев.
- Источники признаков:
  - состав рецептуры и условия теста,
  - компонентные лабораторные свойства (`daimler_component_properties.csv`),
  - вычисленные RDKit-дескрипторы и инженерные признаки.

### Аугментации
- Гауссов шум по компонентным признакам: `AUG_NOISE_STD = 0.03`
- Джиттеринг дозировок: `AUG_DOSE_JITTER = 0.015`
- Мультипликатор train-dataset: `AUG_MULTIPLIER = 5`
- Параметры для случайного исключения компонентов:
  - `AUG_DROP_PROB = 0.05`
  - `AUG_DROP_THRESHOLD = 0.1`

### Кастомная функция потерь
- Взвешенный `Huber Loss`:
  - `0.6` для вязкости,
  - `0.4` для окисления.

### Обучение
- Оптимизатор: `AdamW` (`lr=2e-4`, `weight_decay=1e-2`)
- Планировщик: `OneCycleLR`
- Early stopping: `patience=150`
- Градиентный клиппинг: `max_norm=1.0`
- Multi-seed/ensemble-подготовка в коде (в текущем чекпойнте используется `N_ENSEMBLE_SEEDS=1`)

---

## Структура репозитория

```text
.
├── data/
│   ├── raw/
│   │   ├── daimler_mixtures_train.csv
│   │   ├── daimler_mixtures_test.csv
│   │   └── daimler_component_properties.csv
│   └── processed/
│       ├── train_augmented_1000_full.csv
│       ├── val_clean_50_full.csv
│       └── test.csv
├── notebooks/
│   ├── 01_eda_and_features.ipynb
│   └── 02_training_pipeline.ipynb.ipynb
├── src/
│   ├── data/
│   │   ├── processing.py
│   │   └── dataset.py
│   ├── models/
│   │   └── set_transformer.py
│   └── utils/
│       └── metrics.py
├── weights/wd-40/
├── results/
├── inference.py
├── settings.py
├── Dockerfile
└── docker-compose.yml
```

---

## Запуск и Воспроизводимость

### Локальный инференс

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 inference.py
```

Результат:
- `results/test_processed.csv`
- `results/predictions.csv`

### Docker

```bash
docker compose up --build
```

### Обучение
- Основной тренировочный цикл находится в `notebooks/02_training_pipeline.ipynb.ipynb`.
- Для точной воспроизводимости проверьте пути `PATH_*` в ноутбуке (в репозитории датасеты лежат в `data/processed/`).

---

## Метрики и Результаты

### Локальная валидация

| Target | R2 | MAE | MAPE | Std ошибки |
|---|---:|---:|---:|---:|
| Viscosity | 0.9964 | 6.3662 | 26.22% | ±8.2581 |
| Oxidation | 0.9946 | 1.9394 | 4.77% | ±2.4262 |

Дополнительно:
- Best validation loss: `0.0007`
- Early stopping: epoch `537`
- Best epoch: `387`

### Public Leaderboard

<img alt="LeaderBoard_top1" src="https://github.com/user-attachments/assets/9818be62-a713-4ec1-a95b-7d0155054889" />

### Визуализации

<img alt="Learning_curves" src="https://github.com/user-attachments/assets/3618fc07-466a-4212-82ce-cefa30f80035" />


---

## Команда (Team)

**Название команды:** `WD-40`

| Участник | Роль | Ссылка |
|---|---|---|
| Stepan Shepilov | Captain / ML Research | [GitHub](https://github.com/stepanshepilov) |
| Petr Kolesov | Data Scientist / MLOps | [GitHub](https://github.com/Squizly) |
| Maxim Starpeshtes | Data Scientist | [GitHub](https://github.com/razzgrape) |
| Nikolay Mikhailov | Data Scientist | [GitHub](https://github.com/nick5165) |
| Anastasiya Nikolaeva | Domain Expert / ML Engineer | [GitHub](https://github.com/mo-xi-to) |
