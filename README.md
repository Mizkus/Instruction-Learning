# Instruction Learning

Instruction Learning — учебный проект, где мы ускоряем инструктирование текстовых эмбеддингов: вместо повторного прогона новостей через тяжёлую модель Qwen/Qwen3-0.6B-Embedding мы учим лёгкий линейный адаптер, который переводит базовые эмбеддинги в инструктированные. Качество оцениваем через V-measure кластеризации, скорость выигрываем за счёт одномоментного предрасчёта teacher-эмбеддингов.

## Технологический стек
- **Hydra** — конфигурации (data/model/training/logging/instructions)
- **PyTorch Lightning** — обучение линейного адаптера
- **SentenceTransformers** — расчёт Qwen3 эмбеддингов по инструкции
- **DVC** — управление стадиями `download_data` → `preprocess_embeddings`
- **MLflow** — логирование метрик/артефактов (`val_v_measure_pred`, чекпоинты, latency speedup)

## Требования
- Python 3.10+
- [Poetry](https://python-poetry.org/) для управления зависимостями
- Доступ в интернет (Hugging Face dataset `BrandonZYW/NYTClustering` и модель `Qwen/Qwen3-0.6B-Embedding`)
- (Опционально) токен Hugging Face в `HF_TOKEN`, если требуется аутентификация

## Быстрый старт
1. **Клонируйте репозиторий**
   ```bash
   git clone <this-repo-url>
   cd Instruction-Learning
   ```
2. **Установите зависимости через Poetry**
   ```bash
   poetry install
   poetry run pre-commit install
   ```
3. **(Опционально) Запустите MLflow UI** в отдельном окне:
   ```bash
   poetry run mlflow ui --backend-store-uri mlruns --port 8080
   ```
4. **Соберите данные и эмбеддинги** (через DVC — включает загрузку Hugging Face датасета и предрасчёт Qwen3 эмбеддингов):
   ```bash
   poetry run python -m dvc repro
   # или поэтапно
   poetry run python -m instruction_learning.cli download-data
   poetry run python -m instruction_learning.cli preprocess
   ```
5. **Запустите обучение** (Hydra-конфиг можно переопределять флагами `+group.option=value`):
   ```bash
   poetry run python -m instruction_learning.cli train
   # пример с override: poetry run python -m instruction_learning.cli train training.trainer.max_epochs=30
   ```
6. **Результаты**
   - чекпоинт: `artifacts/checkpoints/linear-adapter.ckpt`
   - эмбеддинги: `data/embeddings/{base,instruct}`
   - автоматический тестовый прогон: после обучения запускается `trainer.test` (по лучшему чекпоинту) с логированием `test_loss`, `test_v_measure_pred` и т.д.
   - логи MLflow: `mlruns/` (по умолчанию сервер `http://127.0.0.1:8080`), туда уходят `train_loss`, `val_loss`, `val_v_measure_pred`, `val_v_measure_teacher`, `test_*`, а также `teacher_instruct_latency_sec`, `adapter_latency_sec`, `latency_speedup`

## Пайплайн данных
- `download_data`: скачивает subset `topic` датасета [BrandonZYW/NYTClustering](https://huggingface.co/datasets/BrandonZYW/NYTClustering) и сохраняет `data/raw/topic.csv`.
- `preprocess_embeddings`: чистит дубликаты/NaN, сохраняет `data/processed/topic_clean.{csv,jsonl}` и предрасчитывает эмбеддинги SentenceTransformer по инструкции `"Instruct: Identify the topic of this news article. Query: "` (база + instruct) + `labels.npy`.
- DVC конфигурация `dvc.yaml` описывает обе стадии; `dvc.lock` фиксирует артефакты.

## Структура репозитория
```
.
├── configs/                    # Hydra-конфиги (data/model/training/logging/embedding_model)
├── src/instruction_learning
│   ├── data/                   # download.py, DataModule
│   ├── models/                 # LinearAdapter LightningModule
│   ├── pipelines/              # NYTClustering preprocessing + embeddings
│   ├── cli.py                  # fire CLI (download-data, preprocess, train)
│   └── train.py                # Hydra entrypoint для обучения
├── dvc.yaml / dvc.lock         # DVC стадии данных
├── README.md                   # этот файл
├── pyproject.toml / poetry.lock # зависимости проекта
└── .pre-commit-config.yaml     # форматирование и линты
```

## Основные команды
| Цель | Команда |
| --- | --- |
| Установка зависимостей | `poetry install` |
| Запуск хуков качества | `poetry run pre-commit run -a` |
| Загрузка данных | `poetry run python -m instruction_learning.cli download-data` |
| Препроцессинг + эмбеддинги | `poetry run python -m instruction_learning.cli preprocess` |
| Обучение | `poetry run python -m instruction_learning.cli train` |
| Полный пайплайн через DVC | `poetry run python -m dvc repro` |
| MLflow UI | `poetry run mlflow ui --backend-store-uri mlruns --port 8080` |

## Примечания
- Если интерпретатор не видит GPU/CUDA, можно поставить CPU‑сборку PyTorch командой `poetry run pip install torch==<версия> --index-url https://download.pytorch.org/whl/cpu` после `poetry install`.
- Для повторного запуска достаточно обновить данные (`dvc repro`) — стадия `preprocess_embeddings` пересчитается, если менялся код/конфиг.
- Hydra складывает логи запуска в `./outputs/`; чтобы фиксировать артефакты в одном месте, используем `paths.artifacts_root`.

Теперь можно заново клонировать репозиторий и воспроизвести весь сценарий по инструкции выше.
