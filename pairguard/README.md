# PairGuard: пилот SAE steering

Пилот проверяет, когда вмешательство через SAE помогает Gemma выбрать верный ответ под давлением пользователя и как часто оно портит уже правильные ответы. Скрипт сравнивает селективную политику с отсутствием вмешательства, постоянным вмешательством, плотным направлением и более строгой инструкцией в промпте.

## Запуск в Cursor на Mac

Открой терминал Cursor в корне репозитория. Нужны нативный arm64 Python 3.12 и доступ к [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it). Прими условия на странице модели и войди с токеном с правом чтения в терминале; не добавляй токен в файлы проекта или чат.

```bash
cd pairguard
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.11.0 -r PAIRGUARD_REQUIREMENTS.txt
python -c 'import torch; print(torch.__version__); assert torch.backends.mps.is_available(), "MPS unavailable"'
hf auth login
```

Сначала запусти небольшой пилот на настоящих весах. Первый запуск скачает Gemma и словарь [Gemma Scope 2, layer 17](https://huggingface.co/google/gemma-scope-2-1b-it/tree/main/resid_post/layer_17_width_16k_l0_medium). Время скачивания и время расчёта лучше записывать отдельно.

```bash
time python -u scripts/run_pairguard.py live \
  --device mps --scale 1b --dtype bfloat16 --counts 8 8 8 --batch 4 \
  --output results/m5-smoke.json
```

Если MPS сообщает, что конкретная операция не поддерживает `bfloat16`, повтори с `--dtype float16` и зафиксируй этот выбор для всех сравнений. Если получишь `401` или `403`, проверь, принят ли доступ к модели для того же аккаунта, под которым выполнен `hf auth login`.

## Полный эксперимент и замер скорости

После успешного пилота запусти команды по отдельности. `live` оценивает выбор A/B по логитам первого токена; `bench` измеряет генерацию 128 токенов без вмешательства. Время `bench` не прогнозирует время `live`: ориентируйся на стадии `live_timing_s` и фактическое время команды `time` в пилоте.

```bash
time python -u scripts/run_pairguard.py live \
  --device mps --scale 1b --dtype bfloat16 --counts 490 163 164 --batch 4 \
  --output results/m5-full-1b.json
```

```bash
time python -u scripts/run_pairguard.py bench \
  --device mps --scale 1b --dtype bfloat16 --batch 1 --tokens 128 --repeats 5 \
  --output results/m5-b1.json
```

Для сравнения с RTX 5090 запусти тот же `bench` с `--device cuda` и отдельным файлом вывода. Используй те же параметры и ревизию модели (`--revision` принимает commit SHA Hugging Face). Для воспроизведения `live` отдельно зафиксируй и `--sae-revision`. Скрипт сравнения проверяет ревизию модели и совпадение входных токенов, но не хеширует сами веса.

```bash
python scripts/compare_pairguard_devices.py results/m5-b1.json results/5090-b1.json
```

## Данные и интерпретация

- **`outcome`**: точность, доли исправлений и поломок по условиям и политикам. Сначала сравни `utility` с `off`, `prompt_only` и `dense_utility` в `pressure_wrong`, затем проверь, что не ухудшились `neutral` и `pressure_right`.
- **Малый пилот**: проверка совместимости, а не доказательство превосходства метода. Для интерпретации эффекта на полном тесте смотри на парный bootstrap по вопросам.
- **CSV**: переработанный набор [TruthfulQA](https://huggingface.co/datasets/truthfulqa/truthful_qa), Stephanie Lin, Jacob Hilton, Owain Evans, [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). Добавлены варианты промптов с давлением и разбиение по вопросам. Веса моделей и результаты запусков в Git не включены.
