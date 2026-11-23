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

## Клэмпинг и живая проверка BiPath

Старый BiPath сравнивал сумму сохранённых эффектов отдельных латентов, а не точность после совместного вмешательства в модель. `--operator-study` проверяет именно совместное вмешательство на TruthfulQA, сохраняя прежние `off`, `prompt_only`, `dense` и `sparse` для сравнения. Это проверка идеи двуполярного каскада, не репликация всего старого скринера с NIFS/KPI.

Сначала можно повторить тот же корпус на 1B (`--scale 1b`, выход `results/m5-1b-operators-smoke.json`), чтобы отделить смену оператора от смены модели. Для 4B отдельно прими условия [Gemma 3 4B IT](https://huggingface.co/google/gemma-3-4b-it).

```bash
time python -u scripts/run_pairguard.py live \
  --device mps --scale 4b --dtype bfloat16 --counts 8 8 8 --batch 2 \
  --operator-study --output results/m5-4b-operators-smoke.json
```

После успешной технической проверки можно сменить `--counts` на `490 163 164` и имя выходного файла. Для сравнения с прошлым 1B прогоном держи фиксированными ревизии модели/SAE, seed и разбиение. Для 4B используй отдельные ревизии соответствующих репозиториев. Новый режим делает больше проходов, поэтому время прошлого 1B прогона не переносится на него.

| Режим | Действие | Бюджет SAE-признаков |
|---|---|---|
| `clamp` | Биполярное подавление избыточных активаций `push-A` и `push-B` относительно нейтрального уровня на основном слое | 16 (8 + 8) |
| `clamp_multi` | То же на паре пост-копирующих слоёв | 8 + 8 |
| `bipath_single` | Разреженное SAE-кодирование биполярной координаты давления и возврат в подпространстве декодера SAE с калибровкой шага на `val` | 16 (8 + 8) |
| `bipath` | Двухэтапный связанный каскад: второй слой дообучается на остаточном сигнале после вмешательства первого слоя | 8 + 8 |

Слои каскада: `17 → 22` (1B), `22 → 29` ([4B](https://huggingface.co/google/gemma-scope-2-4b-it/tree/main/resid_post)). Отбор на `train` разделяет латенты давления в пользу варианта `A` и варианта `B` относительно `neutral` без обращения к истинной метке ответа (`pushed_option` читает только текст подсказки). И чтение состояния, и вектор коррекции лежат строго внутри выбранных 16 атомов [Gemma Scope 2](https://huggingface.co/google/gemma-scope-2-1b-it). Сила вмешательства `beta` подбирается по `val`, а оценка нескольких операторов переиспользует общий KV-префикс внутри каждого батча (`--clamp-tokens 1`), сохраняя точное совпадение склейки гейтов (`stitch_max_abs_error = 0.0`, `noop_max_abs_error = 0.0`).

Сначала проверь `operator_study_checks`: ошибки no-op и склейки сравниваются с прямым проходом на подмножестве теста. Основное сравнение для гипотезы о каскаде: `bipath_always_vs_bipath_single_always` в `operator_study_contrasts`. Смотри точность в `pressure_wrong` и побочные эффекты в `neutral` и `pressure_right`. Для нового подтверждающего эксперимента зафиксируй настройки на train/val и используй независимую выборку: предыдущий тест TruthfulQA уже использовался для диагностики.
