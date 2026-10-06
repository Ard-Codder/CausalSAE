# Стартовый план GitHub-проекта по SAE Steering

## Научный фокус

Проект стоит строить не как демонстрацию «нашли красивый feature и усилили его», а как систематическую проверку следующего вопроса:

> **Когда SAE-признаки дают воспроизводимое, селективное и интерпретируемое причинное управление поведением LLM, и когда они уступают простым плотным направлениям или prompting?**

Это актуально, потому что результаты литературы расходятся. AxBench показал, что prompting и fine-tuning превосходят SAE steering, а Difference-in-Means особенно силён для детекции концептов. Более поздние работы показали, что результат SAE сильно зависит от выбора признаков: фильтрация по их выходному причинному эффекту дала улучшение в 2–3 раза и приблизила SAE к supervised-методам. Следовательно, полезная научная новизна лежит не в очередном единичном steering-примере, а в контролируемом сравнении **feature selection, слоя, интервенции, семейства модели и целевого поведения**.[^1][^2][^3]

## Что уже изучено

| Направление | Состояние литературы | Что остаётся исследовать | Приоритет |
|---|---|---|---|
| Refusal / safety | Уже есть SAE-steering отказа в Phi-3 Mini, где признаки усиливают или подавляют отказ через clamping[^4]. SORRY-Bench предоставляет 44 категории и 440 сбалансированных опасных инструкций, но данные gated[^5][^6]. | Отличать полезное снижение over-refusal от небезопасного снятия отказа; проверять перенос между категориями и моделями; измерять способ отказа, а не только бинарный refusal. Новая работа показывает, что отказ включает общий набор SAE-latents и длинный хвост доменных/стилевых признаков[^7]. | **Очень высокий**, но требует безопасного протокола. |
| Семантические концепты / topic steering | Это наиболее насыщенное направление: AxBench/Concept500 содержит 500 синтетически построенных концептов для нескольких слоёв Gemma-2[^8]. ContrastiveSteer уже сравнивает активации target-domain и general text и калибрует силу вмешательства[^9]. | Селективность, композиция нескольких признаков, устойчивость к перефразировкам, out-of-domain prompts и перенос между семействами. | Высокий как основной benchmark, средний по новизне. |
| Тональность и эмоции | Плотные activation/style vectors подробно исследованы на sentiment, эмоциях и writing style; применялись IMDb и GoEmotions[^10][^11]. Human evaluation подтверждает, что умеренное steering может усиливать пять из шести базовых эмоций, но эффект зависит от эмоции[^12]. | Прямое сравнение SAE-feature steering с DiffMean/ActAdd на одинаковых моделях, human/LLM evaluation и измерение content preservation. | **Высокий**: безопасный и удобный первый эксперимент. |
| Стиль генерации | Есть исследования dense style vectors для sentiment, emotion и modern-versus-Shakespearean style[^11], а AxBench позволяет генерировать concept-specific задачи[^8]. | Fine-grained стили: краткость, формальность, уверенность, структура; отделение стиля от содержания; совместное управление несколькими атрибутами. | **Высокий** для курсового проекта. |
| Instruction following | SAIF связывает SAE-latents с instruction following и сообщает причинный steering; важны точная идентификация feature, финальный слой и позиция инструкции[^13]. | Generalization на невидимые инструкции, разные шаблоны чата и сравнение base/instruct моделей. | Средне-высокий. |
| Hallucination / knowledge awareness | SAE-направления entity recognition причинно влияют на отказ и галлюцинацию: вмешательство может заставить модель отказаться для известной сущности или галлюцинировать для неизвестной[^14]. | Надёжно улучшать фактичность сложнее, чем менять склонность отвечать: steering не добавляет отсутствующие знания, поэтому нужны known/unknown controls и строгая проверка фактов[^15]. | **Очень высокая научная ценность**, но сложнее оценивание. |
| Язык генерации / multilingual | Изменение одного SAE-feature в Gemma-2B/9B давало управляемый переход к целевому языку с сохранением семантики[^16]. Более новые результаты подчёркивают проблему англоязычных SAE и эвристического выбора слоя[^17]. | Русский язык, cross-lingual feature stability, SAE на multilingual data, предсказуемый выбор слоя. | **Лучший кандидат на дополнительную новизну**, особенно для русскоязычного проекта. |
| Cross-model transfer | Публичные SAE существуют для Gemma, Llama, Qwen, Pythia и других моделей[^18]. Но cross-architecture transfer пока ограничен и ухудшается при сильном различии архитектур и словарей[^19]. | Общая процедура сопоставления признаков и проверка, переносится ли причинный эффект, а не только корреляция активаций. | **Очень высокий**, но PRO-уровень. |

## Рекомендуемая постановка

На год лучше взять **один центральный benchmark и два углубления**, а не пытаться одинаково глубоко охватить все поведения.

### Центральная гипотеза

SAE-признаки обеспечивают более интерпретируемое и селективное управление, чем плотные steering-векторы, **только при причинно ориентированном feature selection и корректной калибровке силы вмешательства**.

### Экспериментальные направления

1. **Основной безопасный benchmark:** sentiment + emotion + style + semantic concepts.
2. **Safety case study:** refusal и over-refusal без публикации инструкций, облегчающих обход защиты.
3. **Научное углубление:** перенос признаков между Gemma и Llama/Qwen либо multilingual steering с английского на русский.

Такой дизайн одновременно покрывает требования темы и создаёт проверяемый research contribution. Он также отвечает на противоречие между отрицательным результатом AxBench и последующими работами, где грамотный выбор output-causal features резко повышает качество SAE steering.[^2][^3][^1]

## Модели и SAE

| Роль | Модель / SAE | Почему подходит | Ограничение |
|---|---|---|---|
| Быстрый smoke test | GPT-2 Small + SAELens SAE | Минимальные требования к GPU; SAELens публикует множество готовых SAE[^18]. | Слабо переносится на современные instruction-tuned LLM. |
| Основная модель | Gemma 2 2B/9B или Gemma 3 1B/4B + Gemma Scope | Gemma Scope покрывает слои Gemma и имеет открытые веса SAE; оригинальный Gemma Scope содержит более 400 SAE и свыше 30 млн признаков[^20][^21]. Gemma Scope 2 добавляет SAE и transcoders для всех слоёв Gemma 3[^22]. | Лицензия базовой Gemma отличается от CC-BY лицензии SAE[^21]. |
| Cross-family | Llama 3.1 8B + Llama Scope | Llama Scope публикует 256 TopK SAE по слоям и подслоям, шириной 32K/128K[^23][^24]. | Основной Llama Scope обучен на base-модели; steering instruct-версии потребует отдельной проверки совместимости. |
| Третий кандидат | Qwen 2.5 7B Instruct или Qwen 3 1.7B/4B | В реестре SAELens доступны публичные SAE для Qwen 2.5 и нескольких размеров Qwen 3[^18]. | Покрытие SAE и качество auto-interpretation могут быть менее равномерными, чем у Gemma Scope. |
| Feature browser | Neuronpedia | Открытая платформа для поиска, просмотра и steering признаков; предоставляет API и метаданные[^25][^26]. | API и список поддерживаемых steering-моделей изменяются; эксперимент должен уметь работать локально. |

**Практический выбор:** начать с **Gemma 2 2B-IT или Gemma 3 1B/4B-IT**, затем подтвердить ключевой результат на **Llama 3.1 8B-Instruct**. Полное сравнение трёх семейств следует оставить как stretch goal, поскольку сетка `model × layer × feature × strength × behavior × seed` быстро становится вычислительно дорогой.

## Методы сравнения

Нельзя сравнивать SAE только с unsteered моделью. Минимальный набор baselines:

- **No steering** — исходная модель.
- **Prompt steering** — явная инструкция целевого поведения; AxBench показывает, что это сильнейший baseline.[^2]
- **DiffMean / CAA / ActAdd** — разность средних плотных активаций по контрастным примерам; ActAdd строит направление из пар вроде `Love` и `Hate`.[^10]
- **Linear probe direction** — supervised feature ranking или направление классификатора.
- **Random SAE features** — контроль ложноположительного эффекта вмешательства.
- **SAE top activation difference** — признаки с максимальной разницей средних.
- **SAE classifier selection** — L1-logistic regression по SAE-активациям.
- **SAE auto-interp selection** — фильтрация по объяснениям feature.
- **Output-causal SAE selection** — выбирать признаки по фактическому влиянию decoder direction на выход; литература показывает, что input-relevant features могут быть плохими steering features.[^1]

## Интервенции

Следует реализовать единый интерфейс минимум для четырёх вариантов:

- `add`: добавление `alpha * decoder_direction` в residual stream;
- `clamp`: установка SAE activation в заданное значение;
- `ablate`: обнуление выбранного feature;
- `reconstruct_edit`: encode → edit sparse code → decode → вернуть изменённую реконструкцию.

TransformerLens предоставляет именованные hooks, которые позволяют читать, заменять и аблировать промежуточные активации без изменения кода модели. SAELens умеет загружать готовые SAE и работает не только с TransformerLens, но также с Hugging Face Transformers и NNsight.[^27][^28][^29]

## Метрики

Проект должен разделять **успешность управления**, **селективность** и **сохранение полезности**.

| Группа | Метрики |
|---|---|
| Target behavior | Accuracy/F1 внешнего классификатора, concept score, refusal/fulfillment rate, language ID, style score |
| Dose–response | Эффект как функция `alpha`, монотонность, минимальная эффективная сила, saturation point |
| Content preservation | Semantic similarity исходного и steered ответа, task accuracy, factual consistency |
| Fluency / quality | Perplexity или log-probability, repetition rate, judge score, доля вырожденных генераций |
| Selectivity | Побочные изменения по нецелевым классификаторам; отношение target effect к utility loss |
| Causality | Suppression и amplification в противоположных направлениях; placebo/random feature; feature swap; held-out prompts |
| Stability | Seed variance, prompt paraphrases, layer variance, модель/семейство, base vs instruct |
| Interpretability | AutoInterp score, human agreement, совпадение словесного описания feature с реальным causal effect |

SAEBench важен как источник методов оценки самих SAE: он объединяет восемь тестов на reconstruction, interpretability, concept detection и disentanglement и показывает, что улучшение proxy-метрик не гарантирует улучшения практических задач. Это аргумент в пользу отдельной оценки качества SAE и качества steering, а не смешивания их в один score.[^30]

## Данные и benchmark-и

| Задача | Источник | Доступ | Применение |
|---|---|---|---|
| Общие концепты | [AxBench / Concept500](https://github.com/stanfordnlp/axbench) | Открытые код и данные | Основной массовый тест feature selection и steering; Concept500 построен для 500 концептов[^8]. |
| Refusal | [HarmBench](https://github.com/centerforaisafety/HarmBench) | Открытый framework | 510 harmful behaviors, canonical validation/test split и отдельные classifiers[^31]. Использовать только в контролируемой оценке. |
| Fine-grained refusal | [SORRY-Bench](https://github.com/SORRY-Bench/sorry-bench) | По заявке | 44 категории, 440 unsafe prompts, готовый evaluator и human judgments[^32]. |
| Emotion | [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) | Открытый | Контрастные наборы для шести эмоций или 27 исходных категорий; датасет содержит около 58 тыс. размеченных комментариев[^12]. |
| Sentiment | IMDb | Открытый | Контраст positive/negative и оценка polarity; применялся в activation steering[^10]. |
| Toxicity | RealToxicityPrompts | Открытый | Проверка detoxification и side effects; применялся в ActAdd[^10]. |
| SAE quality | [SAEBench](https://github.com/adamkarvonen/SAEBench) | Открытый | Проверка SAE по нескольким независимым критериям и сравнение архитектур[^33][^30]. |
| Feature labels / inspection | [Neuronpedia](https://www.neuronpedia.org/) | Открытый/API | Поиск top activations, auto-interp labels, быстрый ручной аудит и steering[^25]. |

## Примерный план проекта

План рассчитан на последовательный переход от воспроизводимого базового эксперимента к полноценному сравнительному исследованию. На ранних этапах достаточно одной небольшой модели и одного типа поведения; расширение на другие модели и свойства выполняется только после фиксации корректного протокола оценки.

### Чекпойнт 1 — старт

- Зафиксировать research question, primary hypothesis и критерии успеха.
- Создать репозиторий, окружение, CI, pre-commit, конфиги и data/model manifests.
- Выбрать основной стек: PyTorch, Transformers, TransformerLens, SAELens, Hydra, W&B или MLflow.
- Сделать smoke test: загрузить одну небольшую модель и SAE, извлечь residual activation и воспроизвести один steering feature.
- Оформить `README`, `research_plan.md`, `experiment_protocol.md`, `risk_register.md`.

### Чекпойнт 2 — EDA

- Исследовать распределения SAE activation: частота, среднее, max, sparsity, dead features.
- Построить contrastive prompt sets и проверить leakage.
- Сравнить слои по separability целевого поведения.
- Найти кандидатов feature четырьмя способами: mean difference, sparse probe, auto-interp, output-causal score.
- Зафиксировать train/validation/test split до steering sweeps.

### Чекпойнт 3 — первые модели

- Реализовать no-steering, prompt, DiffMean/CAA и SAE baselines.
- Провести малую сетку `layer × feature × alpha × seed`.
- Построить dose–response curves и Pareto frontier `target effect vs utility loss`.
- Проверить suppression/amplification symmetry, random-feature placebo и held-out generalization.
- Выбрать одно основное поведение и одно дополнительное для большого эксперимента.

### Чекпойнт 4 — ML-сервис

Единый FastAPI-сервис может предоставлять:

- `POST /generate` — обычная и steered генерация;
- `POST /features/search` — поиск связанных features по contrastive prompts;
- `POST /features/inspect` — статистика и auto-interpretation признака;
- `POST /steer` — model, layer, feature IDs, method, strength, seed;
- `POST /evaluate` — target/quality/selectivity metrics;
- `GET /experiments/{id}` — конфиг, revisions, metrics и artifacts.

Сервис должен запрещать небезопасные presets по умолчанию, валидировать диапазон steering strength и логировать точную версию модели/SAE. Neuronpedia можно использовать как эталон UX: платформа позволяет искать признаки и менять их силу при steering.[^34][^25]

## Минимальный эксперимент

Первый воспроизводимый результат можно получить так:

1. Загрузить Gemma 2 2B-IT и совместимый SAE.
2. Взять 200–500 positive/negative или joy/sadness contrastive examples.
3. На нескольких слоях вычислить mean activation difference для SAE-latents.
4. Отобрать top-20 features, затем проверить каждый feature причинным intervention на validation prompts.
5. На test prompts сравнить: unsteered, prompt, DiffMean, top SAE, SAE ensemble и random SAE.
6. Для каждого метода сделать sweep силы и 3–5 seeds.
7. Измерить target classifier score, semantic similarity, repetition, perplexity/log-probability и unrelated behavior drift.
8. Подтвердить лучший протокол на Llama 3.1 8B или Qwen.

Ключевой исследовательский элемент — **nested selection**: test set нельзя использовать ни для выбора feature, ни для выбора слоя, ни для настройки `alpha`. Иначе причинный эффект будет завышен из-за множественного перебора.

## Новизна проекта

Наиболее реалистичные варианты вклада:

- **Causal feature selection benchmark:** сравнить correlation-based, probe-based, auto-interp и output-causal selection в одинаковом протоколе.
- **Cross-family causal stability:** проверить, совпадают ли семантическое описание и знак причинного эффекта в Gemma, Llama и Qwen.
- **Russian multilingual steering:** найти English/Russian features, проверить перенос через слои и модели, оценить semantic preservation.
- **Refusal decomposition:** разделить общий refusal control и category/style-specific latents, измеряя safety и over-refusal одновременно.
- **Multi-objective steering:** подобрать минимальный набор SAE-features, который меняет целевое поведение при ограничении на quality degradation.

Самый сильный и выполнимый вариант: **causal feature selection + cross-model validation**, а multilingual Russian steering оставить вторым вкладом. Он напрямую отвечает на установленную проблему: SAE steering может быть слабым при наивном выборе features, но существенно улучшается при выборе признаков по их выходному причинному действию.[^3][^1][^2]

## Риски

- **Корреляция вместо причинности.** Высокая разница активаций не означает, что feature управляет выходом; нужен intervention-based validation.
- **Oversteering.** Большие значения `alpha` вызывают повторения, потерю связности и off-target concepts; необходим sweep и quality constraints. Decaying steering предложен как способ повысить стабильность.[^35]
- **Feature splitting и absorption.** Один концепт может распределяться по многим latents, а один latent — смешивать концепты; SAEBench показывает необходимость многомерной оценки SAE.[^30]
- **Winner’s curse.** Перебор тысяч features по малому validation set создаёт ложные открытия; нужны held-out test, FDR/bootstrap и preregistered primary metric.
- **Model–SAE mismatch.** SAE от base-модели может не сохранять свойства после instruction tuning; это нужно тестировать, а не предполагать.
- **Judge bias.** LLM-as-a-judge может предпочитать определённый стиль; нужны независимые classifiers, rule metrics и ручная проверка подвыборки.
- **Safety.** Работа с refusal не должна превращаться в инструмент снятия защиты. Публиковать агрегированные метрики и защитные выводы, а не готовые harmful feature presets.

## Полезные источники

### Инструменты

- [SAELens](https://github.com/decoderesearch/SAELens) — обучение, загрузка и анализ SAE; поддерживает TransformerLens, Hugging Face и NNsight.[^29]
- [TransformerLens](https://transformerlensorg.github.io/TransformerLens/) — hooks, cache и activation patching.[^28][^27]
- [Neuronpedia](https://www.neuronpedia.org/) — поиск, визуализация, auto-interpretation и steering features.[^25]
- [Gemma Scope](https://deepmind.google/models/gemma/gemma-scope/) — открытые SAE/инструменты для Gemma.[^36][^22]
- [Llama Scope](https://huggingface.co/OpenMOSS-Team/Llama-Scope) — SAE по слоям Llama 3.1 8B.[^23]
- [SAEBench](https://github.com/adamkarvonen/SAEBench) — стандартизированная оценка SAE.[^33][^30]

### Основные статьи

- **Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2** — открытый набор SAE по слоям и подслоям Gemma 2.[^21]
- **Steering Language Model Refusal with Sparse Autoencoders** — прямой precedent для refusal steering.[^4]
- **AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders** — обязательный отрицательный baseline и benchmark.[^2]
- **SAEs Are Good for Steering — If You Select the Right Features** — различие input и output features и причинно ориентированный selection.[^3][^1]
- **Calibrating Lightweight SAE Feature Steering** — contrastive feature scoring и калибровка силы.[^9]
- **Steering LLM Activations in Sparse Spaces** — contrastive prompt pairing, reinforcement и suppression behaviors.[^37]
- **SAEBench** — оценка SAE по восьми различным критериям и критика proxy-only evaluation.[^30]
- **Causal Language Control via Sparse Feature Steering** — layer-wise multilingual case study.[^16]
- **Do I Know This Entity?** — причинная связь SAE features с entity knowledge, refusal и hallucination.[^14]

---

## References

1. [SAEs Are Good for Steering – If You Select the Right ...](https://arxiv.org/html/2505.20063v1) - Sparse Autoencoders (SAEs) have been proposed as an unsupervised approach to learn a decomposition o...

2. [AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders](https://www.arxiv.org/abs/2501.17148) - Fine-grained steering of language model outputs is essential for safety and reliability. Prompting a...

3. [4.2 Qualitative Results](https://ar5iv.labs.arxiv.org/html/2505.20063) - Sparse Autoencoders (SAEs) have been proposed as an unsupervised approach to learn a decomposition o...

4. [Steering Language Model Refusal with Sparse Autoencoders - arXiv](https://arxiv.org/html/2411.11296v1)

5. [SORRY-Bench](https://sorry-bench.github.io/) - We introduce SORRY-Bench to systematically evaluate LLM safety refusal behaviors, in a balanced, gra...

6. [SORRY-Bench: Systematically Evaluating Large Language ...](https://arxiv.org/html/2406.14598v2)

7. [There Is More to Refusal in Large Language Models than a ...](https://arxiv.org/html/2602.02132v1)

8. [AxBench: Steering LLMs? Even Simple BaselinesOutperform ... - arXiv](https://arxiv.org/html/2501.17148v3)

9. [Calibrating Lightweight Sparse Autoencoder Feature Steering](https://arxiv.org/html/2506.12576v3)

10. [Steering Language Models With Activation Engineering](https://arxiv.org/html/2308.10248v5)

11. [Style Vectors for Steering Generative Large Language ...](https://arxiv.org/html/2402.01618v1) - This research explores strategies for steering the output of large language models (LLMs) towards sp...

12. [The Effectiveness of Style Vectors for Steering Large Language Models: A Human Evaluation](https://arxiv.org/html/2601.21505v1)

13. [SAIF: A Sparse Autoencoder Framework for Interpreting ...](https://arxiv.org/abs/2502.11356) - Автор: Z He · 2025 · Цитируется: 31 — This paper presents a novel framework that leverages sparse au...

14. [Do I Know This Entity? Knowledge Awareness and ...](https://arxiv.org/html/2411.14257v1)

15. [Disentangling Deception and Hallucination Failures in LLMs](https://arxiv.org/html/2602.14529)

16. [Causal Language Control in Multilingual Transformers via ...](https://arxiv.org/html/2507.13410)

17. [Multilingual Sparse Autoencoders and Principled Layer ...](https://arxiv.org/html/2605.23036v1) - SAE-Based Activation and Language Steering. Sparse autoencoders (SAEs) have been widely used to inte...

18. [Pretrained SAEs - SAE Lens](https://decoderesearch.github.io/SAELens/v6.45.3/pretrained_saes/) - Docs for Sparse Autoencoder Training and Analysis Library

19. [Activation Space Interventions Can Be Transferred Between Large ...](https://arxiv.org/html/2503.04429v4)

20. [Gemma Scope: helping the safety community shed light on the inner workings of language models](https://deepmind.google/blog/gemma-scope-helping-the-safety-community-shed-light-on-the-inner-workings-of-language-models/) - Announcing a comprehensive, open suite of sparse autoencoders for language model interpretability.

21. [Gemma Scope: Open Sparse Autoencoders Everywhere ...](https://arxiv.org/html/2408.05147v2)

22. [Gemma Scope | Google AI for Developers](https://ai.google.dev/gemma/docs/gemma_scope)

23. [OpenMOSS-Team/Llama-Scope](https://huggingface.co/OpenMOSS-Team/Llama-Scope) - We introduce a suite of 256 improved TopK SAEs, trained on each layer and sublayer of the Llama-3.1-...

24. [Extracting Millions of Features from Llama-3.1-8B with ...](https://huggingface.co/papers/2410.20526) - Llama-3.1-8B-Base model to extract sparse representations, assessing their generalizability and anal...

25. [Neuronpedia](https://www.neuronpedia.org/) - Neuronpedia is an open source interpretability platform. Explore, visualize, and steer the internals...

26. [Neuronpedia Docs: Introduction](https://docs.neuronpedia.org/) - ⚠️ Warning: These docs are in the process of being updated - the latest significant revision was Sep...

27. [The Hook System - TransformerLens Documentation](https://transformerlensorg.github.io/TransformerLens/content/hook_system.html)

28. [TransformerLens Documentation](https://transformerlensorg.github.io/TransformerLens/)

29. [decoderesearch/SAELens: Training Sparse Autoencoders ...](https://github.com/decoderesearch/SAELens) - Download and Analyse pre-trained sparse autoencoders. · Train your own sparse autoencoders. · Genera...

30. [SAEBench: A Comprehensive Benchmark for Sparse Autoencoders ...](https://arxiv.org/html/2503.09532v3)

31. [HarmBench: A Standardized Evaluation Framework ...](https://arxiv.org/html/2402.04249v2)

32. [Benchmark evaluation code for "SORRY-Bench ...](https://github.com/SORRY-Bench/sorry-bench) - This repo contains code to conveniently benchmark LLM safety refusal behaviors in a balanced, granul...

33. [adamkarvonen/SAEBench](https://github.com/adamkarvonen/SAEBench) - Contribute to adamkarvonen/SAEBench development by creating an account on GitHub.

34. [Steering Using SAE Features](https://docs.neuronpedia.org/steering) - Steering Using SAE Features. Neuronpedia supports steering the output of models by increasing or dec...

35. [A Comparative Analysis of Sparse Autoencoder and ...](https://arxiv.org/html/2510.01246v1) - Sparse autoencoders (SAEs) have recently emerged as a powerful tool for language model steering. Pri...

36. [Gemma Scope](https://deepmind.google/models/gemma/gemma-scope/) - Gemma Scope is a set of interpretability tools built to help researchers understand the inner workin...

37. [Steering Large Language Model Activations in Sparse Spaces](https://arxiv.org/html/2503.00177v1)

