# OntoRAG — Ontology-Augmented Retrieval-Augmented Generation

> Улучшение Retrieval-Augmented Generation с использованием онтологического слоя и адаптивной маршрутизации запросов.  
> **Дипломная работа** — Токанов Расул

---

## Описание проекта

Данный проект реализует **модульный RAG-движок (v3.1)** для сравнительного исследования трёх режимов поиска и генерации ответов:
# OntoRAG — Ontology-Augmented Retrieval-Augmented Generation

> Улучшение Retrieval-Augmented Generation с использованием онтологического слоя и адаптивной маршрутизации запросов.  
> **Дипломная работа** — Токанов Расул

---

## Описание проекта

Данный проект реализует **модульный RAG-движок (v3.2)** для сравнительного исследования трёх режимов поиска и генерации ответов:

| Режим | Описание |
|-------|----------|
| **Standard RAG** | Базовый retrieval (TF-IDF / BM25) без онтологии |
| **Onto-RAG** | Retrieval + OWL-онтология + семантическая BFS-фильтрация + адаптивный реранкинг |
| **Adaptive RAG** | 3-сигнальный классификатор запроса → маршрутизация между Standard и Onto-RAG |

Корпус — фрагменты из учебника **"Introduction to Calculus Volume II"** (J.H. Heinbockel).  
Онтология — OWL-файл с 30+ сущностями и 50+ связями из области математического анализа.

### Ключевые результаты (v3.2)

| Конфигурация | ROUGE-L | Δ vs Baseline |
|---|---|---|
| Standard(TF-IDF) | 0.2701 | — (baseline) |
| Standard(BM25) | 0.3095 | **+14.6%** |
| EntityRAG | 0.2836 | **+5.0%** |
| Full OntoRAG | 0.2839 | **+5.1%** |
| **AdaptiveRAG** | **0.2881** | **+6.7%** |
| OntoRAG(BM25) | 0.2763 | **+2.3%** |

> Все онтологические конфигурации превосходят базовую линию Standard RAG.

---

## Структура проекта

```
code/
├── rag_engine.py                  # Основной движок: TF-IDF, BM25, OntoRAG, AdaptiveRAG (v3.2)
├── metrics.py                     # Метрики оценки: ROUGE-L, BLEU, Cosine, NDCG@5, MRR
├── calculus_corpus.py             # Корпус текстов и тестовые вопросы (18 вопросов, 4 типа)
├── calculus_ontology.owl          # OWL-онтология математического анализа
│
├── run_standard_rag.py            # Запуск Standard RAG (без онтологии)
├── run_onto_rag.py                # Запуск Onto-RAG (с онтологией)
├── run_experiment.py              # Полный эксперимент: Standard vs Onto-RAG (JSON + HTML)
├── run_ablation.py                # Абляционное исследование v3.2 (6 конфигураций)
├── run_failure_analysis.py        # Анализ ошибок
├── run_failure_analysis_revised.py # Расширенный анализ ошибок
├── run_question_type_analysis.py  # Анализ по типам вопросов
├── generate_charts.py             # Генерация графиков для диссертации (matplotlib)
│
├── results/                       # Результаты экспериментов (генерируется автоматически)
│   ├── ablation_results.txt
│   ├── experiment_results.json
│   ├── comparison.html
│   └── ...
│
├── hobbit_corpus.py               # (Архивный) Корпус по "Хоббиту" (не используется)
├── hobbit_ontology.owl            # (Архивный) Онтология по "Хоббиту" (не используется)
├── CHANGELOG.md                   # История изменений
└── .gitignore
```

---

## Требования

- **Python 3.10+** (проверено на Python 3.13)
- **Внешние зависимости отсутствуют** — основной движок использует только стандартную библиотеку Python
- **matplotlib** (опционально) — только для генерации графиков (`generate_charts.py`)

---

## Быстрый старт

### 1. Клонирование репозитория

```bash
git clone https://github.com/<username>/ontorag.git
cd ontorag
```

### 2. Запуск Standard RAG (без онтологии)

```bash
python run_standard_rag.py
```

Результаты будут выведены в консоль и сохранены в `results/standard_rag_results.txt`.

### 3. Запуск Onto-RAG (с онтологией)

```bash
python run_onto_rag.py
```

Результаты будут выведены в консоль и сохранены в `results/onto_rag_results.txt`.

### 4. Полный эксперимент (Standard vs Onto-RAG)

```bash
python run_experiment.py
```

Генерирует:
- `results/experiment_results.json` — полные результаты в JSON
- `results/comparison.html` — HTML-отчёт с визуальным сравнением

### 5. Абляционное исследование (v3.2)

```bash
python run_ablation.py
```

Анализирует влияние отдельных компонентов:
- 6 конфигураций (Standard TF-IDF/BM25, EntityRAG, Full OntoRAG, AdaptiveRAG, OntoRAG BM25)
- Sensitivity: BFS depth sweep, ontology weight sweep, classifier threshold sweep
- Qualitative analysis: wins + avoided degradations

Результат: `results/ablation_results.txt`

### 6. Генерация графиков

```bash
pip install matplotlib numpy  # одноразово
python generate_charts.py
```

Создаёт 5 графиков в директории `pics/`:
- `fig_4_1_rouge_l_comparison.png` — сравнение ROUGE-L шести конфигураций
- `fig_4_2_rouge_l_by_type.png` — ROUGE-L по типам вопросов
- `fig_4_3_ontology_weight.png` — чувствительность к весу онтологии
- `fig_4_4_threshold_sweep.png` — чувствительность к порогу классификатора
- `fig_4_5_routing_decisions.png` — решения маршрутизации AdaptiveRAG

### 7. Анализ по типам вопросов

```bash
python run_question_type_analysis.py
```

Разбивка метрик по категориям: Factual, Relationship, Reasoning, Summary.  
Результат: `results/question_type_analysis.txt`

### 8. Анализ ошибок

```bash
python run_failure_analysis_revised.py
```

Детальный анализ случаев, где Onto-RAG уступает Standard RAG, с диагностикой причин.  
Результат: `results/failure_analysis_revised.txt`

---

## Архитектура системы (v3.2)

```
┌──────────────────────────────────────────────────────────┐
│                   AdaptiveRAG (Router)                   │
│  QueryClassifier (3-signal) → route to Standard / Onto   │
│  Signals: relation_kw + entity_density + negative_kw     │
│           + graph_connectivity_bonus                     │
└──────────┬────────────────────────┬──────────────────────┘
           │                        │
    ┌──────▼──────┐          ┌──────▼───────────────────┐
    │ StandardRAG │          │        OntoRAG (v3.2)    │
    │  (baseline) │          │                          │
    └──────┬──────┘          │  OntologyGraph           │
           │                 │  GraphExpander (BFS)     │
    ┌──────▼──────┐          │    + relation-type prio  │
    │  Retrieval  │          │    + query-relevance     │
    │  Index      │          │  OntologyReranker        │
    │ (TF-IDF /   │◄─────── │    + adaptive weight     │
    │   BM25)     │          │    + query-aware overlap │
    └─────────────┘          └─────────────────────────┘
```

### Слои:

1. **Retrieval Layer** — TF-IDF или BM25 индекс (pure Python)
2. **Ontology Layer** — OWL-парсер + семантическая BFS-фильтрация графа (приоритизация связей, query-relevance)
3. **Ranking Layer** — адаптивный реранкинг (вес масштабируется по количеству сущностей, query-aware overlap)
4. **Routing Layer** — 3-сигнальный эвристический классификатор (relation keywords, entity density, negative markers, graph connectivity)

---

## Метрики оценки

| Метрика | Описание |
|---------|----------|
| **ROUGE-L** | F-мера на основе наибольшей общей подпоследовательности |
| **BLEU** | Precision по n-граммам (до 4-грамм) |
| **Cosine Similarity** | Косинусное сходство (bag-of-words) |
| **NDCG@5** | Normalized Discounted Cumulative Gain по retrieved chunks |
| **MRR** | Mean Reciprocal Rank |

---

## Пример вывода (Ablation Study v3.2)

```
================================================================================
  ABLATION STUDY v3.2 -- RAG Engine Modular Architecture
  Corpus: Introduction to Calculus Vol. II (Heinbockel)
  Questions: 18  |  Metrics: ROUGE-L, BLEU, Cosine, NDCG@5, MRR
================================================================================

  TABLE 1. Average metrics across all questions
  ----------------------------------------------------------------------------
  Configuration         ROUGE-L     BLEU   Cosine   NDCG@5      MRR
  ----------------------------------------------------------------------------
  Standard(TF-IDF)       0.2701   0.0972   0.5489   0.9435   0.9630
  Standard(BM25)         0.3095   0.1277   0.5797   0.9578   0.9630
  EntityRAG              0.2836   0.1031   0.5600   0.9448   0.9630
  Full OntoRAG           0.2839   0.1052   0.5575   0.9506   0.9630
  AdaptiveRAG            0.2881   0.1095   0.5648   0.9503   0.9630
  OntoRAG(BM25)          0.2763   0.0945   0.5355   0.9475   0.9352
  ----------------------------------------------------------------------------

  CONCLUSION:
  1. Full OntoRAG vs Standard RAG: ROUGE-L +5.1%
  2. AdaptiveRAG vs Standard RAG: ROUGE-L +6.7%
  3. AdaptiveRAG per-question: W=7 / T=8 / L=3
```

---

## Автор

**Токанов Расул** — дипломная работа, 2026

---

## Лицензия

Данный проект создан в рамках дипломной работы и предназначен для академического использования.

| Режим | Описание |
|-------|----------|
| **Standard RAG** | Базовый retrieval (TF-IDF / BM25) без онтологии |
| **Onto-RAG** | Retrieval + OWL-онтология + BFS-расширение + реранкинг |
| **Adaptive RAG** | Классификатор запроса → маршрутизация между Standard и Onto-RAG |

Корпус — фрагменты из учебника **"Introduction to Calculus Volume II"** (J.H. Heinbockel).  
Онтология — OWL-файл с 30+ сущностями и 50+ связями из области математического анализа.

---

## Структура проекта

```
code/
├── rag_engine.py                  # Основной движок: TF-IDF, BM25, OntoRAG, AdaptiveRAG
├── metrics.py                     # Метрики оценки: ROUGE-L, BLEU, Cosine, NDCG@5, MRR
├── calculus_corpus.py             # Корпус текстов и тестовые вопросы (18 вопросов, 4 типа)
├── calculus_ontology.owl          # OWL-онтология математического анализа
│
├── run_standard_rag.py            # Запуск Standard RAG (без онтологии)
├── run_onto_rag.py                # Запуск Onto-RAG (с онтологией)
├── run_experiment.py              # Полный эксперимент: Standard vs Onto-RAG (JSON + HTML)
├── run_ablation.py                # Абляционное исследование (анализ параметров)
├── run_failure_analysis.py        # Анализ ошибок
├── run_failure_analysis_revised.py # Расширенный анализ ошибок
├── run_question_type_analysis.py  # Анализ по типам вопросов
│
├── results/                       # Результаты экспериментов (генерируется автоматически)
│   ├── standard_rag_results.txt
│   ├── onto_rag_results.txt
│   ├── experiment_results.json
│   ├── comparison.html
│   └── ...
│
├── hobbit_corpus.py               # (Архивный) Корпус по "Хоббиту" (не используется)
├── hobbit_ontology.owl            # (Архивный) Онтология по "Хоббиту" (не используется)
├── CHANGELOG.md                   # История изменений
└── .gitignore
```

---

## Требования

- **Python 3.10+** (проверено на Python 3.13)
- **Внешние зависимости отсутствуют** — проект использует только стандартную библиотеку Python

---

## Быстрый старт

### 1. Клонирование репозитория

```bash
git clone https://github.com/<username>/ontorag.git
cd ontorag
```

### 2. Запуск Standard RAG (без онтологии)

```bash
python run_standard_rag.py
```

Результаты будут выведены в консоль и сохранены в `results/standard_rag_results.txt`.

### 3. Запуск Onto-RAG (с онтологией)

```bash
python run_onto_rag.py
```

Результаты будут выведены в консоль и сохранены в `results/onto_rag_results.txt`.

### 4. Полный эксперимент (Standard vs Onto-RAG)

```bash
python run_experiment.py
```

Генерирует:
- `results/experiment_results.json` — полные результаты в JSON
- `results/comparison.html` — HTML-отчёт с визуальным сравнением

### 5. Абляционное исследование

```bash
python run_ablation.py
```

Анализирует влияние отдельных компонентов (BFS depth, ontology weight, retrieval mode).  
Результат: `results/ablation_results.txt`

### 6. Анализ по типам вопросов

```bash
python run_question_type_analysis.py
```

Разбивка метрик по категориям: Factual, Relationship, Reasoning, Summary.  
Результат: `results/question_type_analysis.txt`

### 7. Анализ ошибок

```bash
python run_failure_analysis_revised.py
```

Детальный анализ случаев, где Onto-RAG уступает Standard RAG, с диагностикой причин.  
Результат: `results/failure_analysis_revised.txt`

---

## Архитектура системы

```
┌──────────────────────────────────────────────┐
│              AdaptiveRAG (Router)             │
│  QueryClassifier → route to Standard / Onto  │
└──────────┬────────────────────┬───────────────┘
           │                    │
    ┌──────▼──────┐     ┌──────▼──────────────┐
    │ StandardRAG │     │      OntoRAG        │
    │  (baseline) │     │                     │
    └──────┬──────┘     │  OntologyGraph      │
           │            │  GraphExpander (BFS) │
    ┌──────▼──────┐     │  OntologyReranker   │
    │  Retrieval  │     └──────┬──────────────┘
    │  Index      │            │
    │ (TF-IDF /   │◄───────────┘
    │   BM25)     │
    └─────────────┘
```

### Слои:

1. **Retrieval Layer** — TF-IDF или BM25 индекс (pure Python)
2. **Ontology Layer** — OWL-парсер + BFS-расширение графа
3. **Ranking Layer** — реранкинг на основе совпадения сущностей
4. **Routing Layer** — эвристический классификатор запросов

---

## Метрики оценки

| Метрика | Описание |
|---------|----------|
| **ROUGE-L** | F-мера на основе наибольшей общей подпоследовательности |
| **BLEU** | Precision по n-граммам (до 4-грамм) |
| **Cosine Similarity** | Косинусное сходство (bag-of-words) |
| **NDCG@5** | Normalized Discounted Cumulative Gain по retrieved chunks |
| **MRR** | Mean Reciprocal Rank |

---

## Пример вывода

```
======================================================================
  STANDARD RAG -- Результаты (без онтологии) / Results (no ontology)
  Корпус / Corpus: Introduction to Calculus Vol. II (Heinbockel)
======================================================================

  Корпус / Corpus: 30 фрагментов / chunks
  Вопросов / Questions: 18

----------------------------------------------------------------------
  Вопрос #1 [Фактический / Factual]
----------------------------------------------------------------------
  Вопрос:  What is the formal definition of a limit?
  Эталон:  A limit uses the epsilon-delta definition: ...

  Ответ RAG / RAG Answer:
    A limit describes the value that a function approaches ...

  Метрики / Metrics:
    ROUGE-L:  0.5217
    BLEU:     0.2834
    Cosine:   0.6412
    NDCG@5:   0.8930
    MRR:      1.0000
```
