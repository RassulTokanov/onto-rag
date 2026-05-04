# OntoRAG — Ontology-Augmented Retrieval-Augmented Generation

> Улучшение Retrieval-Augmented Generation с использованием онтологического слоя и адаптивной маршрутизации запросов.  
> **Дипломная работа** — Токанов Расул

---

## Описание проекта

Данный проект реализует **модульный RAG-движок (v3.1)** для сравнительного исследования трёх режимов поиска и генерации ответов:

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

---

## Автор

**Токанов Расул** — дипломная работа, 2026

---

## Лицензия

Данный проект создан в рамках дипломной работы и предназначен для академического использования.
