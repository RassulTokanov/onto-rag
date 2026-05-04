# -*- coding: utf-8 -*-
"""
Onto-RAG (with ontology) / Onto-RAG (с онтологией)
— text output of results for the corpus
"Introduction to Calculus Vol. II" by J.H. Heinbockel
with OWL ontology enhancement.
"""

import os
import sys
from pathlib import Path

# ── Пути ──────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
OWL_PATH = SCRIPT_DIR / "calculus_ontology.owl"

sys.path.insert(0, str(SCRIPT_DIR))
from calculus_corpus import get_corpus, get_questions  # noqa: E402
from rag_engine import OntoRAG                        # noqa: E402
from metrics import (rouge_l, bleu_score, cosine_similarity,  # noqa: E402
                     ndcg_score, mrr_score)


# ═══════════════════════════════════════════════════════════════
# Основная функция
# ═══════════════════════════════════════════════════════════════

TYPE_LABELS = {
    "factual": "Фактический / Factual",
    "relationship": "О связях / Relationship",
    "reasoning": "Рассуждение / Reasoning",
    "summary": "Обобщение / Summary",
}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    corpus = get_corpus()
    questions = get_questions()

    onto_rag = OntoRAG(corpus, str(OWL_PATH), top_k=3, hop_depth=2)

    metric_names = ["rouge_l", "bleu", "cosine", "ndcg", "mrr"]
    totals = {m: [] for m in metric_names}

    lines = []

    def out(text=""):
        print(text)
        lines.append(text)

    out("=" * 70)
    out("  ONTO-RAG -- Результаты (с онтологией) / Results (with ontology)")
    out("  Корпус / Corpus: Introduction to Calculus Vol. II (Heinbockel)")
    out("  Онтология / Ontology: calculus_ontology.owl")
    out("=" * 70)
    out(f"\n  Корпус / Corpus: {len(corpus)} фрагментов / chunks")
    out(f"  Вопросов / Questions: {len(questions)}")
    out(f"  Сущностей в онтологии / Entities: {len(onto_rag.ontology.labels)}")
    out(f"  Связей (рёбер) / Edges: {len(onto_rag.ontology.edges)}")
    out()

    for i, q in enumerate(questions):
        question = q["question"]
        reference = q["reference"]
        qtype = q["type"]

        result = onto_rag.answer(question)
        answer = result["answer"]

        metrics = {
            "rouge_l": rouge_l(reference, answer),
            "bleu": bleu_score(reference, answer),
            "cosine": cosine_similarity(reference, answer),
            "ndcg": ndcg_score(reference, result["retrieved_chunks"]),
            "mrr": mrr_score(reference, result["retrieved_chunks"]),
        }

        for m in metric_names:
            totals[m].append(metrics[m])

        entities_found = result.get("entities_found", [])
        entities_expanded = result.get("entities_expanded", [])
        onto_context = result.get("ontology_context", "")

        out("-" * 70)
        out(f"  Вопрос #{i+1} [{TYPE_LABELS.get(qtype, qtype)}]")
        out("-" * 70)
        out(f"  Вопрос:  {question}")
        out(f"  Эталон:  {reference}")
        out()

        # Онтологическая информация
        if entities_found:
            out(f"  Найденные сущности / Entities found: {', '.join(entities_found)}")
        if entities_expanded:
            out(f"  Расширенные сущности (BFS) / Expanded: {', '.join(entities_expanded)}")
        if onto_context:
            out(f"  Контекст из онтологии / Ontology context:")
            # Разбиваем контекст на предложения
            sentences = [s.strip() for s in onto_context.split(". ") if s.strip()]
            for s in sentences[:6]:
                s_text = s if s.endswith(".") else s + "."
                out(f"    • {s_text}")
        out()

        out(f"  Ответ Onto-RAG / Onto-RAG Answer:")
        words = answer.split()
        line = "    "
        for w in words:
            if len(line) + len(w) + 1 > 80:
                out(line)
                line = "    " + w
            else:
                line += " " + w if line.strip() else "    " + w
        if line.strip():
            out(line)
        out()
        out(f"  Метрики / Metrics:")
        out(f"    ROUGE-L:  {metrics['rouge_l']:.4f}")
        out(f"    BLEU:     {metrics['bleu']:.4f}")
        out(f"    Cosine:   {metrics['cosine']:.4f}")
        out(f"    NDCG@5:   {metrics['ndcg']:.4f}")
        out(f"    MRR:      {metrics['mrr']:.4f}")
        out()

    # Сводная таблица
    out("=" * 70)
    out("  СВОДНАЯ ТАБЛИЦА / SUMMARY -- Onto-RAG (средние / averages)")
    out("=" * 70)
    out()
    out(f"  {'Метрика/Metric':<15} {'Значение/Value':>10}")
    out(f"  {'-' * 27}")
    for m in metric_names:
        avg = sum(totals[m]) / len(totals[m])
        label = {"rouge_l": "ROUGE-L", "bleu": "BLEU", "cosine": "Cosine Sim",
                 "ndcg": "NDCG@5", "mrr": "MRR"}
        out(f"  {label[m]:<15} {avg:>10.4f}")
    out()
    out("=" * 70)
    out("  Завершено / Done.")
    out("=" * 70)

    # Сохранение в файл
    txt_path = RESULTS_DIR / "onto_rag_results.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Результаты сохранены: {txt_path}")


if __name__ == "__main__":
    main()
