# -*- coding: utf-8 -*-
"""
Standard RAG (without ontology) / Стандартный RAG (без онтологии)
— text output of results for the corpus
"Introduction to Calculus Vol. II" by J.H. Heinbockel.
"""

import os
import sys
from pathlib import Path

# ── Пути ──────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"

sys.path.insert(0, str(SCRIPT_DIR))
from calculus_corpus import get_corpus, get_questions  # noqa: E402
from rag_engine import StandardRAG                    # noqa: E402
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

    rag = StandardRAG(corpus, top_k=3)

    metric_names = ["rouge_l", "bleu", "cosine", "ndcg", "mrr"]
    totals = {m: [] for m in metric_names}

    lines = []  # для сохранения в файл

    def out(text=""):
        print(text)
        lines.append(text)

    out("=" * 70)
    out("  STANDARD RAG -- Результаты (без онтологии) / Results (no ontology)")
    out("  Корпус / Corpus: Introduction to Calculus Vol. II (Heinbockel)")
    out("=" * 70)
    out(f"\n  Корпус / Corpus: {len(corpus)} фрагментов / chunks")
    out(f"  Вопросов / Questions: {len(questions)}")
    out()

    for i, q in enumerate(questions):
        question = q["question"]
        reference = q["reference"]
        qtype = q["type"]

        result = rag.answer(question)
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

        out("-" * 70)
        out(f"  Вопрос #{i+1} [{TYPE_LABELS.get(qtype, qtype)}]")
        out("-" * 70)
        out(f"  Вопрос:  {question}")
        out(f"  Эталон:  {reference}")
        out()
        out(f"  Ответ RAG / RAG Answer:")
        # Разбиваем длинный ответ на строки по ~80 символов
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
    out("  СВОДНАЯ ТАБЛИЦА / SUMMARY -- Standard RAG (средние / averages)")
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
    txt_path = RESULTS_DIR / "standard_rag_results.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Результаты сохранены: {txt_path}")


if __name__ == "__main__":
    main()
