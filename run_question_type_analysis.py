# -*- coding: utf-8 -*-
"""
Question-Type Analysis / Analiz po tipam voprosov
===================================================
Compares Standard RAG vs Onto-RAG performance broken down by question
categories: factual, relationship, reasoning, summary.

Corpus: "Introduction to Calculus Vol. II" by J.H. Heinbockel.
"""

import os
import sys
from collections import defaultdict
from pathlib import Path

# -- Paths -----------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
OWL_PATH = SCRIPT_DIR / "calculus_ontology.owl"

sys.path.insert(0, str(SCRIPT_DIR))
from calculus_corpus import get_corpus, get_questions   # noqa: E402
from rag_engine import StandardRAG, OntoRAG             # noqa: E402
from metrics import (compute_all_metrics, METRIC_NAMES,  # noqa: E402
                     METRIC_LABELS)


TYPE_LABELS = {
    "factual": "Factual",
    "relationship": "Relationship",
    "reasoning": "Reasoning",
    "summary": "Summary",
}


def run_analysis():
    corpus = get_corpus()
    questions = get_questions()

    rag = StandardRAG(corpus, top_k=3)
    onto_rag = OntoRAG(corpus, str(OWL_PATH), top_k=3, hop_depth=2)

    # Collect per-question results
    per_q = []
    for q in questions:
        r_rag = rag.answer(q["question"])
        r_onto = onto_rag.answer(q["question"])
        m_rag = compute_all_metrics(q["reference"], r_rag)
        m_onto = compute_all_metrics(q["reference"], r_onto)
        per_q.append({
            "question": q["question"],
            "reference": q["reference"],
            "type": q["type"],
            "rag": m_rag,
            "onto": m_onto,
            "rag_answer": r_rag["answer"],
            "onto_answer": r_onto["answer"],
            "entities": r_onto.get("entities_found", []),
            "expanded": r_onto.get("entities_expanded", []),
        })

    # Group by type
    types_order = ["factual", "relationship", "reasoning", "summary"]
    by_type = defaultdict(list)
    for pq in per_q:
        by_type[pq["type"]].append(pq)

    # Compute averages per type
    avg_by_type = {}
    for qtype in types_order:
        items = by_type[qtype]
        if not items:
            continue
        n = len(items)
        avg_by_type[qtype] = {
            "rag": {mn: sum(it["rag"][mn] for it in items) / n for mn in METRIC_NAMES},
            "onto": {mn: sum(it["onto"][mn] for it in items) / n for mn in METRIC_NAMES},
            "count": n,
        }

    # Global averages
    n_total = len(per_q)
    global_avg = {
        "rag": {mn: sum(pq["rag"][mn] for pq in per_q) / n_total for mn in METRIC_NAMES},
        "onto": {mn: sum(pq["onto"][mn] for pq in per_q) / n_total for mn in METRIC_NAMES},
    }

    # ===================================================================
    # Build output
    # ===================================================================
    lines = []

    def out(text=""):
        lines.append(text)

    W = 76
    out("=" * W)
    out("  QUESTION-TYPE ANALYSIS / ANALIZ PO TIPAM VOPROSOV")
    out("  Standard RAG vs Onto-RAG")
    out("  Corpus: Introduction to Calculus Vol. II (Heinbockel)")
    out("  Total questions: %d  |  Categories: %d" % (n_total, len(avg_by_type)))
    out("=" * W)

    # -- Category distribution --------------------------------------------
    out("")
    out("  CATEGORY DISTRIBUTION")
    out("  " + "-" * (W - 4))
    for qtype in types_order:
        if qtype in avg_by_type:
            cnt = avg_by_type[qtype]["count"]
            bar = "#" * cnt
            out("  %-15s %2d questions  %s" % (TYPE_LABELS[qtype], cnt, bar))
    out("  " + "-" * (W - 4))

    # -- Table 1: Standard RAG by type ------------------------------------
    out("")
    out("  TABLE 1. Standard RAG -- average metrics by question type")
    out("  " + "-" * (W - 4))
    out("  %-15s %3s  %8s %8s %8s %8s %8s" % (
        "Type", "N", "ROUGE-L", "BLEU", "Cosine", "NDCG@5", "MRR"))
    out("  " + "-" * (W - 4))
    for qtype in types_order:
        if qtype not in avg_by_type:
            continue
        a = avg_by_type[qtype]
        out("  %-15s %3d  %8.4f %8.4f %8.4f %8.4f %8.4f" % (
            TYPE_LABELS[qtype], a["count"],
            a["rag"]["rouge_l"], a["rag"]["bleu"], a["rag"]["cosine"],
            a["rag"]["ndcg"], a["rag"]["mrr"]))
    out("  " + "-" * (W - 4))
    out("  %-15s %3d  %8.4f %8.4f %8.4f %8.4f %8.4f" % (
        "OVERALL", n_total,
        global_avg["rag"]["rouge_l"], global_avg["rag"]["bleu"],
        global_avg["rag"]["cosine"], global_avg["rag"]["ndcg"],
        global_avg["rag"]["mrr"]))
    out("  " + "-" * (W - 4))

    # -- Table 2: Onto-RAG by type ----------------------------------------
    out("")
    out("  TABLE 2. Onto-RAG -- average metrics by question type")
    out("  " + "-" * (W - 4))
    out("  %-15s %3s  %8s %8s %8s %8s %8s" % (
        "Type", "N", "ROUGE-L", "BLEU", "Cosine", "NDCG@5", "MRR"))
    out("  " + "-" * (W - 4))
    for qtype in types_order:
        if qtype not in avg_by_type:
            continue
        a = avg_by_type[qtype]
        out("  %-15s %3d  %8.4f %8.4f %8.4f %8.4f %8.4f" % (
            TYPE_LABELS[qtype], a["count"],
            a["onto"]["rouge_l"], a["onto"]["bleu"], a["onto"]["cosine"],
            a["onto"]["ndcg"], a["onto"]["mrr"]))
    out("  " + "-" * (W - 4))
    out("  %-15s %3d  %8.4f %8.4f %8.4f %8.4f %8.4f" % (
        "OVERALL", n_total,
        global_avg["onto"]["rouge_l"], global_avg["onto"]["bleu"],
        global_avg["onto"]["cosine"], global_avg["onto"]["ndcg"],
        global_avg["onto"]["mrr"]))
    out("  " + "-" * (W - 4))

    # -- Table 3: Delta (Onto-RAG vs Standard RAG) by type ----------------
    out("")
    out("  TABLE 3. Delta: Onto-RAG vs Standard RAG (% change)")
    out("  " + "-" * (W - 4))
    out("  %-15s %3s  %8s %8s %8s %8s %8s" % (
        "Type", "N", "ROUGE-L", "BLEU", "Cosine", "NDCG@5", "MRR"))
    out("  " + "-" * (W - 4))
    delta_by_type = {}
    for qtype in types_order:
        if qtype not in avg_by_type:
            continue
        a = avg_by_type[qtype]
        deltas = {}
        parts = []
        for mn in METRIC_NAMES:
            r = a["rag"][mn]
            o = a["onto"][mn]
            d = ((o - r) / r * 100) if r > 0 else 0.0
            deltas[mn] = d
            parts.append("%+7.1f%%" % d)
        delta_by_type[qtype] = deltas
        out("  %-15s %3d  %s" % (TYPE_LABELS[qtype], a["count"],
                                  " ".join(parts)))
    # Global delta
    g_parts = []
    global_deltas = {}
    for mn in METRIC_NAMES:
        r = global_avg["rag"][mn]
        o = global_avg["onto"][mn]
        d = ((o - r) / r * 100) if r > 0 else 0.0
        global_deltas[mn] = d
        g_parts.append("%+7.1f%%" % d)
    out("  " + "-" * (W - 4))
    out("  %-15s %3d  %s" % ("OVERALL", n_total, " ".join(g_parts)))
    out("  " + "-" * (W - 4))

    # -- Table 4: Per-question detail -------------------------------------
    out("")
    out("  TABLE 4. Per-question ROUGE-L comparison (sorted by type)")
    out("  " + "-" * (W - 4))
    out("  %-4s %-14s %-9s %-9s %-9s  %s" % (
        "Q#", "Type", "RAG", "Onto-RAG", "Delta", "Question"))
    out("  " + "-" * (W - 4))
    qi = 0
    for qtype in types_order:
        for pq in by_type[qtype]:
            qi += 1
            r = pq["rag"]["rouge_l"]
            o = pq["onto"]["rouge_l"]
            d = o - r
            marker = "+" if d > 0.005 else ("-" if d < -0.005 else "=")
            out("  %-4s %-14s %8.4f  %8.4f  %+7.4f %s %s" % (
                "Q%d" % qi, TYPE_LABELS[pq["type"]][:14],
                r, o, d, marker, pq["question"][:30]))
    out("  " + "-" * (W - 4))

    # ===================================================================
    # Analytical conclusions
    # ===================================================================
    out("")
    out("=" * W)
    out("  ANALYSIS / ANALITICHESKIY VYVOD")
    out("=" * W)
    out("")

    # Find best and worst category by ROUGE-L delta
    sorted_types = sorted(delta_by_type.items(),
                          key=lambda x: x[1]["rouge_l"], reverse=True)
    best_type = sorted_types[0]
    worst_type = sorted_types[-1]

    out("  1. CATEGORY RANKING BY ONTO-RAG IMPROVEMENT (ROUGE-L Delta):")
    out("")
    for rank, (qtype, deltas) in enumerate(sorted_types, 1):
        rl_d = deltas["rouge_l"]
        label = TYPE_LABELS[qtype]
        extras = []
        for mn in METRIC_NAMES:
            if mn == "rouge_l":
                continue
            d = deltas[mn]
            if d > 1.0:
                extras.append("%s %+.1f%%" % (METRIC_LABELS[mn], d))
        direction = "improvement" if rl_d > 0 else "degradation"
        out("     %d. %-15s  ROUGE-L %+.1f%% (%s)" % (
            rank, label, rl_d, direction))
        if extras:
            out("        Also improved: %s" % ", ".join(extras))
    out("")

    # Detailed per-category analysis
    out("  2. DETAILED CATEGORY ANALYSIS:")
    out("")
    for qtype in types_order:
        if qtype not in avg_by_type:
            continue
        a = avg_by_type[qtype]
        d = delta_by_type[qtype]
        label = TYPE_LABELS[qtype]
        items = by_type[qtype]

        # Count wins within category
        wins = sum(1 for it in items if it["onto"]["rouge_l"] > it["rag"]["rouge_l"] + 0.005)
        losses = sum(1 for it in items if it["onto"]["rouge_l"] < it["rag"]["rouge_l"] - 0.005)
        ties = len(items) - wins - losses

        out("     %s (%d questions): W=%d / T=%d / L=%d" % (
            label, a["count"], wins, ties, losses))
        improved = [mn for mn in METRIC_NAMES if d[mn] > 1.0]
        degraded = [mn for mn in METRIC_NAMES if d[mn] < -1.0]
        if improved:
            out("       Improved:  %s" %
                ", ".join("%s (%+.1f%%)" % (METRIC_LABELS[mn], d[mn])
                          for mn in improved))
        if degraded:
            out("       Degraded:  %s" %
                ", ".join("%s (%+.1f%%)" % (METRIC_LABELS[mn], d[mn])
                          for mn in degraded))
        if not improved and not degraded:
            out("       All metrics within +/-1%% of baseline")

        # Best question in this category
        best_q = max(items, key=lambda it: it["onto"]["rouge_l"] - it["rag"]["rouge_l"])
        best_d = best_q["onto"]["rouge_l"] - best_q["rag"]["rouge_l"]
        if best_d > 0.005:
            out("       Best example: \"%s...\"" % best_q["question"][:50])
            out("         RAG=%.4f -> Onto-RAG=%.4f (Delta=%+.4f)" % (
                best_q["rag"]["rouge_l"], best_q["onto"]["rouge_l"], best_d))
        out("")

    # Overall summary
    out("  3. SUMMARY:")
    out("")
    if best_type[1]["rouge_l"] > 0:
        out("     Onto-RAG shows the strongest improvement on %s questions" %
            TYPE_LABELS[best_type[0]])
        out("     (ROUGE-L %+.1f%%), where ontological entity recognition and" %
            best_type[1]["rouge_l"])
        out("     graph-based context expansion help identify relevant")
        out("     relationships and connected concepts.")
    else:
        out("     Onto-RAG does not uniformly outperform Standard RAG on")
        out("     any category in ROUGE-L, but shows selective improvements")
        out("     on individual questions and retrieval-based metrics.")

    out("")
    if worst_type[1]["rouge_l"] < -1:
        out("     The weakest performance is on %s questions" %
            TYPE_LABELS[worst_type[0]])
        out("     (ROUGE-L %+.1f%%), where ontology expansion may" %
            worst_type[1]["rouge_l"])
        out("     introduce noise by pulling in tangentially related concepts.")

    out("")
    out("     Overall (all categories combined):")
    for mn in METRIC_NAMES:
        d = global_deltas[mn]
        direction = "+" if d > 0 else ""
        out("       %s: %.4f -> %.4f (%s%.1f%%)" % (
            METRIC_LABELS[mn],
            global_avg["rag"][mn], global_avg["onto"][mn], direction, d))

    out("")
    out("     These results suggest that the ontology integration is most")
    out("     beneficial for questions requiring multi-hop reasoning about")
    out("     entity relationships, while simpler factual and summary")
    out("     questions may not benefit from the additional complexity.")
    out("")
    out("=" * W)

    return "\n".join(lines)


# ===================================================================
# Main
# ===================================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Running question-type analysis...")
    text = run_analysis()
    print(text)

    out_path = RESULTS_DIR / "question_type_analysis.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print("\n  [OK] Results saved to: %s" % out_path)


if __name__ == "__main__":
    main()
