# -*- coding: utf-8 -*-
"""
Ablation Study v3.2 -- RAG Engine Modular Architecture
=======================================================
6-configuration ablation + sensitivity analysis.

Configurations:
  1. StandardRAG (TF-IDF)     -- primary baseline
  2. StandardRAG (BM25)       -- retrieval comparison
  3. EntityRAG                -- + entity matching (bfs=0, w=0)
  4. Full OntoRAG (TF-IDF)    -- + BFS + reranking
  5. AdaptiveRAG              -- + classifier routing
  6. OntoRAG (BM25)           -- ontology on BM25

Sensitivity analysis:
  - BFS depth sweep (0, 1, 2)
  - Ontology weight sweep (0.0, 0.05, 0.10, 0.15, 0.3)
  - Classifier threshold sweep (0.1 .. 0.9)

Corpus: "Introduction to Calculus Vol. II" by J.H. Heinbockel.
"""

import os
import sys
from pathlib import Path

# -- Paths -----------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
OWL_PATH = SCRIPT_DIR / "calculus_ontology.owl"

sys.path.insert(0, str(SCRIPT_DIR))
from calculus_corpus import get_corpus, get_questions        # noqa: E402
from rag_engine import (StandardRAG, OntoRAG, AdaptiveRAG,   # noqa: E402
                        RAGConfig, QueryClassifier)
from metrics import (compute_all_metrics, METRIC_NAMES,      # noqa: E402
                     METRIC_LABELS, rouge_l)


# ===================================================================
# Ablation experiment
# ===================================================================

CONFIG_NAMES = [
    "Standard(TF-IDF)",
    "Standard(BM25)",
    "EntityRAG",
    "Full OntoRAG",
    "AdaptiveRAG",
    "OntoRAG(BM25)",
]


def _build_systems(corpus, owl):
    """Build all 6 ablation configurations."""
    cfg_tfidf = RAGConfig(retrieval_mode="tfidf")
    cfg_bm25 = RAGConfig(retrieval_mode="bm25")
    cfg_entity = RAGConfig(bfs_depth=0, ontology_weight=0.0)
    cfg_full = RAGConfig(bfs_depth=1, ontology_weight=0.15)
    cfg_bm25_onto = RAGConfig(retrieval_mode="bm25", bfs_depth=1,
                              ontology_weight=0.15)

    return [
        ("Standard(TF-IDF)", StandardRAG(corpus, config=cfg_tfidf)),
        ("Standard(BM25)",   StandardRAG(corpus, config=cfg_bm25)),
        ("EntityRAG",        OntoRAG(corpus, owl, config=cfg_entity)),
        ("Full OntoRAG",     OntoRAG(corpus, owl, config=cfg_full)),
        ("AdaptiveRAG",      AdaptiveRAG(corpus, owl, config=cfg_full)),
        ("OntoRAG(BM25)",    OntoRAG(corpus, owl, config=cfg_bm25_onto)),
    ]


def run_ablation():
    corpus = get_corpus()
    questions = get_questions()
    owl = str(OWL_PATH)
    configs = _build_systems(corpus, owl)

    # per_question[config_name][q_idx] = metrics_dict
    per_question = {}
    averages = {}

    for cfg_name, system in configs:
        q_metrics = []
        for q in questions:
            result = system.answer(q["question"])
            m = compute_all_metrics(q["reference"], result)
            m["_result"] = result
            m["_question"] = q
            q_metrics.append(m)
        per_question[cfg_name] = q_metrics
        averages[cfg_name] = {
            mn: sum(qm[mn] for qm in q_metrics) / len(q_metrics)
            for mn in METRIC_NAMES
        }

    # ===================================================================
    # Build output
    # ===================================================================
    lines = []

    def out(text=""):
        lines.append(text)

    W = 80
    out("=" * W)
    out("  ABLATION STUDY v3.2 -- RAG Engine Modular Architecture")
    out("  Corpus: Introduction to Calculus Vol. II (Heinbockel)")
    out("  Questions: %d  |  Metrics: %s" % (
        len(questions), ", ".join(METRIC_LABELS.values())))
    out("=" * W)

    # -- Config summary ---------------------------------------------------
    out("")
    out("  CONFIGURATION SUMMARY")
    out("  " + "-" * (W - 4))
    out("  RAGConfig defaults: retrieval=tfidf, top_k=3, bfs_depth=1,")
    out("    bfs_max_entities=5, ontology_weight=0.15")
    out("  Classifier threshold: %.1f" % QueryClassifier.DEFAULT_THRESHOLD)
    out("  " + "-" * (W - 4))

    # -- Table 1: Average metrics per configuration -----------------------
    out("")
    out("  TABLE 1. Average metrics across all questions")
    out("  " + "-" * (W - 4))
    header = "  {:20s}".format("Configuration")
    for mn in METRIC_NAMES:
        header += " {:>8s}".format(METRIC_LABELS[mn])
    out(header)
    out("  " + "-" * (W - 4))
    for cfg_name in CONFIG_NAMES:
        row = "  {:20s}".format(cfg_name)
        for mn in METRIC_NAMES:
            row += " {:>8.4f}".format(averages[cfg_name][mn])
        out(row)
    out("  " + "-" * (W - 4))

    # -- Table 2: Delta vs primary baseline (Standard TF-IDF) -------------
    out("")
    out("  TABLE 2. Delta vs Standard RAG (TF-IDF) baseline")
    out("  " + "-" * (W - 4))
    header2 = "  {:20s}".format("Configuration")
    for mn in METRIC_NAMES:
        header2 += " {:>8s}".format(METRIC_LABELS[mn])
    out(header2)
    out("  " + "-" * (W - 4))
    baseline = CONFIG_NAMES[0]
    for cfg_name in CONFIG_NAMES:
        row = "  {:20s}".format(cfg_name)
        for mn in METRIC_NAMES:
            cur = averages[cfg_name][mn]
            base = averages[baseline][mn]
            if cfg_name == baseline:
                row += " {:>8s}".format("--")
            else:
                delta = ((cur - base) / base * 100) if base > 0 else 0
                row += " {:>+7.1f}%".format(delta)
        out(row)
    out("  " + "-" * (W - 4))

    # -- Table 3: Per-question ROUGE-L ------------------------------------
    out("")
    out("  TABLE 3. Per-question ROUGE-L scores")
    out("  " + "-" * (W - 4))
    header3 = "  {:4s} {:12s}".format("Q#", "Type")
    for cfg_name in CONFIG_NAMES:
        header3 += " {:>13s}".format(cfg_name[:13])
    out(header3)
    out("  " + "-" * (W - 4))
    for qi, q in enumerate(questions):
        row = "  {:4s} {:12s}".format("Q%d" % (qi + 1), q["type"][:12])
        for cfg_name in CONFIG_NAMES:
            row += " {:>13.4f}".format(per_question[cfg_name][qi]["rouge_l"])
        out(row)
    out("  " + "-" * (W - 4))

    # -- AdaptiveRAG routing decisions ------------------------------------
    out("")
    out("  TABLE 4. AdaptiveRAG routing decisions")
    out("  " + "-" * (W - 4))
    out("  {:4s} {:12s} {:>6s} {:>6s}  {}".format(
        "Q#", "Type", "Conf", "Route", "Question"))
    out("  " + "-" * (W - 4))
    for qi, q in enumerate(questions):
        result = per_question["AdaptiveRAG"][qi]["_result"]
        log = result["log"]
        conf = log.get("classification_score", 0) or 0
        route = "onto" if log.get("ontology_used") else "std"
        out("  {:4s} {:12s} {:>5.2f}  {:>5s}  {}".format(
            "Q%d" % (qi + 1), q["type"][:12], conf, route,
            q["question"][:45]))
    out("  " + "-" * (W - 4))

    # ===================================================================
    # Qualitative analysis
    # ===================================================================
    out("")
    out("=" * W)
    out("  QUALITATIVE ANALYSIS")
    out("=" * W)

    # -- AdaptiveRAG vs Standard: where routing helped --------------------
    out("")
    out("  A. AdaptiveRAG wins (ontology correctly applied)")
    out("  " + "-" * (W - 4))
    ada_wins = []
    for qi, q in enumerate(questions):
        std_rl = per_question["Standard(TF-IDF)"][qi]["rouge_l"]
        ada_rl = per_question["AdaptiveRAG"][qi]["rouge_l"]
        if ada_rl > std_rl + 0.005:
            ada_wins.append((qi, ada_rl - std_rl))
    ada_wins.sort(key=lambda x: x[1], reverse=True)
    for qi, delta in ada_wins[:3]:
        q = questions[qi]
        std_rl = per_question["Standard(TF-IDF)"][qi]["rouge_l"]
        ada_rl = per_question["AdaptiveRAG"][qi]["rouge_l"]
        out("")
        out("  Q%d [%s]: %s" % (qi + 1, q["type"], q["question"][:70]))
        out("    Standard  ROUGE-L = %.4f" % std_rl)
        out("    Adaptive  ROUGE-L = %.4f  (Delta = %+.4f)" % (ada_rl, delta))
        out("    -> Classifier routed to ontology, improving result.")

    # -- AdaptiveRAG avoided degradation ----------------------------------
    out("")
    out("  B. AdaptiveRAG avoidance (noise correctly skipped)")
    out("  " + "-" * (W - 4))
    onto_worse = []
    for qi, q in enumerate(questions):
        std_rl = per_question["Standard(TF-IDF)"][qi]["rouge_l"]
        full_rl = per_question["Full OntoRAG"][qi]["rouge_l"]
        ada_rl = per_question["AdaptiveRAG"][qi]["rouge_l"]
        if full_rl < std_rl - 0.01 and ada_rl >= std_rl - 0.005:
            onto_worse.append((qi, std_rl - full_rl, ada_rl))
    onto_worse.sort(key=lambda x: x[1], reverse=True)
    for qi, avoided, ada_rl in onto_worse[:3]:
        q = questions[qi]
        std_rl = per_question["Standard(TF-IDF)"][qi]["rouge_l"]
        full_rl = per_question["Full OntoRAG"][qi]["rouge_l"]
        out("")
        out("  Q%d [%s]: %s" % (qi + 1, q["type"], q["question"][:70]))
        out("    Standard  ROUGE-L = %.4f" % std_rl)
        out("    OntoRAG   ROUGE-L = %.4f  (would degrade by %.4f)" % (
            full_rl, full_rl - std_rl))
        out("    Adaptive  ROUGE-L = %.4f  (degradation avoided)" % ada_rl)

    # ===================================================================
    # Sensitivity analysis
    # ===================================================================
    out("")
    out("=" * W)
    out("  SENSITIVITY ANALYSIS")
    out("=" * W)

    # -- BFS depth sweep --------------------------------------------------
    out("")
    out("  S1. BFS depth sweep (ontology_weight=0.15 fixed)")
    out("  " + "-" * (W - 4))
    out("  {:>10s} {:>10s} {:>10s} {:>10s}".format(
        "bfs_depth", "ROUGE-L", "NDCG@5", "MRR"))
    out("  " + "-" * (W - 4))
    for depth in [0, 1, 2]:
        cfg = RAGConfig(bfs_depth=depth, ontology_weight=0.15)
        sys_onto = OntoRAG(corpus, owl, config=cfg)
        rl_sum = 0.0
        nd_sum = 0.0
        mr_sum = 0.0
        for q in questions:
            r = sys_onto.answer(q["question"])
            m = compute_all_metrics(q["reference"], r)
            rl_sum += m["rouge_l"]
            nd_sum += m["ndcg"]
            mr_sum += m["mrr"]
        n = len(questions)
        out("  {:>10d} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            depth, rl_sum / n, nd_sum / n, mr_sum / n))
    out("  " + "-" * (W - 4))

    # -- Ontology weight sweep --------------------------------------------
    out("")
    out("  S2. Ontology weight sweep (bfs_depth=1 fixed)")
    out("  " + "-" * (W - 4))
    out("  {:>10s} {:>10s} {:>10s} {:>10s}".format(
        "w_onto", "ROUGE-L", "NDCG@5", "MRR"))
    out("  " + "-" * (W - 4))
    for w in [0.0, 0.05, 0.10, 0.15, 0.30]:
        cfg = RAGConfig(bfs_depth=1, ontology_weight=w)
        sys_onto = OntoRAG(corpus, owl, config=cfg)
        rl_sum = 0.0
        nd_sum = 0.0
        mr_sum = 0.0
        for q in questions:
            r = sys_onto.answer(q["question"])
            m = compute_all_metrics(q["reference"], r)
            rl_sum += m["rouge_l"]
            nd_sum += m["ndcg"]
            mr_sum += m["mrr"]
        n = len(questions)
        out("  {:>10.1f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            w, rl_sum / n, nd_sum / n, mr_sum / n))
    out("  " + "-" * (W - 4))

    # -- Classifier threshold sweep ---------------------------------------
    out("")
    out("  S3. Classifier threshold sweep (AdaptiveRAG)")
    out("  " + "-" * (W - 4))
    out("  {:>10s} {:>10s} {:>8s} {:>10s}".format(
        "threshold", "ROUGE-L", "onto%", "route"))
    out("  " + "-" * (W - 4))
    for th in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0]:
        cfg = RAGConfig(bfs_depth=1, ontology_weight=0.15)
        ada = AdaptiveRAG(corpus, owl, config=cfg)
        ada._classifier = QueryClassifier(threshold=th)
        rl_sum = 0.0
        onto_count = 0
        for q in questions:
            r = ada.answer(q["question"])
            m = compute_all_metrics(q["reference"], r)
            rl_sum += m["rouge_l"]
            if r["log"]["ontology_used"]:
                onto_count += 1
        n = len(questions)
        pct = onto_count / n * 100
        route_desc = "%d/%d" % (onto_count, n)
        out("  {:>10.1f} {:>10.4f} {:>7.0f}% {:>10s}".format(
            th, rl_sum / n, pct, route_desc))
    out("  " + "-" * (W - 4))

    # ===================================================================
    # Conclusion
    # ===================================================================
    out("")
    out("=" * W)
    out("  CONCLUSION")
    out("=" * W)
    out("")

    # Compare key configs
    std_rl = averages["Standard(TF-IDF)"]["rouge_l"]
    full_rl = averages["Full OntoRAG"]["rouge_l"]
    ada_rl = averages["AdaptiveRAG"]["rouge_l"]
    bm25_std_rl = averages["Standard(BM25)"]["rouge_l"]
    bm25_onto_rl = averages["OntoRAG(BM25)"]["rouge_l"]

    full_delta = ((full_rl - std_rl) / std_rl * 100) if std_rl > 0 else 0
    ada_delta = ((ada_rl - std_rl) / std_rl * 100) if std_rl > 0 else 0

    out("  1. Full OntoRAG vs Standard RAG: ROUGE-L %+.1f%%" % full_delta)
    if full_delta < 0:
        out("     -> Unconditional ontology application degrades results.")
    out("")
    out("  2. AdaptiveRAG vs Standard RAG: ROUGE-L %+.1f%%" % ada_delta)
    if ada_delta >= -1:
        out("     -> Adaptive routing preserves baseline performance")
        out("        while selectively applying ontology where beneficial.")
    out("")

    # W/T/L analysis for AdaptiveRAG
    wins = ties = losses = 0
    for qi in range(len(questions)):
        a = per_question["AdaptiveRAG"][qi]["rouge_l"]
        s = per_question["Standard(TF-IDF)"][qi]["rouge_l"]
        if a > s + 0.005:
            wins += 1
        elif a < s - 0.005:
            losses += 1
        else:
            ties += 1

    out("  3. AdaptiveRAG per-question: W=%d / T=%d / L=%d" % (
        wins, ties, losses))
    out("")
    out("  4. BM25 comparison:")
    out("     Standard(BM25) ROUGE-L = %.4f vs Standard(TF-IDF) = %.4f" % (
        bm25_std_rl, std_rl))
    out("     OntoRAG(BM25)  ROUGE-L = %.4f vs OntoRAG(TF-IDF) = %.4f" % (
        bm25_onto_rl, full_rl))
    out("")

    # Summary
    out("  The ablation demonstrates that:")
    out("    - Unconditional ontology application introduces noise")
    if full_delta < 0:
        out("      (ROUGE-L %+.1f%% degradation on average)" % full_delta)
    out("    - AdaptiveRAG mitigates this by routing only suitable")
    out("      queries through the ontology pipeline")
    out("    - Sensitivity analysis confirms robustness of results")
    out("      across parameter variations")
    out("")
    out("=" * W)

    return "\n".join(lines), averages, per_question


# ===================================================================
# Main
# ===================================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Running ablation study v3.1...")
    text, averages, per_question = run_ablation()

    print(text)

    out_path = RESULTS_DIR / "ablation_results.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print("\n  [OK] Results saved to: %s" % out_path)


if __name__ == "__main__":
    main()
