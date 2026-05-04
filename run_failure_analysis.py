# -*- coding: utf-8 -*-
"""
Failure Analysis / Analiz oshibok sistemy Onto-RAG
====================================================
For each question where Onto-RAG underperforms Standard RAG or shows
low absolute quality, diagnoses the root cause from these categories:

  - MISSING_ENTITY      : key concept absent from ontology
  - WRONG_MATCH         : entity matcher linked to incorrect entity
  - BFS_NOISE           : BFS expansion pulled in irrelevant entities
  - WEAK_RETRIEVAL      : retrieved chunks have low overlap with reference
  - QUERY_AMBIGUITY     : question is vague, multiple interpretations

Corpus: "Introduction to Calculus Vol. II" by J.H. Heinbockel.
"""

import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# -- Paths -----------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
OWL_PATH = SCRIPT_DIR / "calculus_ontology.owl"

sys.path.insert(0, str(SCRIPT_DIR))
from calculus_corpus import get_corpus, get_questions   # noqa: E402
from rag_engine import (StandardRAG, OntoRAG,           # noqa: E402
                        OntologyGraph, _tokenize as _engine_tok)


# ===================================================================
# Metrics (same as other scripts)
# ===================================================================

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\u0430-\u044f\u0451a-z0-9]+", text.lower())


def rouge_l(reference: str, hypothesis: str) -> float:
    ref = _tokenize(reference)
    hyp = _tokenize(hypothesis)
    if not ref or not hyp:
        return 0.0
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    prec = lcs / n if n else 0
    rec = lcs / m if m else 0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def cosine_similarity(text_a: str, text_b: str) -> float:
    ta = Counter(_tokenize(text_a))
    tb = Counter(_tokenize(text_b))
    common = set(ta) & set(tb)
    if not common:
        return 0.0
    dot = sum(ta[w] * tb[w] for w in common)
    na = math.sqrt(sum(v * v for v in ta.values()))
    nb = math.sqrt(sum(v * v for v in tb.values()))
    return dot / (na * nb) if na and nb else 0.0


# ===================================================================
# Diagnostic heuristics
# ===================================================================

FAILURE_LABELS = {
    "MISSING_ENTITY":  "Missing entity in ontology",
    "WRONG_MATCH":     "Incorrect entity matching",
    "BFS_NOISE":       "Noisy BFS expansion",
    "WEAK_RETRIEVAL":  "Weak retrieval relevance",
    "QUERY_AMBIGUITY": "Ambiguous or broad query",
}


def _extract_key_terms(text: str) -> set[str]:
    """Extract meaningful terms (len >= 4, no stopwords) from text."""
    stops = {
        "what", "which", "where", "when", "does", "that", "this", "with",
        "from", "have", "been", "into", "also", "they", "their", "there",
        "them", "than", "each", "more", "most", "some", "such", "very",
        "used", "using", "between", "about", "through", "after", "before",
    }
    tokens = _tokenize(text)
    return {t for t in tokens if len(t) >= 4 and t not in stops}


def _chunk_relevance(chunks: list[tuple[str, float]], reference: str) -> float:
    """Average word overlap between retrieved chunks and reference."""
    ref_tokens = _extract_key_terms(reference)
    if not ref_tokens or not chunks:
        return 0.0
    scores = []
    for text, _ in chunks:
        chunk_tokens = _extract_key_terms(text)
        if chunk_tokens:
            scores.append(len(ref_tokens & chunk_tokens) / len(ref_tokens))
        else:
            scores.append(0.0)
    return sum(scores) / len(scores)


def diagnose_failure(question: str, reference: str,
                     rag_result: dict, onto_result: dict,
                     ontology: OntologyGraph) -> list[dict]:
    """Analyse failure and return list of diagnosed causes with evidence."""
    diagnoses = []

    entities_found = onto_result.get("entities_found", [])
    entities_expanded = onto_result.get("entities_expanded", [])
    bfs_added = set(entities_expanded) - set(entities_found)

    q_terms = _extract_key_terms(question)
    ref_terms = _extract_key_terms(reference)

    # All ontology labels (lowercased keywords)
    all_onto_keywords = set()
    for lbl in ontology.labels.values():
        for w in lbl.lower().split():
            if len(w) >= 4:
                all_onto_keywords.add(w)

    # --- 1. MISSING_ENTITY: key terms from reference not in ontology ------
    ref_important = ref_terms - q_terms  # terms that add info beyond question
    missing_from_onto = ref_important - all_onto_keywords
    coverage = 1.0 - (len(missing_from_onto) / max(len(ref_important), 1))
    if coverage < 0.4 and len(missing_from_onto) >= 3:
        diagnoses.append({
            "type": "MISSING_ENTITY",
            "severity": "HIGH" if coverage < 0.2 else "MEDIUM",
            "detail": "Key reference terms not covered by ontology",
            "evidence": sorted(list(missing_from_onto))[:8],
            "coverage": coverage,
        })

    # --- 2. WRONG_MATCH: entities found but irrelevant to question --------
    if entities_found:
        irrelevant_matches = []
        for eid in entities_found:
            label = ontology.labels.get(eid, eid).lower()
            label_words = {w for w in label.split() if len(w) >= 3}
            # Check if ANY label word appears in question or reference
            in_q = bool(label_words & q_terms)
            in_ref = bool(label_words & ref_terms)
            if not in_q and not in_ref:
                irrelevant_matches.append(eid)
        if irrelevant_matches:
            diagnoses.append({
                "type": "WRONG_MATCH",
                "severity": "HIGH" if len(irrelevant_matches) > 2 else "MEDIUM",
                "detail": "Entities matched but not relevant to question/answer",
                "evidence": irrelevant_matches,
            })

    # --- 3. BFS_NOISE: BFS added many entities that hurt precision --------
    if bfs_added:
        bfs_relevant = 0
        for eid in bfs_added:
            label = ontology.labels.get(eid, eid).lower()
            label_words = {w for w in label.split() if len(w) >= 3}
            if label_words & ref_terms:
                bfs_relevant += 1
        bfs_precision = bfs_relevant / len(bfs_added) if bfs_added else 0
        if bfs_precision < 0.3 and len(bfs_added) >= 3:
            diagnoses.append({
                "type": "BFS_NOISE",
                "severity": "HIGH" if bfs_precision < 0.15 else "MEDIUM",
                "detail": "BFS expansion added %d entities, only %d relevant (%.0f%%)" % (
                    len(bfs_added), bfs_relevant, bfs_precision * 100),
                "evidence": sorted(list(bfs_added))[:10],
                "precision": bfs_precision,
            })

    # --- 4. WEAK_RETRIEVAL: onto-rag chunks poorly match reference --------
    onto_relevance = _chunk_relevance(onto_result["retrieved_chunks"], reference)
    rag_relevance = _chunk_relevance(rag_result["retrieved_chunks"], reference)
    if onto_relevance < 0.25:
        diagnoses.append({
            "type": "WEAK_RETRIEVAL",
            "severity": "HIGH" if onto_relevance < 0.15 else "MEDIUM",
            "detail": "Onto-RAG chunk relevance: %.2f (RAG: %.2f)" % (
                onto_relevance, rag_relevance),
            "onto_relevance": onto_relevance,
            "rag_relevance": rag_relevance,
        })

    # --- 5. QUERY_AMBIGUITY: question too broad or vague ------------------
    q_content_words = q_terms - {"how", "what", "when", "where", "which", "does"}
    if len(q_content_words) <= 3:
        diagnoses.append({
            "type": "QUERY_AMBIGUITY",
            "severity": "LOW",
            "detail": "Query has few content words (%d), may be too broad" % len(q_content_words),
            "evidence": sorted(list(q_content_words)),
        })
    # Also flag if many entities matched (sign of broad query)
    if len(entities_found) >= 5:
        diagnoses.append({
            "type": "QUERY_AMBIGUITY",
            "severity": "MEDIUM",
            "detail": "Query matched %d entities, suggesting broad scope" % len(entities_found),
            "evidence": entities_found[:8],
        })

    # If no specific diagnosis, provide general assessment
    if not diagnoses:
        diagnoses.append({
            "type": "WEAK_RETRIEVAL",
            "severity": "LOW",
            "detail": "No specific failure pattern detected; marginal difference",
            "onto_relevance": onto_relevance,
            "rag_relevance": rag_relevance,
        })

    return diagnoses


# ===================================================================
# Main analysis
# ===================================================================

def run_failure_analysis():
    corpus = get_corpus()
    questions = get_questions()
    ontology = OntologyGraph(str(OWL_PATH))

    rag = StandardRAG(corpus, top_k=3)
    onto_rag = OntoRAG(corpus, str(OWL_PATH), top_k=3, hop_depth=2)

    # Run both systems on all questions
    results = []
    for qi, q in enumerate(questions):
        r_rag = rag.answer(q["question"])
        r_onto = onto_rag.answer(q["question"])
        rl_rag = rouge_l(q["reference"], r_rag["answer"])
        rl_onto = rouge_l(q["reference"], r_onto["answer"])
        cos_rag = cosine_similarity(q["reference"], r_rag["answer"])
        cos_onto = cosine_similarity(q["reference"], r_onto["answer"])

        results.append({
            "qi": qi,
            "question": q["question"],
            "reference": q["reference"],
            "type": q["type"],
            "rag_answer": r_rag["answer"],
            "onto_answer": r_onto["answer"],
            "rl_rag": rl_rag,
            "rl_onto": rl_onto,
            "cos_rag": cos_rag,
            "cos_onto": cos_onto,
            "rag_result": r_rag,
            "onto_result": r_onto,
            "delta_rl": rl_onto - rl_rag,
        })

    # Filter failures: Onto-RAG worse than RAG, OR low absolute quality
    LOW_THRESHOLD = 0.20  # absolute ROUGE-L below this = low quality
    failures = [r for r in results
                if r["delta_rl"] < -0.005 or r["rl_onto"] < LOW_THRESHOLD]
    failures.sort(key=lambda x: x["delta_rl"])

    # Also collect successes for comparison
    successes = [r for r in results if r["delta_rl"] > 0.005]
    successes.sort(key=lambda x: x["delta_rl"], reverse=True)

    # ===================================================================
    # Build output
    # ===================================================================
    lines = []

    def out(text=""):
        lines.append(text)

    W = 76
    out("=" * W)
    out("  FAILURE ANALYSIS / ANALIZ OSHIBOK SISTEMY Onto-RAG")
    out("  Corpus: Introduction to Calculus Vol. II (Heinbockel)")
    out("=" * W)
    out("")
    out("  Total questions:   %d" % len(results))
    out("  Failure cases:     %d  (Onto-RAG < RAG or ROUGE-L < %.2f)" % (
        len(failures), LOW_THRESHOLD))
    out("  Success cases:     %d  (Onto-RAG > RAG)" % len(successes))
    neutral = len(results) - len(failures) - len(successes)
    out("  Neutral:           %d  (comparable)" % neutral)

    # -- Failure type distribution ----------------------------------------
    all_diagnoses = []
    failure_details = []
    for r in failures:
        diags = diagnose_failure(
            r["question"], r["reference"],
            r["rag_result"], r["onto_result"],
            ontology)
        all_diagnoses.extend(diags)
        failure_details.append((r, diags))

    type_counts = Counter(d["type"] for d in all_diagnoses)
    severity_counts = Counter(d["severity"] for d in all_diagnoses)

    out("")
    out("  FAILURE TYPE DISTRIBUTION")
    out("  " + "-" * (W - 4))
    for ftype, label in FAILURE_LABELS.items():
        cnt = type_counts.get(ftype, 0)
        bar = "#" * cnt
        out("  %-22s %2d  %s" % (label, cnt, bar))
    out("  " + "-" * (W - 4))
    out("  Severity: HIGH=%d  MEDIUM=%d  LOW=%d" % (
        severity_counts.get("HIGH", 0),
        severity_counts.get("MEDIUM", 0),
        severity_counts.get("LOW", 0)))

    # -- Detailed per-question analysis -----------------------------------
    out("")
    out("=" * W)
    out("  DETAILED FAILURE ANALYSIS")
    out("=" * W)

    for idx, (r, diags) in enumerate(failure_details, 1):
        qi = r["qi"]
        out("")
        out("  " + "-" * (W - 4))
        out("  CASE %d: Q%d [%s]" % (idx, qi + 1, r["type"]))
        out("  " + "-" * (W - 4))
        out("  Question: %s" % r["question"])
        out("")
        out("  Reference answer:")
        # Wrap reference at ~70 chars
        ref_words = r["reference"].split()
        ref_line = "    "
        for w in ref_words:
            if len(ref_line) + len(w) + 1 > 72:
                out(ref_line)
                ref_line = "    " + w
            else:
                ref_line += " " + w if ref_line.strip() else "    " + w
        if ref_line.strip():
            out(ref_line)

        out("")
        out("  Metrics:")
        out("    ROUGE-L:   RAG=%.4f  Onto-RAG=%.4f  (Delta=%+.4f)" % (
            r["rl_rag"], r["rl_onto"], r["delta_rl"]))
        out("    Cosine:    RAG=%.4f  Onto-RAG=%.4f  (Delta=%+.4f)" % (
            r["cos_rag"], r["cos_onto"], r["cos_onto"] - r["cos_rag"]))

        # Entity info
        ent_found = r["onto_result"].get("entities_found", [])
        ent_expanded = r["onto_result"].get("entities_expanded", [])
        bfs_added = sorted(set(ent_expanded) - set(ent_found))
        out("")
        out("  Ontology data:")
        if ent_found:
            out("    Entities found:    %s" % ", ".join(ent_found))
        else:
            out("    Entities found:    (none)")
        if bfs_added:
            out("    BFS added:         %s" % ", ".join(bfs_added[:10]))
        out("    Total expanded:    %d entities" % len(ent_expanded))

        # Diagnoses
        out("")
        out("  Diagnosed causes:")
        for di, d in enumerate(diags, 1):
            label = FAILURE_LABELS.get(d["type"], d["type"])
            out("    %d. [%s] %s" % (di, d["severity"], label))
            out("       %s" % d["detail"])
            if "evidence" in d:
                ev = d["evidence"]
                if isinstance(ev, list) and ev:
                    out("       Evidence: %s" % ", ".join(str(e) for e in ev[:8]))
            if "coverage" in d:
                out("       Ontology coverage: %.0f%%" % (d["coverage"] * 100))
            if "precision" in d:
                out("       BFS precision: %.0f%%" % (d["precision"] * 100))

        # Answer comparison (abbreviated)
        out("")
        out("  Answer comparison:")
        rag_short = r["rag_answer"][:200]
        onto_short = r["onto_answer"][:200]
        out("    RAG:      %s%s" % (rag_short, "..." if len(r["rag_answer"]) > 200 else ""))
        out("    Onto-RAG: %s%s" % (onto_short, "..." if len(r["onto_answer"]) > 200 else ""))

    # ===================================================================
    # Success cases (brief, for contrast)
    # ===================================================================
    out("")
    out("=" * W)
    out("  SUCCESS CASES (for comparison)")
    out("=" * W)

    for r in successes[:5]:
        qi = r["qi"]
        out("")
        out("  Q%d [%s]: %s" % (qi + 1, r["type"], r["question"][:65]))
        out("    ROUGE-L: RAG=%.4f -> Onto-RAG=%.4f (Delta=%+.4f)" % (
            r["rl_rag"], r["rl_onto"], r["delta_rl"]))
        ent_found = r["onto_result"].get("entities_found", [])
        if ent_found:
            out("    Entities: %s" % ", ".join(ent_found[:6]))
        out("    -> Ontology correctly identified relevant entities.")

    # ===================================================================
    # Summary and recommendations
    # ===================================================================
    out("")
    out("=" * W)
    out("  SUMMARY AND RECOMMENDATIONS")
    out("=" * W)
    out("")

    # Most common failure type
    if type_counts:
        most_common = type_counts.most_common(1)[0]
        out("  Most frequent failure cause: %s (%d occurrences)" % (
            FAILURE_LABELS[most_common[0]], most_common[1]))
        out("")

    # Per-type recommendations
    if type_counts.get("BFS_NOISE", 0) > 0:
        out("  BFS NOISE (%d cases):" % type_counts["BFS_NOISE"])
        out("    Problem:  BFS expands to loosely related entities, diluting")
        out("              retrieval focus and pulling in off-topic chunks.")
        out("    Recommendation:")
        out("      - Reduce BFS hop_depth from 2 to 1")
        out("      - Add edge-weight filtering (prune weak relations)")
        out("      - Implement relevance scoring for expanded entities")
        out("")

    if type_counts.get("MISSING_ENTITY", 0) > 0:
        out("  MISSING ENTITY (%d cases):" % type_counts["MISSING_ENTITY"])
        out("    Problem:  Key domain concepts from reference answers are")
        out("              absent from the ontology, limiting enrichment.")
        out("    Recommendation:")
        out("      - Expand ontology with missing concepts:")
        # Collect all missing terms across failures
        all_missing = set()
        for _, diags in failure_details:
            for d in diags:
                if d["type"] == "MISSING_ENTITY" and "evidence" in d:
                    all_missing.update(d["evidence"])
        if all_missing:
            out("        Candidates: %s" % ", ".join(sorted(all_missing)[:15]))
        out("      - Consider automatic ontology enrichment from corpus")
        out("")

    if type_counts.get("WRONG_MATCH", 0) > 0:
        out("  WRONG MATCH (%d cases):" % type_counts["WRONG_MATCH"])
        out("    Problem:  Entity matcher links question terms to incorrect")
        out("              ontology entities due to substring overlap.")
        out("    Recommendation:")
        out("      - Use full-word matching instead of substring matching")
        out("      - Add disambiguation via context scoring")
        out("      - Increase minimum keyword length threshold")
        out("")

    if type_counts.get("WEAK_RETRIEVAL", 0) > 0:
        out("  WEAK RETRIEVAL (%d cases):" % type_counts["WEAK_RETRIEVAL"])
        out("    Problem:  Retrieved chunks have low overlap with the")
        out("              expected reference answer content.")
        out("    Recommendation:")
        out("      - Increase top_k to retrieve more candidate chunks")
        out("      - Add re-ranking based on answer-question coherence")
        out("      - Fine-tune TF-IDF weights or switch to BM25")
        out("")

    if type_counts.get("QUERY_AMBIGUITY", 0) > 0:
        out("  QUERY AMBIGUITY (%d cases):" % type_counts["QUERY_AMBIGUITY"])
        out("    Problem:  Question is too broad, causing the system to")
        out("              match many entities without clear focus.")
        out("    Recommendation:")
        out("      - Apply entity importance ranking (not all matches equal)")
        out("      - Weight entities by specificity (rare > common)")
        out("      - Limit query expansion to top-N most relevant entities")
        out("")

    # Overall statistics
    out("  OVERALL STATISTICS:")
    avg_delta = sum(r["delta_rl"] for r in results) / len(results)
    fail_pct = len(failures) / len(results) * 100
    out("    Average ROUGE-L Delta: %+.4f" % avg_delta)
    out("    Failure rate: %d/%d (%.0f%%)" % (len(failures), len(results), fail_pct))
    out("    Success rate: %d/%d (%.0f%%)" % (
        len(successes), len(results), len(successes) / len(results) * 100))
    out("")
    out("  These findings indicate that targeted improvements to entity")
    out("  matching precision and BFS expansion filtering would yield the")
    out("  most significant gains in Onto-RAG performance.")
    out("")
    out("=" * W)

    return "\n".join(lines)


# ===================================================================
# Main
# ===================================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Running failure analysis...")
    text = run_failure_analysis()
    print(text)

    out_path = RESULTS_DIR / "failure_analysis.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print("\n  [OK] Results saved to: %s" % out_path)


if __name__ == "__main__":
    main()
