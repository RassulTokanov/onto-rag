# -*- coding: utf-8 -*-
"""
Centralized evaluation metrics for RAG experiments.
All experiment scripts import from this module (DRY).

Metrics:
  - ROUGE-L (F-measure based on LCS)
  - BLEU (simplified, up to 4-grams)
  - Cosine Similarity (word-level bag-of-words)
  - NDCG@K (based on token overlap with reference)
  - MRR (Mean Reciprocal Rank)
"""

import math
import re
from collections import Counter


# ===================================================================
# Tokenization
# ===================================================================

def tokenize(text: str) -> list[str]:
    """Lowercase tokenization for metric computation."""
    return re.findall(r"[a-z0-9]+", text.lower())


# ===================================================================
# Text-generation metrics
# ===================================================================

def rouge_l(reference: str, hypothesis: str) -> float:
    """ROUGE-L: F-measure based on Longest Common Subsequence."""
    ref = tokenize(reference)
    hyp = tokenize(hypothesis)
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


def bleu_score(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """Simplified BLEU score (up to max_n-grams)."""
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)
    if not hyp_tokens or not ref_tokens:
        return 0.0
    scores = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(
            tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1)
        )
        hyp_ngrams = Counter(
            tuple(hyp_tokens[i:i + n]) for i in range(len(hyp_tokens) - n + 1)
        )
        clipped = sum(min(hyp_ngrams[ng], ref_ngrams.get(ng, 0))
                       for ng in hyp_ngrams)
        total = sum(hyp_ngrams.values())
        if total == 0:
            scores.append(0)
        else:
            scores.append(clipped / total)
    if any(s == 0 for s in scores):
        return 0.0
    log_avg = sum(math.log(s) for s in scores) / len(scores)
    bp = min(1.0, math.exp(1 - len(ref_tokens) / len(hyp_tokens)))
    return bp * math.exp(log_avg)


def cosine_similarity(text_a: str, text_b: str) -> float:
    """Word-level cosine similarity (bag-of-words proxy)."""
    ta = Counter(tokenize(text_a))
    tb = Counter(tokenize(text_b))
    common = set(ta) & set(tb)
    if not common:
        return 0.0
    dot = sum(ta[w] * tb[w] for w in common)
    na = math.sqrt(sum(v * v for v in ta.values()))
    nb = math.sqrt(sum(v * v for v in tb.values()))
    return dot / (na * nb) if na and nb else 0.0


# ===================================================================
# Retrieval metrics
# ===================================================================

def ndcg_score(reference: str, retrieved_chunks: list[tuple[str, float]],
               k: int = 5) -> float:
    """NDCG@K based on token overlap with reference answer."""
    ref_tokens = set(tokenize(reference))
    if not ref_tokens:
        return 0.0
    gains = []
    for text, _ in retrieved_chunks[:k]:
        chunk_tokens = set(tokenize(text))
        overlap = len(ref_tokens & chunk_tokens)
        gains.append(overlap / len(ref_tokens) if ref_tokens else 0)
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(gains, reverse=True)
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def mrr_score(reference: str,
              retrieved_chunks: list[tuple[str, float]]) -> float:
    """Mean Reciprocal Rank (threshold = 0.2 token overlap)."""
    ref_tokens = set(tokenize(reference))
    for i, (text, _) in enumerate(retrieved_chunks):
        chunk_tokens = set(tokenize(text))
        if len(ref_tokens & chunk_tokens) / max(len(ref_tokens), 1) > 0.2:
            return 1.0 / (i + 1)
    return 0.0


# ===================================================================
# Convenience
# ===================================================================

METRIC_NAMES = ["rouge_l", "bleu", "cosine", "ndcg", "mrr"]
METRIC_LABELS = {
    "rouge_l": "ROUGE-L",
    "bleu": "BLEU",
    "cosine": "Cosine",
    "ndcg": "NDCG@5",
    "mrr": "MRR",
}


def compute_all_metrics(reference: str, result: dict) -> dict[str, float]:
    """Compute all 5 metrics for a single question result."""
    return {
        "rouge_l": rouge_l(reference, result["answer"]),
        "bleu": bleu_score(reference, result["answer"]),
        "cosine": cosine_similarity(reference, result["answer"]),
        "ndcg": ndcg_score(reference, result["retrieved_chunks"]),
        "mrr": mrr_score(reference, result["retrieved_chunks"]),
    }
