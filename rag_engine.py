# -*- coding: utf-8 -*-
"""
RAG Engine v3.1 -- Modular Architecture for Academic Reproducibility.

Layers:
  1. Retrieval Layer   -- RetrievalIndex (TF-IDF / BM25)
  2. Ontology Layer    -- OntologyGraph + GraphExpander
  3. Ranking Layer     -- OntologyReranker
  4. Routing Layer     -- QueryClassifier + AdaptiveRAG

System modes:
  - StandardRAG   : retrieval only (baseline)
  - OntoRAG       : retrieval + ontology + reranking
  - AdaptiveRAG   : classifier-controlled routing

All parameters are explicit and configurable via RAGConfig.
"""

import math
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path


# ===================================================================
# Configuration
# ===================================================================

@dataclass
class RAGConfig:
    """Global configuration -- all parameters explicit and loggable.

    Fields:
      retrieval_mode   -- "tfidf" (primary, backward compat) or "bm25"
      top_k            -- number of retrieved chunks
      bfs_depth        -- 0 = entity-only, 1 = 1-hop BFS, 2 = deep BFS
      bfs_max_entities -- post-filter cap after full BFS traversal
      ontology_weight  -- reranker weight: 0.0 = off, 1.0 = full ontology
    """
    retrieval_mode: str = "tfidf"
    top_k: int = 3
    bfs_depth: int = 1
    bfs_max_entities: int = 5
    ontology_weight: float = 0.15

    def to_dict(self) -> dict:
        return asdict(self)


# ===================================================================
# NLP Utilities
# ===================================================================

_STOP_WORDS = set(
    # Russian
    "и в на с по к у о из за от до а но не что как это он она "
    "они его её их мы вы я ты был была были будет для при из-за "
    "так же ещё уже все всё то тот эта этот эти также через между "
    "после перед или когда если бы ли только даже где кто чем "
    "чего тем кем ком нас вас них ней ему ей нам вам ими ею"
    # English
    " the a an is are was were be been being have has had do does did "
    "will would shall should may might can could of in to for on with "
    "at by from as into through during before after above below between "
    "out off over under again further then once here there when where "
    "why how all each every both few more most other some such no nor "
    "not only own same so than too very and but or if while that this "
    "these those it its he she they them their his her we you".split()
)


def _tokenize(text: str) -> list[str]:
    """Lowercase tokenization: letters + digits, length > 1."""
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1]


def _remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in _STOP_WORDS]


# ===================================================================
# Retrieval Layer -- TF-IDF Index (preserved from v1)
# ===================================================================

class TfIdfIndex:
    """Lightweight TF-IDF index (pure Python)."""

    def __init__(self, documents: list[str]):
        self.docs = documents
        self.n = len(documents)
        self.doc_tokens: list[list[str]] = []
        self.doc_tf: list[dict[str, float]] = []
        self.idf: dict[str, float] = {}
        self._build()

    def _build(self):
        df: dict[str, int] = defaultdict(int)
        for doc in self.docs:
            tokens = _remove_stopwords(_tokenize(doc))
            self.doc_tokens.append(tokens)
            tf = Counter(tokens)
            total = len(tokens) if tokens else 1
            self.doc_tf.append({t: c / total for t, c in tf.items()})
            for t in set(tokens):
                df[t] += 1
        for term, freq in df.items():
            self.idf[term] = math.log((self.n + 1) / (freq + 1)) + 1

    def _tfidf_vector(self, tf_dict: dict[str, float]) -> dict[str, float]:
        return {t: v * self.idf.get(t, 0) for t, v in tf_dict.items()}

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        common = set(a) & set(b)
        if not common:
            return 0.0
        dot = sum(a[k] * b[k] for k in common)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """Returns top-K (index, score)."""
        qtokens = _remove_stopwords(_tokenize(query))
        qtf = Counter(qtokens)
        total = len(qtokens) if qtokens else 1
        qtf_norm = {t: c / total for t, c in qtf.items()}
        qvec = self._tfidf_vector(qtf_norm)
        scores = []
        for i, tf in enumerate(self.doc_tf):
            dvec = self._tfidf_vector(tf)
            scores.append((i, self._cosine(qvec, dvec)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ===================================================================
# Retrieval Layer -- BM25 Index (new in v3.1)
# ===================================================================

class BM25Index:
    """Okapi BM25 scoring (k1=1.5, b=0.75)."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = documents
        self.n = len(documents)
        self.k1 = k1
        self.b = b
        self.doc_tokens: list[list[str]] = []
        self.doc_tf: list[Counter] = []
        self.doc_len: list[int] = []
        self.idf: dict[str, float] = {}
        self.avgdl: float = 1.0
        self._build()

    def _build(self):
        df: dict[str, int] = defaultdict(int)
        total_len = 0
        for doc in self.docs:
            tokens = _remove_stopwords(_tokenize(doc))
            self.doc_tokens.append(tokens)
            self.doc_tf.append(Counter(tokens))
            self.doc_len.append(len(tokens))
            total_len += len(tokens)
            for t in set(tokens):
                df[t] += 1
        self.avgdl = total_len / self.n if self.n > 0 else 1.0
        for term, freq in df.items():
            self.idf[term] = math.log(
                (self.n - freq + 0.5) / (freq + 0.5) + 1.0
            )

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """Returns top-K (index, score)."""
        qtokens = _remove_stopwords(_tokenize(query))
        scores = []
        for i in range(self.n):
            score = 0.0
            dl = self.doc_len[i]
            for qt in qtokens:
                if qt not in self.idf:
                    continue
                tf = self.doc_tf[i].get(qt, 0)
                idf_val = self.idf[qt]
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += idf_val * num / den
            scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ===================================================================
# Retrieval Layer -- Unified Interface
# ===================================================================

class RetrievalIndex:
    """Unified search interface: mode='tfidf' | 'bm25'."""

    def __init__(self, documents: list[str], mode: str = "tfidf"):
        self.mode = mode
        if mode == "bm25":
            self._engine = BM25Index(documents)
        else:
            self._engine = TfIdfIndex(documents)

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        return self._engine.search(query, top_k)


# ===================================================================
# Ontology Layer -- OWL Parser
# ===================================================================

_NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "calc": "http://example.org/calculus#",
}
_BASE = "http://example.org/calculus#"


def _short(uri: str) -> str:
    return uri.replace(_BASE, "") if uri else ""


_PRED_TRANSLATIONS = {
    "isPartOf": "is part of",
    "requires": "requires",
    "usedIn": "is used in",
    "proves": "proves / applies to",
    "generalizationOf": "is a generalization of",
    "relatedTo": "is related to",
    "inverseOf": "is the inverse of",
    "appliedIn": "is applied in",
    "extendsTo": "extends",
}


class OntologyGraph:
    """Graph from OWL ontology."""

    def __init__(self, owl_path: str):
        self.labels: dict[str, str] = {}
        self.descriptions: dict[str, str] = {}
        self.edges: list[tuple[str, str, str]] = []
        self.adj: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self._parse(owl_path)

    def _parse(self, path: str):
        tree = ET.parse(path)
        root = tree.getroot()
        for elem in root.iter():
            about = elem.get(f"{{{_NS['rdf']}}}about", "")
            if about:
                eid = _short(about)
                label_el = elem.find("calc:label_en", _NS)
                if label_el is not None and label_el.text:
                    self.labels[eid] = label_el.text
                desc_el = elem.find("calc:description_en", _NS)
                if desc_el is not None and desc_el.text:
                    self.descriptions[eid] = desc_el.text
                for child in elem:
                    tag = child.tag.replace(f"{{{_NS['calc']}}}", "")
                    if tag in ("label_en", "description_en"):
                        continue
                    res = child.get(f"{{{_NS['rdf']}}}resource", "")
                    if res:
                        obj_id = _short(res)
                        self.edges.append((eid, tag, obj_id))
                        self.adj[eid].append((tag, obj_id))
                        self.adj[obj_id].append((tag + "_inv", eid))

    def find_entities_in_text(self, text: str) -> list[str]:
        """Find ontology entity mentions in text (word-boundary match)."""
        text_lower = text.lower()
        found = []
        for eid, label in self.labels.items():
            keywords = [w for w in label.lower().split() if len(w) >= 3]
            for kw in keywords:
                # v3.1: word-boundary match instead of substring
                if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                    found.append(eid)
                    break
        return found

    def get_related_entities(self, entity_id: str,
                             depth: int = 2) -> set[str]:
        """BFS traversal -- returns set of related entity IDs.
        Kept for backward compatibility; prefer GraphExpander for new code.
        """
        visited = {entity_id}
        frontier = [entity_id]
        for _ in range(depth):
            next_frontier = []
            for node in frontier:
                for _, neighbor in self.adj.get(node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
            frontier = next_frontier
        return visited

    def get_context_sentences(self, entity_ids: set[str]) -> list[str]:
        """Generate natural-language sentences from ontology for entities."""
        sentences = []
        seen = set()
        for eid in entity_ids:
            label = self.labels.get(eid, eid)
            desc = self.descriptions.get(eid, "")
            if desc and eid not in seen:
                sentences.append(f"{label}: {desc}.")
                seen.add(eid)
            for pred, obj_id in self.adj.get(eid, []):
                if pred.endswith("_inv"):
                    continue
                obj_label = self.labels.get(obj_id, obj_id)
                key = (eid, pred, obj_id)
                if key not in seen:
                    pred_en = _PRED_TRANSLATIONS.get(pred, pred)
                    sentences.append(f"{label} {pred_en} {obj_label}.")
                    seen.add(key)
        return sentences


# ===================================================================
# Ontology Layer -- GraphExpander (isolated BFS, new in v3.1)
# ===================================================================

class GraphExpander:
    """Isolated BFS module with query-relevance filtering and weight decay.

    Design decisions (v3.2):
      - max_depth is a CLASS parameter (not passed dynamically)
      - max_entities is applied as POST-FILTER (full BFS first, then cap)
      - Weight combines hop distance AND query relevance
      - Semantic relations prioritized over structural ones
      - Generic/abstract nodes excluded from results (not from traversal)
    """

    _GENERIC_NODES = frozenset({
        "SingleVariableCalculus", "MultivariableCalculus",
        "VectorCalculus", "DifferentialEquations",
    })

    # Semantic relations get full weight; structural ones are penalized
    _SEMANTIC_RELATIONS = frozenset({
        "inverseOf", "relatedTo", "requires", "proves",
        "generalizationOf", "extendsTo",
    })
    _STRUCTURAL_RELATIONS = frozenset({
        "isPartOf", "usedIn", "appliedIn",
    })
    _STRUCTURAL_PENALTY = 0.4  # structural edges get 40% of normal weight

    def __init__(self, ontology: OntologyGraph, max_depth: int = 1,
                 max_entities: int = 5):
        self._ontology = ontology
        self._max_depth = max_depth
        self._max_entities = max_entities

    @staticmethod
    def _token_overlap(text_a: str, text_b: str) -> float:
        """Fraction of tokens in text_a found in text_b."""
        tokens_a = set(w for w in re.findall(r"[a-z0-9]+", text_a.lower())
                       if len(w) > 2)
        tokens_b = set(w for w in re.findall(r"[a-z0-9]+", text_b.lower())
                       if len(w) > 2)
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / len(tokens_a)

    def expand(self, seed_entities: list[str],
               query: str = "") -> list[tuple[str, float]]:
        """BFS with query-relevance filtering.

        Args:
            seed_entities: entities found in the query
            query: original query text (for relevance scoring)

        Returns:
            [(entity_id, weight)] sorted by weight desc.
        """
        if not seed_entities:
            return []

        # Full BFS with distance and relation-type tracking
        # result: entity_id -> (min_hop_distance, best_relation_type)
        result: dict[str, tuple[int, str]] = {}

        for seed in seed_entities:
            visited = {seed: 0}
            edge_type = {seed: "seed"}
            frontier = [seed]
            for depth_level in range(1, self._max_depth + 1):
                next_frontier = []
                for node in frontier:
                    for rel, neighbor in self._ontology.adj.get(node, []):
                        if neighbor not in visited:
                            visited[neighbor] = depth_level
                            # Strip _inv suffix to get base relation
                            base_rel = rel.replace("_inv", "")
                            edge_type[neighbor] = base_rel
                            next_frontier.append(neighbor)
                frontier = next_frontier

            for eid, dist in visited.items():
                if eid in self._GENERIC_NODES:
                    continue
                if eid not in result or dist < result[eid][0]:
                    result[eid] = (dist, edge_type.get(eid, "unknown"))

        # Weight by: (1) distance, (2) relation type, (3) query relevance
        weighted = []
        for eid, (dist, rel_type) in result.items():
            # Base weight by distance
            base_w = 1.0 / (1 + dist)

            # Relation-type modifier
            if rel_type in self._STRUCTURAL_RELATIONS:
                base_w *= self._STRUCTURAL_PENALTY
            # seed entities always get full weight (rel_type == "seed")

            # Query-relevance modifier (if query provided)
            if query and dist > 0:
                label = self._ontology.labels.get(eid, eid)
                desc = self._ontology.descriptions.get(eid, "")
                entity_text = label + " " + desc
                relevance = self._token_overlap(query, entity_text)
                # Blend: 60% structure + 40% relevance
                base_w *= (0.6 + 0.4 * relevance)

            weighted.append((eid, base_w))

        # Post-filter: sort by weight desc, cap at max_entities
        weighted.sort(key=lambda x: (-x[1], x[0]))  # stable sort by id
        return weighted[:self._max_entities]


# ===================================================================
# Ranking Layer -- OntologyReranker (new in v3.1)
# ===================================================================

class OntologyReranker:
    """Weighted reranking with adaptive weight scaling (v3.2).

    score = (1 - w_eff) * base_score + w_eff * relevance_signal

    Improvements over v3.1:
      - Adaptive w_eff: scales with entity count (few entities = low weight)
      - Query-aware overlap: entity keywords weighted by query co-occurrence
      - Preserves base ranking when ontology signal is weak
    """

    def __init__(self, w_onto: float = 0.15):
        self.w_onto = w_onto

    def rerank(self, candidates: list[tuple[int, float]],
               corpus: list[str],
               entity_keywords: set[str],
               query: str = "",
               n_seed_entities: int = 0) -> list[tuple[int, float]]:
        """Rerank candidates by entity overlap with adaptive weighting.

        Args:
            candidates: [(chunk_index, base_score)]
            corpus: full document list
            entity_keywords: lowercase keywords from expanded entities
            query: original query (for co-occurrence weighting)
            n_seed_entities: number of entities found in query directly

        Returns:
            [(chunk_index, final_score)] sorted by score desc.
        """
        if not entity_keywords or self.w_onto == 0.0:
            result = sorted(candidates, key=lambda x: x[1], reverse=True)
            return result

        # Adaptive weight: scale by entity confidence
        # 1 entity -> 30% of w_onto, 2 -> 70%, 3+ -> 100%
        if n_seed_entities <= 1:
            w_eff = self.w_onto * 0.3
        elif n_seed_entities == 2:
            w_eff = self.w_onto * 0.7
        else:
            w_eff = self.w_onto

        # Pre-compute query tokens for co-occurrence weighting
        query_tokens = set(w for w in re.findall(r"[a-z0-9]+", query.lower())
                           if len(w) > 2) if query else set()

        reranked = []
        n_kw = len(entity_keywords)
        for idx, base_score in candidates:
            text_lower = corpus[idx].lower()
            text_tokens = set(re.findall(r"[a-z0-9]+", text_lower))

            # Weighted overlap: keywords that also appear in query get 2x
            weighted_hits = 0.0
            for kw in entity_keywords:
                if kw in text_lower:
                    if kw in query_tokens:
                        weighted_hits += 2.0  # query-relevant entity keyword
                    else:
                        weighted_hits += 1.0
            # Normalize: max possible = 2 * n_kw (if all co-occur with query)
            overlap_ratio = weighted_hits / (2.0 * n_kw) if n_kw else 0.0

            # Query relevance guard: if chunk has low base_score,
            # don't let entity overlap push it up too much
            score = (1.0 - w_eff) * base_score + w_eff * overlap_ratio
            reranked.append((idx, score))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked


# ===================================================================
# Routing Layer -- QueryClassifier (new in v3.1)
# ===================================================================

class QueryClassifier:
    """3-signal heuristic classifier for adaptive routing (v3.2).

    Signals:
      - relation: presence of relation keywords (how/why/compare/...)
      - entity_density: fraction of ontology entities found in query
      - negative: presence of factual/definitional keywords (penalty)
      - graph_connectivity: bonus if multiple entities are connected

    Design: threshold raised to 0.5 for more selective routing.
    """

    _RELATION_KEYWORDS = frozenset({
        "how", "why", "relate", "relationship", "compare", "comparison",
        "difference", "between", "connect", "versus", "affect", "influence",
    })

    # Negative keywords: factual/definitional queries where ontology hurts
    _NEGATIVE_KEYWORDS = frozenset({
        "what", "define", "definition", "state", "list", "name",
        "give", "formal", "formula",
    })

    # Design constant: 0.5 chosen via threshold sweep (v3.2)
    DEFAULT_THRESHOLD = 0.5

    def __init__(self, threshold: float = None):
        self.threshold = (threshold if threshold is not None
                          else self.DEFAULT_THRESHOLD)

    def _entities_connected(self, entities: list[str],
                            ontology: OntologyGraph) -> bool:
        """Check if any pair of entities is directly connected in graph."""
        entity_set = set(entities)
        for eid in entities:
            for _, neighbor in ontology.adj.get(eid, []):
                if neighbor in entity_set and neighbor != eid:
                    return True
        return False

    def classify(self, query: str, ontology: OntologyGraph) -> dict:
        """Classify query and return routing decision + confidence.

        Returns dict with keys:
          use_ontology (bool), confidence (float),
          scores (dict), entities_found (list)
        """
        words = _tokenize(query)

        # Score 1: relation keyword presence (binary)
        has_relation = any(w in self._RELATION_KEYWORDS for w in words)
        score_relation = 1.0 if has_relation else 0.0

        # Score 2: entity density
        entities = ontology.find_entities_in_text(query)
        score_entity = min(len(entities) / 3.0, 1.0)

        # Score 3: negative signal (penalty for factual queries)
        has_negative = any(w in self._NEGATIVE_KEYWORDS for w in words)
        penalty = 0.3 if (has_negative and not has_relation) else 0.0

        # Bonus: graph connectivity (entities connected in ontology)
        connectivity_bonus = 0.0
        if len(entities) >= 2 and self._entities_connected(entities, ontology):
            connectivity_bonus = 0.2

        score_total = (0.4 * score_relation + 0.4 * score_entity
                       + connectivity_bonus - penalty)
        score_total = max(0.0, min(1.0, score_total))  # clamp to [0, 1]

        return {
            "use_ontology": score_total > self.threshold,
            "confidence": round(score_total, 4),
            "scores": {
                "relation": round(score_relation, 4),
                "entity_density": round(score_entity, 4),
                "negative_penalty": round(penalty, 4),
                "connectivity_bonus": round(connectivity_bonus, 4),
            },
            "entities_found": entities,
        }


# ===================================================================
# Answer Extraction Utilities
# ===================================================================

def _sentence_split(text: str) -> list[str]:
    """Split text into sentences (by . ! ?)."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def _relevance_to_question(sentence: str, question_tokens: set[str]) -> float:
    """Fraction of question words found in sentence."""
    sent_tokens = set(_tokenize(sentence))
    if not question_tokens:
        return 0.0
    return len(question_tokens & sent_tokens) / len(question_tokens)


def _extract_answer(retrieved: list[tuple[str, float]], question: str,
                    n_sentences: int = 5) -> str:
    """Extract best sentences from retrieved chunks."""
    q_tokens = set(_remove_stopwords(_tokenize(question)))
    all_sentences = []
    for text, tfidf_score in retrieved:
        for sent in _sentence_split(text):
            rel = _relevance_to_question(sent, q_tokens)
            all_sentences.append((sent, rel + tfidf_score * 0.3))
    all_sentences.sort(key=lambda x: x[1], reverse=True)

    seen = set()
    best = []
    for sent, _ in all_sentences:
        key = sent[:50].lower()
        if key not in seen:
            seen.add(key)
            best.append(sent)
            if len(best) >= n_sentences:
                break
    return " ".join(best)


# ===================================================================
# System Mode: StandardRAG (baseline)
# ===================================================================

class StandardRAG:
    """Standard RAG: retrieval + best sentences from top-K chunks.
    No ontology, no classifier.
    """

    def __init__(self, corpus: list[str], top_k: int = 3,
                 config: RAGConfig = None):
        self.config = config or RAGConfig(top_k=top_k)
        self.corpus = corpus
        self.top_k = self.config.top_k
        self.index = RetrievalIndex(corpus, mode=self.config.retrieval_mode)

    def answer(self, question: str) -> dict:
        results = self.index.search(question, self.config.top_k)
        retrieved = [(self.corpus[i], score) for i, score in results]
        answer_text = _extract_answer(retrieved, question, n_sentences=5)

        return {
            "answer": answer_text,
            "retrieved_chunks": retrieved,
            "context_size": len(answer_text),
            "ontology_context": "",
            "log": {
                "mode_used": "standard",
                "ontology_used": False,
                "classification_score": None,
                "retrieval_mode": self.config.retrieval_mode,
                "entity_count": 0,
                "bfs_expanded_count": 0,
                "reranking_applied": False,
            },
        }


# ===================================================================
# System Mode: OntoRAG (full ontology pipeline)
# ===================================================================

class OntoRAG:
    """Ontology-augmented RAG via composition of isolated layers.

    Pipeline: entity recognition -> BFS expansion -> enriched retrieval
              -> reranking -> answer extraction.

    Ablation control via RAGConfig:
      bfs_depth=0, ontology_weight=0.0  =>  EntityRAG equivalent
      bfs_depth=1, ontology_weight=0.0  =>  GraphRAG equivalent
      bfs_depth=1, ontology_weight=0.3  =>  Full OntoRAG
    """

    def __init__(self, corpus: list[str], ontology_path: str,
                 top_k: int = 3, hop_depth: int = 1,
                 config: RAGConfig = None):
        self.config = config or RAGConfig(top_k=top_k, bfs_depth=hop_depth)
        self.corpus = corpus
        self.top_k = self.config.top_k
        self.index = RetrievalIndex(corpus, mode=self.config.retrieval_mode)
        self.ontology = OntologyGraph(ontology_path)
        self.expander = GraphExpander(
            self.ontology,
            max_depth=self.config.bfs_depth,
            max_entities=self.config.bfs_max_entities,
        )
        self.reranker = OntologyReranker(w_onto=self.config.ontology_weight)

    def answer(self, question: str) -> dict:
        # 1. Entity recognition (Ontology Layer)
        q_entities = self.ontology.find_entities_in_text(question)

        # 2. Graph expansion with query relevance (v3.2)
        expanded = self.expander.expand(q_entities, query=question)
        expanded_ids = {eid for eid, _ in expanded}

        # 3. Selective query enrichment (v3.2)
        # Only use seed entities + high-weight expanded entities for query
        seed_set = set(q_entities)
        seed_labels = []
        expanded_labels = []
        for eid, weight in expanded:
            lbl = self.ontology.labels.get(eid, "")
            if lbl:
                if eid in seed_set:
                    seed_labels.append(lbl)
                elif weight >= 0.4:  # only high-relevance expansions
                    expanded_labels.append(lbl)

        # Enriched query: seed labels at full strength only
        enriched_query = question + " " + " ".join(seed_labels)

        # 4. Retrieval (Retrieval Layer)
        # Primary: original query (full weight)
        candidates = {}
        for idx, score in self.index.search(question, self.config.top_k * 2):
            candidates[idx] = score
        # Secondary: enriched query (reduced weight: 0.7 instead of 0.9)
        for idx, score in self.index.search(enriched_query,
                                            self.config.top_k * 2):
            old = candidates.get(idx, 0)
            candidates[idx] = max(old, score * 0.7)
        # Tertiary: expanded labels (low weight: 0.4)
        if expanded_labels:
            exp_query = question + " " + " ".join(expanded_labels)
            for idx, score in self.index.search(exp_query,
                                                self.config.top_k):
                old = candidates.get(idx, 0)
                candidates[idx] = max(old, score * 0.4)

        # 5. Reranking with adaptive weight (v3.2)
        entity_kw = set()
        all_labels = seed_labels + expanded_labels
        for lbl in all_labels:
            for w in lbl.lower().split():
                if len(w) >= 3:
                    entity_kw.add(w)

        candidate_list = list(candidates.items())
        reranked = self.reranker.rerank(
            candidate_list, self.corpus, entity_kw,
            query=question, n_seed_entities=len(q_entities)
        )
        retrieved = [(self.corpus[i], s)
                     for i, s in reranked[:self.config.top_k]]

        # 6. Answer extraction with ontology-enriched context (v3.2)
        # Add ontology descriptions as supplementary sentences
        onto_sentences = self.ontology.get_context_sentences(expanded_ids)
        onto_context = " ".join(onto_sentences)

        # Inject relevant ontology sentences as low-priority retrieval
        enriched_retrieved = list(retrieved)
        if onto_sentences and q_entities:
            # Add ontology context as a pseudo-chunk with low score
            min_score = min(s for _, s in retrieved) if retrieved else 0.0
            enriched_retrieved.append((onto_context, min_score * 0.3))

        answer_text = _extract_answer(enriched_retrieved, question,
                                      n_sentences=5)

        return {
            "answer": answer_text,
            "retrieved_chunks": retrieved,
            "context_size": len(answer_text),
            "ontology_context": onto_context,
            "entities_found": list(q_entities),
            "entities_expanded": [eid for eid, _ in expanded],
            "log": {
                "mode_used": "ontology",
                "ontology_used": True,
                "classification_score": None,
                "retrieval_mode": self.config.retrieval_mode,
                "entity_count": len(q_entities),
                "bfs_expanded_count": len(expanded),
                "reranking_applied": self.config.ontology_weight > 0,
            },
        }


# ===================================================================
# System Mode: AdaptiveRAG (pure routing layer)
# ===================================================================

class AdaptiveRAG:
    """Classifier-controlled routing: decide -> delegate -> return.

    Contains ZERO retrieval/ontology logic.  Only classifies the query
    and delegates to either StandardRAG or OntoRAG.
    """

    def __init__(self, corpus: list[str], ontology_path: str,
                 top_k: int = 3, hop_depth: int = 1,
                 config: RAGConfig = None):
        self.config = config or RAGConfig(top_k=top_k, bfs_depth=hop_depth)
        self._standard = StandardRAG(corpus, config=self.config)
        self._onto = OntoRAG(corpus, ontology_path, config=self.config)
        self._classifier = QueryClassifier()
        self._ontology = self._onto.ontology

    def answer(self, question: str) -> dict:
        # 1. Classify
        classification = self._classifier.classify(question, self._ontology)

        # 2. Delegate
        if classification["use_ontology"]:
            result = self._onto.answer(question)
        else:
            result = self._standard.answer(question)

        # 3. Enrich log with classification info
        result["log"]["mode_used"] = "adaptive"
        result["log"]["classification_score"] = classification["confidence"]
        result["log"]["classification_detail"] = classification["scores"]

        return result
