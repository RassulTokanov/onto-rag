# -*- coding: utf-8 -*-
"""
Experiment / Эксперимент: Standard RAG vs Onto-RAG
on the corpus "Introduction to Calculus Vol. II" by J.H. Heinbockel.
Generates HTML page with side-by-side answer comparison,
metric charts, and JSON results file.
"""

import json
import os
import sys
from pathlib import Path

# ── Пути ──────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
OWL_PATH = SCRIPT_DIR / "calculus_ontology.owl"

# ── Импорт ────────────────────────────────────────────────────
sys.path.insert(0, str(SCRIPT_DIR))
from calculus_corpus import get_corpus, get_questions  # noqa: E402
from rag_engine import StandardRAG, OntoRAG             # noqa: E402
from metrics import (rouge_l, bleu_score, cosine_similarity,  # noqa: E402
                     ndcg_score, mrr_score)


# ═══════════════════════════════════════════════════════════════
# Эксперимент
# ═══════════════════════════════════════════════════════════════

def run_experiment():
    """Основной эксперимент."""
    print("=" * 60)
    print("  Experiment / Эксперимент: RAG vs Onto-RAG (Calculus)")
    print("=" * 60)

    corpus = get_corpus()
    questions = get_questions()
    print(f"\n  Корпус / Corpus: {len(corpus)} фрагментов / chunks")
    print(f"  Вопросов / Questions: {len(questions)}")

    # Инициализация
    rag = StandardRAG(corpus, top_k=3)
    onto_rag = OntoRAG(corpus, str(OWL_PATH), top_k=3, hop_depth=2)

    results = []
    totals = {"rag": {}, "onto_rag": {}}
    metric_names = ["rouge_l", "bleu", "cosine", "ndcg", "mrr"]
    for m in metric_names:
        totals["rag"][m] = []
        totals["onto_rag"][m] = []

    for i, q in enumerate(questions):
        question = q["question"]
        reference = q["reference"]
        qtype = q["type"]

        rag_result = rag.answer(question)
        onto_result = onto_rag.answer(question)

        rag_metrics = {
            "rouge_l": rouge_l(reference, rag_result["answer"]),
            "bleu": bleu_score(reference, rag_result["answer"]),
            "cosine": cosine_similarity(reference, rag_result["answer"]),
            "ndcg": ndcg_score(reference, rag_result["retrieved_chunks"]),
            "mrr": mrr_score(reference, rag_result["retrieved_chunks"]),
        }
        onto_metrics = {
            "rouge_l": rouge_l(reference, onto_result["answer"]),
            "bleu": bleu_score(reference, onto_result["answer"]),
            "cosine": cosine_similarity(reference, onto_result["answer"]),
            "ndcg": ndcg_score(reference, onto_result["retrieved_chunks"]),
            "mrr": mrr_score(reference, onto_result["retrieved_chunks"]),
        }

        for m in metric_names:
            totals["rag"][m].append(rag_metrics[m])
            totals["onto_rag"][m].append(onto_metrics[m])

        results.append({
            "id": i + 1,
            "question": question,
            "reference": reference,
            "type": qtype,
            "rag_answer": rag_result["answer"][:500],
            "onto_rag_answer": onto_result["answer"][:800],
            "onto_context": onto_result.get("ontology_context", "")[:400],
            "entities_found": onto_result.get("entities_found", []),
            "rag_metrics": rag_metrics,
            "onto_rag_metrics": onto_metrics,
            "rag_context_size": rag_result["context_size"],
            "onto_rag_context_size": onto_result["context_size"],
        })

        status = "+" if onto_metrics["rouge_l"] > rag_metrics["rouge_l"] else "~"
        print(f"  [{status}] Q{i+1:2d} ({qtype:12s}): "
              f"RAG={rag_metrics['rouge_l']:.3f}  "
              f"Onto={onto_metrics['rouge_l']:.3f}")

    # Средние
    averages = {}
    for system in ("rag", "onto_rag"):
        averages[system] = {m: sum(v) / len(v) for m, v in totals[system].items()}

    print("\n" + "-" * 60)
    print("  СРЕДНИЕ МЕТРИКИ / AVERAGE METRICS:")
    print(f"  {'Метрика/Metric':<12} {'RAG':>8} {'Onto-RAG':>10} {'Delta':>8}")
    print("  " + "-" * 42)
    for m in metric_names:
        r = averages["rag"][m]
        o = averages["onto_rag"][m]
        delta = ((o - r) / r * 100) if r > 0 else 0
        print(f"  {m:<12} {r:>8.3f} {o:>10.3f} {delta:>+7.1f}%")

    return results, averages, totals


# ═══════════════════════════════════════════════════════════════
# Генерация HTML
# ═══════════════════════════════════════════════════════════════

def _generate_html(results: list[dict], averages: dict, totals: dict) -> str:
    """Генерирует HTML со сравнением ответов."""

    type_labels = {
        "factual": "Фактический / Factual",
        "relationship": "О связях / Relationship",
        "reasoning": "Рассуждение / Reasoning",
        "summary": "Обобщение / Summary",
    }

    # SVG-графики метрик
    metric_labels = {
        "rouge_l": "ROUGE-L",
        "bleu": "BLEU",
        "cosine": "Cosine Sim",
        "ndcg": "NDCG@5",
        "mrr": "MRR",
    }

    def bar_chart_svg(title: str, rag_val: float, onto_val: float) -> str:
        max_val = max(rag_val, onto_val, 0.01)
        rag_w = rag_val / max_val * 200
        onto_w = onto_val / max_val * 200
        return f"""
        <div class="chart-card">
          <div class="chart-title">{title}</div>
          <div class="bar-row">
            <span class="bar-label">RAG</span>
            <div class="bar-bg"><div class="bar rag-bar" style="width:{rag_w}px"></div></div>
            <span class="bar-value">{rag_val:.3f}</span>
          </div>
          <div class="bar-row">
            <span class="bar-label">Onto</span>
            <div class="bar-bg"><div class="bar onto-bar" style="width:{onto_w}px"></div></div>
            <span class="bar-value">{onto_val:.3f}</span>
          </div>
        </div>"""

    charts_html = ""
    for m, label in metric_labels.items():
        charts_html += bar_chart_svg(label, averages["rag"][m], averages["onto_rag"][m])

    # По типам
    type_charts = ""
    for qtype, qlabel in type_labels.items():
        rag_vals = [r["rag_metrics"]["rouge_l"] for r in results if r["type"] == qtype]
        onto_vals = [r["onto_rag_metrics"]["rouge_l"] for r in results if r["type"] == qtype]
        if rag_vals and onto_vals:
            r_avg = sum(rag_vals) / len(rag_vals)
            o_avg = sum(onto_vals) / len(onto_vals)
            type_charts += bar_chart_svg(f"ROUGE-L: {qlabel}", r_avg, o_avg)

    # Карточки вопросов
    cards_html = ""
    for r in results:
        ql = type_labels.get(r["type"], r["type"])
        is_better = r["onto_rag_metrics"]["rouge_l"] > r["rag_metrics"]["rouge_l"]
        badge_class = "badge-better" if is_better else "badge-same"
        badge_text = "Onto-RAG лучше / better" if is_better else "Сопоставимо / comparable"

        entities_html = ""
        if r.get("entities_found"):
            entities_html = f"""
            <div class="entities">
              <strong>Найденные сущности / Entities found:</strong> {', '.join(r['entities_found'])}
            </div>"""

        onto_ctx = r.get("onto_context", "")
        onto_ctx_html = ""
        if onto_ctx:
            onto_ctx_html = f"""
            <div class="onto-ctx">
              <strong>Контекст из онтологии / Ontology context:</strong><br>
              <em>{onto_ctx[:300]}{'...' if len(onto_ctx) > 300 else ''}</em>
            </div>"""

        cards_html += f"""
        <div class="qa-card">
          <div class="qa-header">
            <span class="q-number">#{r['id']}</span>
            <span class="q-type">{ql}</span>
            <span class="badge {badge_class}">{badge_text}</span>
          </div>
          <div class="question">{r['question']}</div>
          <div class="reference"><strong>Эталон / Reference:</strong> {r['reference']}</div>
          {entities_html}
          {onto_ctx_html}
          <div class="answers-grid">
            <div class="answer-box rag-box">
              <div class="answer-title">Standard RAG</div>
              <div class="answer-text">{r['rag_answer'][:400]}{'...' if len(r['rag_answer']) > 400 else ''}</div>
              <div class="metrics-row">
                ROUGE-L: {r['rag_metrics']['rouge_l']:.3f} |
                BLEU: {r['rag_metrics']['bleu']:.3f} |
                Cosine: {r['rag_metrics']['cosine']:.3f}
              </div>
            </div>
            <div class="answer-box onto-box">
              <div class="answer-title">Onto-RAG</div>
              <div class="answer-text">{r['onto_rag_answer'][:500]}{'...' if len(r['onto_rag_answer']) > 500 else ''}</div>
              <div class="metrics-row">
                ROUGE-L: {r['onto_rag_metrics']['rouge_l']:.3f} |
                BLEU: {r['onto_rag_metrics']['bleu']:.3f} |
                Cosine: {r['onto_rag_metrics']['cosine']:.3f}
              </div>
            </div>
          </div>
        </div>
        """

    # Сводная таблица
    summary_rows = ""
    metric_names = ["rouge_l", "bleu", "cosine", "ndcg", "mrr"]
    for m in metric_names:
        r = averages["rag"][m]
        o = averages["onto_rag"][m]
        delta = ((o - r) / r * 100) if r > 0 else 0
        color = "#4ade80" if delta > 0 else "#f87171"
        summary_rows += f"""
        <tr>
          <td>{metric_labels.get(m, m)}</td>
          <td>{r:.4f}</td>
          <td>{o:.4f}</td>
          <td style="color:{color};font-weight:700">{delta:+.1f}%</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAG vs Onto-RAG -- Experiment / Эксперимент (Calculus)</title>
<style>
  :root {{
    --bg: #0f172a; --surface: #1e293b; --surface2: #334155;
    --text: #e2e8f0; --text-dim: #94a3b8; --accent: #818cf8;
    --rag: #f97316; --onto: #22d3ee; --better: #4ade80;
    --radius: 12px;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: var(--bg); color: var(--text);
    line-height: 1.6; padding: 2rem;
  }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  h1 {{
    text-align: center; font-size: 2rem;
    background: linear-gradient(135deg, var(--accent), var(--onto));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
  }}
  .subtitle {{ text-align: center; color: var(--text-dim); margin-bottom: 2rem; }}
  h2 {{
    font-size: 1.4rem; color: var(--accent);
    margin: 2rem 0 1rem; padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--surface2);
  }}

  /* Сводная таблица */
  .summary-table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
  .summary-table th, .summary-table td {{
    padding: 0.75rem 1rem; text-align: center;
    border-bottom: 1px solid var(--surface2);
  }}
  .summary-table th {{ background: var(--surface); color: var(--accent); }}
  .summary-table tr:hover {{ background: var(--surface); }}

  /* Графики */
  .charts-grid {{
    display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 1rem; margin: 1rem 0;
  }}
  .chart-card {{
    background: var(--surface); border-radius: var(--radius);
    padding: 1rem; border: 1px solid var(--surface2);
  }}
  .chart-title {{ font-weight: 600; margin-bottom: 0.75rem; color: var(--text); }}
  .bar-row {{ display: flex; align-items: center; gap: 0.5rem; margin: 0.4rem 0; }}
  .bar-label {{ width: 40px; font-size: 0.8rem; color: var(--text-dim); }}
  .bar-bg {{ flex: 1; height: 20px; background: var(--surface2); border-radius: 4px; overflow: hidden; }}
  .bar {{ height: 100%; border-radius: 4px; transition: width 0.5s; }}
  .rag-bar {{ background: linear-gradient(90deg, var(--rag), #fb923c); }}
  .onto-bar {{ background: linear-gradient(90deg, var(--onto), #67e8f9); }}
  .bar-value {{ width: 50px; font-size: 0.85rem; text-align: right; font-weight: 600; }}

  /* Карточки вопросов */
  .qa-card {{
    background: var(--surface); border-radius: var(--radius);
    padding: 1.5rem; margin: 1.5rem 0;
    border: 1px solid var(--surface2);
    transition: border-color 0.3s;
  }}
  .qa-card:hover {{ border-color: var(--accent); }}
  .qa-header {{ display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.75rem; }}
  .q-number {{ font-weight: 700; color: var(--accent); font-size: 1.1rem; }}
  .q-type {{
    background: var(--surface2); padding: 2px 10px; border-radius: 20px;
    font-size: 0.8rem; color: var(--text-dim);
  }}
  .badge {{
    margin-left: auto; padding: 2px 12px; border-radius: 20px;
    font-size: 0.8rem; font-weight: 600;
  }}
  .badge-better {{ background: rgba(74,222,128,0.15); color: var(--better); }}
  .badge-same {{ background: rgba(148,163,184,0.15); color: var(--text-dim); }}
  .question {{ font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem; }}
  .reference {{
    color: var(--text-dim); font-size: 0.9rem;
    margin-bottom: 0.75rem; padding: 0.5rem;
    background: rgba(129,140,248,0.06); border-radius: 8px;
  }}
  .entities {{
    font-size: 0.85rem; color: var(--onto);
    margin-bottom: 0.5rem; padding: 0.4rem 0.6rem;
    background: rgba(34,211,238,0.08); border-radius: 8px;
  }}
  .onto-ctx {{
    font-size: 0.82rem; color: var(--text-dim);
    margin-bottom: 0.75rem; padding: 0.5rem;
    background: rgba(34,211,238,0.05); border-radius: 8px;
    border-left: 3px solid var(--onto);
  }}
  .answers-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
  @media (max-width: 768px) {{ .answers-grid {{ grid-template-columns: 1fr; }} }}
  .answer-box {{
    padding: 1rem; border-radius: 8px;
    border: 1px solid var(--surface2);
  }}
  .rag-box {{ border-left: 3px solid var(--rag); background: rgba(249,115,22,0.04); }}
  .onto-box {{ border-left: 3px solid var(--onto); background: rgba(34,211,238,0.04); }}
  .answer-title {{ font-weight: 700; margin-bottom: 0.5rem; font-size: 0.95rem; }}
  .answer-text {{ font-size: 0.85rem; color: var(--text-dim); line-height: 1.5; }}
  .metrics-row {{
    margin-top: 0.75rem; padding-top: 0.5rem;
    border-top: 1px solid var(--surface2);
    font-size: 0.78rem; color: var(--text-dim); font-family: monospace;
  }}
  footer {{ text-align: center; color: var(--text-dim); margin-top: 3rem; font-size: 0.85rem; }}
</style>
</head>
<body>
<div class="container">

<h1>RAG vs Onto-RAG</h1>
<p class="subtitle">Experimental comparison / Экспериментальное сравнение -- "Introduction to Calculus Vol. II" (Heinbockel)</p>

<h2>Сводка метрик / Metrics Summary</h2>
<table class="summary-table">
  <thead>
    <tr><th>Метрика / Metric</th><th>Standard RAG</th><th>Onto-RAG</th><th>Изменение / Change</th></tr>
  </thead>
  <tbody>{summary_rows}</tbody>
</table>

<h2>Средние метрики / Average Metrics</h2>
<div class="charts-grid">{charts_html}</div>

<h2>ROUGE-L по типам вопросов / by Question Type</h2>
<div class="charts-grid">{type_charts}</div>

<h2>Сравнение ответов / Answer Comparison</h2>
{cards_html}

<footer>
  Диссертация / Dissertation: Integration of ontologies into RAG architecture<br>
  Experiment completed automatically / Корпус / Corpus: "Introduction to Calculus Vol. II" (Heinbockel)
</footer>

</div>
</body>
</html>"""
    return html


# ═══════════════════════════════════════════════════════════════
# Главная функция
# ═══════════════════════════════════════════════════════════════

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results, averages, totals = run_experiment()

    # HTML
    html = _generate_html(results, averages, totals)
    html_path = RESULTS_DIR / "comparison.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n  [OK] HTML: {html_path}")

    # JSON
    json_data = {
        "averages": averages,
        "results": results,
    }
    json_path = RESULTS_DIR / "experiment_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"  [OK] JSON: {json_path}")

    print(f"\n{'=' * 60}")
    print("  Эксперимент завершён / Experiment completed!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
