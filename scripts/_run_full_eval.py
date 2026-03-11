#!/usr/bin/env python3
"""Run the full evaluation harness (4 retrieval modes + ER) and write eval_results.json."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from riskfolio_graphrag_agent.er.pipeline import EntityRecord, run_er_pipeline
from riskfolio_graphrag_agent.eval.evaluator import Evaluator, build_default_eval_samples
from riskfolio_graphrag_agent.retrieval.retriever import HybridRetriever, RetrievalMode

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"


def _run_er() -> dict[str, float]:
    # Build a representative entity set from DOMAIN_ALIASES and known aliases
    entities = [
        EntityRecord(entity_id="e1", name="CVaR", source="graph"),
        EntityRecord(entity_id="e2", name="conditional value at risk", source="graph"),
        EntityRecord(entity_id="e3", name="expected shortfall", source="graph"),
        EntityRecord(entity_id="e4", name="Hierarchical Risk Parity", source="graph"),
        EntityRecord(entity_id="e5", name="hierarchical risk parity", source="graph"),
        EntityRecord(entity_id="e6", name="hrp", source="graph"),
        EntityRecord(entity_id="e7", name="VaR", source="graph"),
        EntityRecord(entity_id="e8", name="value at risk", source="graph"),
        EntityRecord(entity_id="e9", name="value-at-risk", source="graph"),
        EntityRecord(entity_id="e10", name="Semi Deviation", source="graph"),
        EntityRecord(entity_id="e11", name="downside deviation", source="graph"),
        EntityRecord(entity_id="e12", name="semidev", source="graph"),
    ]
    gold_pairs = {
        ("e1", "e2"),
        ("e1", "e3"),
        ("e2", "e3"),  # CVaR cluster
        ("e4", "e5"),
        ("e4", "e6"),
        ("e5", "e6"),  # HRP cluster
        ("e7", "e8"),
        ("e7", "e9"),
        ("e8", "e9"),  # VaR cluster
        ("e10", "e11"),
        ("e10", "e12"),
        ("e11", "e12"),  # SemiDev cluster
    }
    result = run_er_pipeline(entities, gold_pairs=gold_pairs)
    metrics = result.metrics
    if metrics is None:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {"precision": metrics.precision, "recall": metrics.recall, "f1": metrics.f1}


def _run_mode(mode: RetrievalMode, er_metrics: dict[str, float]) -> dict:
    samples = build_default_eval_samples()
    with HybridRetriever(
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
        top_k=5,
        retrieval_mode=mode,
    ) as retriever:
        evaluator = Evaluator(
            samples=samples,
            retriever=retriever,
            metric_profile="ragas-style",
            runtime_config={"retrieval_mode": mode, "embedding_provider": "hash"},
            er_metrics=er_metrics,
        )
        report = evaluator.run()
    return asdict(report)


def main() -> int:
    er_metrics = _run_er()
    print(f"ER metrics: {er_metrics}")

    results: dict[str, dict] = {}
    for mode in ("dense", "sparse", "graph", "hybrid_rerank"):
        print(f"Running evaluator mode={mode} ...")
        results[mode] = _run_mode(mode, er_metrics)
        r = results[mode]
        print(
            f"  recall={r['context_recall']:.4f}  precision={r['context_precision']:.4f}"
            f"  faithfulness={r['answer_faithfulness']:.4f}  latency={r['avg_latency_ms']:.1f}ms"
        )

    # Primary result for eval_results.json = sparse (new router default)
    primary = dict(results["sparse"])  # copy to avoid mutating
    primary["er_precision"] = er_metrics["precision"]
    primary["er_recall"] = er_metrics["recall"]
    primary["er_f1"] = er_metrics["f1"]

    all_modes_summary = {
        mode: {
            "context_recall": r["context_recall"],
            "context_precision": r["context_precision"],
            "answer_faithfulness": r["answer_faithfulness"],
            "answer_relevance": r["answer_relevance"],
            "grounding": r["grounding"],
            "multi_hop_accuracy": r["multi_hop_accuracy"],
            "avg_latency_ms": r["avg_latency_ms"],
            "estimated_cost_usd": r["estimated_cost_usd"],
        }
        for mode, r in results.items()
    }
    primary["all_modes"] = all_modes_summary

    out = Path("eval_results.json")
    out.write_text(json.dumps(primary, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
