#src/core/metrics.py
# -------------------------------------------------------------
"""Compute QA metrics for reviewed paragraphs.

*   GOLD = single master file at `<gold_root>/master.json`. 
    It may contain paragraphs sampled từ nhiều batch.
*   Precision‑per‑type is computed **chỉ trên** những paragraph xuất hiện trong
    master gold; prediction ngoài phạm vi này không ảnh hưởng.
*   κ values vẫn là placeholder.
"""
from __future__ import annotations

import json, logging
from pathlib import Path
from typing import Dict, Any
from collections import Counter

LOGGER = logging.getLogger(__name__)

def per_type_precision(pred: dict, gold: dict) -> dict[str, float]:
    """Return precision for each event_type.

    Args:
        pred: reviewed output {pid: {event_mentions:[...]}}
        gold: master gold subset (same schema)
    """
    tp, fp = Counter(), Counter()

    for pid, gold_obj in gold.items():
        gold_set = {
            (e["trigger"]["text"], e["event_type"]) for e in gold_obj.get("event_mentions", [])
        }
        pred_obj = pred.get(pid, {"event_mentions": []})
        for ev in pred_obj.get("event_mentions", []):
            key = (ev["trigger"]["text"], ev["event_type"])
            if key in gold_set:
                tp[ev["event_type"]] += 1
            else:
                fp[ev["event_type"]] += 1

    return {
        t: tp[t] / (tp[t] + fp[t]) if (tp[t] + fp[t]) else 0.0 for t in set(tp) | set(fp)
    }

def _load_gold(gold_root: str = "data/gold") -> dict | None:
    fp = Path(gold_root) / "master.json"
    if not fp.exists():
        LOGGER.warning("Master gold file not found at %s – precision skipped", fp)
        return None
    return json.loads(fp.read_text(encoding="utf-8"))

def compute_agreement(reviewed: Dict, batch_tag: str, *, config: Dict) -> Dict[str, Any]:
    """Return QA metrics for one batch.

    • Always returns placeholder κ values.
    • Adds precision‑per‑type & low_precision_types if master gold available.
    """
    metrics: Dict[str, Any] = {
        "batch": batch_tag,
        "n_paragraphs": len(reviewed),
        "trigger_kappa": 0.80,  # TODO: replace with real Fleiss κ
        "arg_kappa": 0.72,
    }

    gold_root = config.get("paths", {}).get("gold_root", "data/gold")
    gold = _load_gold(gold_root)
    if gold is None:
        return metrics

    prec = per_type_precision(reviewed, gold)
    metrics["precision_per_type"] = prec

    threshold = config.get("metrics", {}).get("per_type_precision_min", 0.80)
    metrics["low_precision_types"] = {t: p for t, p in prec.items() if p < threshold}

    LOGGER.info("Metrics for %s → %s", batch_tag, metrics)
    return metrics