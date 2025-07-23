import json
from pathlib import Path
from typing import Dict, Set, Tuple
from sklearn.metrics import cohen_kappa_score

def load_json(fp: str) -> dict:
    return json.loads(Path(fp).read_text(encoding="utf-8"))

def save_json(data: dict, fp: str):
    Path(fp).parent.mkdir(parents=True, exist_ok=True)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def extract_trigger_labels(data: Dict) -> Dict[str, Set[Tuple[str, str]]]:
    return {
        pid: {(e["trigger"]["text"].lower(), e["event_type"]) for e in para.get("event_mentions", [])}
        for pid, para in data.items()
    }

def safe_kappa(a_vec, b_vec) -> float:
    # Handle empty or uniform vectors
    if sum(a_vec) + sum(b_vec) == 0 or len(set(a_vec + b_vec)) < 2:
        return 1.0  # Perfect agreement assumed on no-label case
    return cohen_kappa_score(a_vec, b_vec, labels=[0, 1])

def compute_paragraph_kappa(agent_a: dict, agent_b: dict, threshold: float = 0.65):
    a_labels = extract_trigger_labels(agent_a)
    b_labels = extract_trigger_labels(agent_b)

    low_kappa_pids = []
    all_pids = list(set(a_labels) & set(b_labels))

    for pid in all_pids:
        all_keys = list(a_labels[pid] | b_labels[pid])
        a_vec = [1 if k in a_labels[pid] else 0 for k in all_keys]
        b_vec = [1 if k in b_labels[pid] else 0 for k in all_keys]
        kappa = safe_kappa(a_vec, b_vec)
        if kappa < threshold:
            low_kappa_pids.append(pid)

    return set(low_kappa_pids)