import json
import argparse
from pathlib import Path
from sklearn.metrics import cohen_kappa_score
from typing import Dict, List, Tuple
from src.core.utils import load_json, save_json, extract_trigger_labels, safe_kappa

def compute_kappa_sorted(agent_a: dict, agent_b: dict) -> List[Tuple[str, float]]:
    a_labels = extract_trigger_labels(agent_a)
    b_labels = extract_trigger_labels(agent_b)
    pids = list(set(a_labels) & set(b_labels))
    result = []

    for pid in pids:
        all_keys = list(a_labels[pid] | b_labels[pid])
        a_vec = [1 if k in a_labels[pid] else 0 for k in all_keys]
        b_vec = [1 if k in b_labels[pid] else 0 for k in all_keys]
        kappa = safe_kappa(a_vec, b_vec)
        result.append((pid, kappa))

    result.sort(key=lambda x: x[1])  # sort by kappa ascending
    return result

def main(batch_tag: str, config_path: str = "config/pipeline.yaml"):
    if config_path.endswith(".yaml"):
        import yaml
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    else:
        config = load_json(config_path)

    agent_a = load_json(f"data/processed/agentA/{batch_tag}.json")
    agent_b = load_json(f"data/processed/agentB/{batch_tag}.json")

    sorted_kappa = compute_kappa_sorted(agent_a, agent_b)
    total_samples = max(1, len(agent_b) // 10)
    selected_pids = [pid for pid, _ in sorted_kappa[:total_samples]]
    qc_data = {pid: agent_b[pid] for pid in selected_pids}

    qc_outfile = f"data/processed/human/{batch_tag}_sample.json"
    save_json(qc_data, qc_outfile)

    print(f"Selected {len(selected_pids)} worst-agreement samples for Human QC")
    print(f"QC file saved to: {qc_outfile}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True, help="Batch tag (e.g. tokenized_data_500)")
    parser.add_argument("--config", default="config/pipeline.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    main(batch_tag=args.batch, config_path=args.config)