import json
import argparse
from pathlib import Path
from typing import Dict, Any
import logging
import yaml
from src.core.metrics import compute_agreement

# Setup logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def save_json(data: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main(review_path: str, config_path: str, batch_tag: str):
    reviewed = load_json(review_path)

    if config_path.endswith(".yaml"):
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    else:
        config = load_json(config_path)

    metrics = compute_agreement(reviewed, batch_tag, config=config)
    metrics_path = f"reports/{batch_tag}_metrics.json"
    save_json(metrics, metrics_path)

    low_types = metrics.get("low_precision_types", {})
    if low_types:
        print(f"Batch '{batch_tag}' failed QA for types: {list(low_types.keys())}")
        save_json(reviewed, f"data/final/{batch_tag}_flagged_for_qc.json")
    else:
        print(f"Batch '{batch_tag}' passed QA.")
        save_json(reviewed, f"data/final/{batch_tag}_accepted.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", required=True, help="Path to reviewed Agent B JSON")
    parser.add_argument("--config", default="config/pipeline.yaml", help="Path to config file (YAML or JSON)")
    parser.add_argument("--batch", required=True, help="Batch tag (e.g., tokenized_data_500)")
    args = parser.parse_args()

    main(args.review, args.config, args.batch)