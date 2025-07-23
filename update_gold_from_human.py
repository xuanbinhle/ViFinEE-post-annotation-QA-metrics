import argparse
from pathlib import Path
from src.core.utils import load_json, save_json

def update_gold(qc_file, gold_file):
    qc_data = load_json(qc_file)
    gold_path = Path(gold_file)

    if gold_path.exists():
        gold_data = load_json(gold_path)
    else:
        gold_data = {}

    count_updated = 0
    for pid, item in qc_data.items():
        gold_data[pid] = item
        count_updated += 1

    save_json(gold_data, gold_path)
    print(f"Cập nhật {count_updated} đoạn vào gold set.")
    print(f"Đã lưu tại: {gold_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cập nhật master.json từ mẫu đã human QC.")
    parser.add_argument("--qc_file", required=True, help="Đường dẫn tới file QC đã sửa (json).")
    parser.add_argument("--gold_file", required=True, help="Đường dẫn tới master.json.")
    args = parser.parse_args()

    update_gold(args.qc_file, args.gold_file)
