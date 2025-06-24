# Hướng Dẫn Quy Trình Sau Gán Nhãn (Post-Annotation QC & Đánh Giá)

Quy trình này áp dụng sau khi Agent A và Agent B hoàn thành việc gán nhãn, với các mục tiêu sau:

1.  **Chọn mẫu QC**: Lấy **10%** mẫu có điểm kappa thấp nhất giữa hai Agent để chuyên viên kiểm định lại.
2.  **Cập nhật Gold Set**: Tích hợp các mẫu đã QC vào bộ dữ liệu Gold (`master.json`).
3.  **Đánh giá Batch**: Đánh giá toàn bộ lô dữ liệu dựa trên Gold Set.

---

## Ví dụ: Batch `tokenized_data_500`

**Đầu vào cần có:**

* `data/processed/agentA/tokenized_data_500.json`
* `data/processed/agentB/tokenized_data_500.json`
* `config/pipeline.yaml`

---
## Setup môi trường
```bash
pip install -r requirements.txt
```

## Các Bước Thực Hiện

### Bước 1: Lấy 10% Mẫu Cần QC

Chạy lệnh sau để chọn 10% đoạn có kappa thấp nhất:

```bash
python prepare_QC_samples.py \
  --batch tokenized_data_500 \
  --config config/pipeline.yaml
```
Kết quả: File mẫu cần QC được lưu tại ```data/processed/human/tokenized_data_500_sample.json.```

### Bước 2: QC Thủ Công & Cập Nhật Gold Set
Sau khi chuyên viên đã kiểm tra và sửa lỗi trong file *_sample.json, chạy lệnh sau để cập nhật vào Gold Set:

```Bash

python update_gold_from_human.py \
  --qc_file data/processed/human/tokenized_data_500_sample.json \
  --gold_file data/gold/master.json
```

### Bước 3: Đánh Giá Batch với Gold Set

Để đánh giá toàn bộ batch, chạy lệnh:

```Bash

python evaluate_batch.py \
  --review data/processed/agentB/tokenized_data_500.json \
  --config config/pipeline.yaml \
  --batch tokenized_data_500
```

## Kết quả:

Nếu đạt yêu cầu: Batch được lưu tại ```data/final/tokenized_data_500_accepted.json```.
Nếu chưa đạt: Batch được đánh dấu để QC lại, lưu tại ```data/final/tokenized_data_500_flagged_for_qc.json```.
---

## Tổng Kết Đầu Ra

| File                                | Mô tả                     |
| :---------------------------------- | :------------------------ |
| `tokenized_data_500_sample.json`    | Mẫu cần QC thủ công       |
| `master.json`                       | Gold Set đã được cập nhật |
| `*_accepted.json`                   | Batch đã được chấp nhận   |
| `*_flagged_for_qc.json`             | Batch cần QC lại          |
| `reports/*.json`                    | Báo cáo chi tiết          |