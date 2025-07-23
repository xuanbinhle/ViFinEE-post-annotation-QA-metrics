#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os

def filter_json_by_keys(reference_file_path, target_file_path, output_file_path):
    """
    Lọc các sample từ target file dựa trên các key có trong reference file
    
    Args:
        reference_file_path (str): Đường dẫn đến file chứa các key cần lấy
        target_file_path (str): Đường dẫn đến file nguồn cần lọc
        output_file_path (str): Đường dẫn file output
    """

    try:
        # Đọc file reference để lấy danh sách keys
        print(f"Đang đọc file reference: {reference_file_path}")
        with open(reference_file_path, 'r', encoding='utf-8') as f:
            reference_data = json.load(f)

        # Lấy tất cả keys từ reference file
        reference_keys = set(reference_data.keys())
        print(f"Tìm thấy {len(reference_keys)} keys trong file reference")

        # Đọc file target
        print(f"Đang đọc file target: {target_file_path}")
        with open(target_file_path, 'r', encoding='utf-8') as f:
            target_data = json.load(f)

        print(f"Tìm thấy {len(target_data)} samples trong file target")

        # Lọc các samples có key trùng với reference
        filtered_data = {}
        for key in reference_keys:
            if key in target_data:
                filtered_data[key] = target_data[key]

        print(f"Đã lọc được {len(filtered_data)} samples khớp với reference keys")

        # Tạo thư mục output nếu chưa tồn tại
        output_dir = os.path.dirname(output_file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Ghi file output
        print(f"Đang ghi file output: {output_file_path}")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)

        print("Hoàn thành!")
        print(f"- Keys trong reference file: {len(reference_keys)}")
        print(f"- Keys trong target file: {len(target_data)}")
        print(f"- Keys được lọc ra: {len(filtered_data)}")

        # Hiển thị một số keys mẫu
        if filtered_data:
            sample_keys = list(filtered_data.keys())[:5]
            print(f"- Một số keys mẫu: {sample_keys}")

    except FileNotFoundError as e:
        print(f"Lỗi: Không tìm thấy file - {e}")
    except json.JSONDecodeError as e:
        print(f"Lỗi: File JSON không hợp lệ - {e}")
    except Exception as e:
        print(f"Lỗi không xác định: {e}")

def main():
    # Đường dẫn file với định dạng Linux
    reference_file = "/mnt/d/ViFinEE-post-annotation-QA-metrics/data/processed/human/tokenized_data_1000_sample.json"
    target_file = "/mnt/d/ViFinEE-post-annotation-QA-metrics/tokenized_data_1000.json"
    output_file = "/mnt/d/ViFinEE-post-annotation-QA-metrics/tokenized_data_1000_filtered.json"
    
    print("=== BẮT ĐẦU LỌC DỮ LIỆU JSON ===")
    print(f"Reference file: {reference_file}")
    print(f"Target file: {target_file}")
    print(f"Output file: {output_file}")
    print()
    
    filter_json_by_keys(reference_file, target_file, output_file)

if __name__ == "__main__":
    main()
