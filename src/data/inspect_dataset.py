import pandas as pd
from pathlib import Path
import json


def inspect_parquet(path: str | Path, n: int = 1):
    path = Path(path)
    df = pd.read_parquet(path)

    print("=" * 80)
    print("PATH:", path)
    print("ROWS:", len(df))
    print("COLUMNS:")
    for col in df.columns:
        print("-", col, "|", df[col].dtype)

    print("=" * 80)
    print("SAMPLES:")
    for i, row in df.head(n).iterrows():
        print("-" * 80)
        print((row.to_dict()))

import pandas as pd
from pathlib import Path
import sys
import pandas as pd
from pathlib import Path
import sys

def inspect_and_generate_md(parquet_path: str | Path, output_md_path: str | Path):
    parquet_path = Path(parquet_path)
    output_md_path = Path(output_md_path)
    
    # 1. Đọc dữ liệu từ file Parquet
    df = pd.read_parquet(parquet_path)
    num_rows = len(df)
    
    # Lấy ra hàng đầu tiên làm sample để build thông tin
    sample_row = df.iloc[0] if num_rows > 0 else None

    # 2. Định nghĩa cấu trúc các cột quan trọng và ý nghĩa của chúng
    important_columns = {
        "topic_title": "chủ đề",
        "subject_title": "đề mục",
        "article_title": "tiêu đề điều pháp điển",
        "source_note_text": "chứa Điều gốc + mã văn bản",
        "content_text": "nội dung điều",
        "source_url": "URL nguồn"
    }

    # 3. Tiến hành build nội dung file Markdown dựa trên sample dữ liệu thực tế
    md_lines = []
    md_lines.append("# Pháp điển articles\n")
    
    md_lines.append("Path:")
    md_lines.append(f"{parquet_path}\n")
    
    md_lines.append("Rows:")
    md_lines.append(f"{num_rows:,}\n") # Sẽ in ra số hàng thực tế, ví dụ: 64,123
    
    md_lines.append("Important columns:")
    for col, desc in important_columns.items():
        if col in df.columns and sample_row is not None:
            # Lấy giá trị thực tế từ sample để minh họa, giới hạn ký tự nếu quá dài
            actual_val = str(sample_row[col])
            if len(actual_val) > 70:
                actual_val = actual_val[:67] + "..."
            
            md_lines.append(f"- {col}: {desc} (Ví dụ thực tế: `{actual_val}`)")
        else:
            md_lines.append(f"- {col}: {desc}")
    md_lines.append("") # Dòng trống
    
    md_lines.append("Decision:")
    md_lines.append("- Dùng content_text làm text chính")
    md_lines.append("- Dùng source_note_text để parse doc_code và Điều gốc")
    md_lines.append("- Không dùng article_title trực tiếp làm Điều output\n")
    
    # 4. Log chi tiết toàn bộ các trường của Sample Row dưới dạng Markdown để lưu trữ
    if sample_row is not None:
        md_lines.append("Full Sample Details:")
        for col in df.columns:
            val = sample_row[col]
            # Xử lý hiển thị nếu giá trị là chuỗi dài hoặc mảng/đối tượng phức tạp
            if isinstance(val, str) and len(val) > 100:
                md_lines.append(f"- **{col}**:\n  > {val}")
            else:
                md_lines.append(f"- **{col}**: `{val}`")
                
    md_content = "\n".join(md_lines)

    # 5. Ghi nội dung ra file Markdown tương ứng
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    print("=" * 80)
    print(f"Đã trích xuất sample dữ liệu và tạo file markdown tại: {output_md_path}")
    print("=" * 80)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Cú pháp: python script.py <đường_dẫn_parquet> <đường_dẫn_md_đầu_ra>")
        sys.exit(1)
        
    inspect_and_generate_md(sys.argv[1], sys.argv[2])