# AGENTS.md

## Mục tiêu

Đây là hệ thống hỏi đáp pháp luật Việt Nam theo hướng multi-agent RAG.
Luồng chính:

```text
Question
-> Legal Understanding
-> Query Planning
-> Hybrid / Exact / Summary retrieval
-> Fusion
-> Graph + Context expansion
-> ColBERT + Cross-encoder rerank
-> Evidence Selection
-> Sufficiency Check
-> Legal Reasoning
-> Verification
-> Submission formatting
-> results.json
```

Giữ thay đổi nhỏ, dễ kiểm tra và tương thích với pipeline hiện tại. Không thêm
abstraction hoặc agent mới nếu chưa có nhu cầu thực tế.

## Cấu trúc

```text
scripts/        Entrypoint xử lý, ingest, inference và submission
src/agents/     Agent và prompt nghiệp vụ
src/chunking/   Tạo retrieval corpus
src/common/     Cấu hình, embedding và BM25
src/data/       Download và chuẩn hóa dữ liệu
src/indexing/   Graph index và Qdrant collection
src/retrieval/  Retrieval, fusion, expansion và rerank
src/schema/     Pydantic schema
src/submission/ Validate, ghi và nén kết quả
tests/          Unit test
```

## Quy tắc làm việc

- Chỉ trả lời bằng tiếng Việt.
- Không sửa code khi người dùng chỉ yêu cầu phân tích.
- Trước khi sửa, đọc schema, module liên quan và test gần nhất.
- Ưu tiên pattern sẵn có, type hint rõ ràng và hàm ngắn.
- Không thay đổi thuật toán retrieval hoặc schema nếu không cần thiết.
- Không xóa hoặc hoàn tác thay đổi của người dùng.
- Không chạy download, embedding, ingest hoặc rebuild Qdrant nếu chưa được yêu
  cầu rõ ràng.
- Sau khi sửa, chạy test tập trung; báo rõ test nào không chạy được.

## Dữ liệu và Git

Không commit:

```text
.env
data/raw/
data/processed/
data/embedding_shards*/
data/qdrant_storage/
results.json
results.zip
*.parquet
*.pt
*.bin
*.safetensors
*.log
```

Không ghi secret thật vào source, README, test hoặc log. Nếu secret từng xuất
hiện công khai, phải coi là đã lộ và thay mới.

## Schema

- Dữ liệu qua ranh giới module phải dùng schema trong `src/schema/`.
- Giữ ổn định `unit_id`, `chunk_id`, `doc_code`, `article` và metadata nguồn.
- Không tự tạo mã văn bản hoặc điều luật.
- Field mới nên optional nếu nguồn dữ liệu không cung cấp đầy đủ.
- Khi đổi schema phải cập nhật test và các consumer trong cùng thay đổi.

## Retrieval

- Collection mặc định: `legal_agent_rag_harrier_idf`.
- Dense vector: `dense`, model `mainguyen9/vietlegal-harrier-0.6b`.
- Sparse vector: `sparse`, dùng BM25 với Qdrant IDF.
- ColBERT và cross-encoder chỉ dùng để rerank candidate, không lưu ColBERT cho
  toàn bộ corpus.
- Deduplicate theo `(doc_code, article)` hoặc `unit_id`.
- Filter hiệu lực và định danh phải được áp dụng trước khi trả kết quả cuối.
- Context expansion chỉ bổ sung đơn vị pháp lý có quan hệ rõ ràng.
- Luôn đóng Qdrant client sau khi dùng.

## Agent

- Agent nhận dependency từ bên ngoài và expose `run(...)`.
- Output truyền sang bước khác phải được Pydantic validate.
- LLM chỉ được trả lời từ selected evidence.
- Evidence gửi lên LLM phải đủ nội dung nhưng có giới hạn hợp lý.
- Retry retrieval/reasoning phải có giới hạn.
- Formatter không được thay đổi ý nghĩa câu trả lời.
- Verification thất bại thì không được export submission.

## Lệnh thường dùng

```bash
conda activate legal_rag_agent
docker compose up -d

python scripts/01_download_data.py
python scripts/02_process_data.py
python scripts/03_run_inference.py
python scripts/04_build_submission.py
python scripts/05_create_payload_indexes.py

pytest
```

Modal và ingest Qdrant là tác vụ tốn tài nguyên. Chỉ chạy theo quy trình trong
README và luôn kiểm tra checkpoint trước khi dùng `--recreate`.

## Submission

- `results.json` là danh sách `SubmissionItem`.
- Validate JSON bằng Pydantic trước khi nén.
- `results.zip` phải phẳng và chỉ chứa `results.json`.
- Citation chỉ được trỏ đến selected evidence.
- Giữ nguyên UTF-8 tiếng Việt và thứ tự kết quả ổn định.

## Kiểm tra sau thay đổi

1. Chạy `python -m py_compile` cho file Python đã sửa.
2. Chạy test gần nhất với module.
3. Chạy nhóm test downstream nếu thay đổi contract.
4. Kiểm tra `git diff --check`.
5. Không xem task hoàn tất nếu còn placeholder rỗng hoặc lệnh tài liệu không
   chạy được.
