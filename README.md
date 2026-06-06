# Legal Agent RAG

Hệ thống hỏi đáp pháp luật tiếng Việt theo hướng multi-agent RAG. Mục tiêu là truy hồi đúng điều luật/văn bản liên quan, chọn evidence, kiểm chứng citation và xuất kết quả cuối theo định dạng submission.

## Pipeline

```mermaid
flowchart TD
    A[Test Question] --> B[Supervisor Agent]
    B --> C[Legal Understanding Agent]
    C --> D[Query Planner Agent]

    D --> E[Retrieval Router]

    E --> R1[Exact Search<br/>mã văn bản / điều / từ khóa]
    E --> R2[BM25 Search]
    E --> R3[Dense Retrieval<br/>Fine-tuned BGE-M3]
    E --> R4[Sparse Retrieval<br/>BGE-M3 sparse / SPLADE]
    E --> R5[ColBERT / Multi-vector Retrieval]
    E --> R6[Legal Graph Retrieval]
    E --> R7[Summary / RAPTOR / LightRAG Retrieval]

    R1 --> F[Candidate Fusion<br/>Weighted RRF + Vote Boost]
    R2 --> F
    R3 --> F
    R4 --> F
    R5 --> F
    R6 --> F
    R7 --> F

    F --> G[Legal Filter<br/>hiệu lực / lĩnh vực / loại văn bản]
    G --> H[Article-level Reranker]
    H --> I[Context Expansion<br/>Điều → Khoản → Điểm → Văn bản cha]
    I --> J[Evidence Selector Agent]

    J --> K{Evidence đủ mạnh?}

    K -- Không --> D
    K -- Có --> L[Legal Reasoning Agent]

    L --> M[Citation Verifier Agent]
    M --> N{Answer khớp evidence?}

    N -- Không --> L
    N -- Có --> O[Submission Formatter Agent]

    O --> P[JSON Schema Validator]
    P --> Q[results.json]
    Q --> Z[results.zip phẳng]
```

Luồng chính:

```text
Question
-> Legal Understanding
-> Query Planning
-> Multi-retrieval
-> Fusion + Filter + Rerank
-> Evidence Selection
-> Legal Reasoning
-> Citation Verification
-> results.json
-> results.zip
```

## Trạng thái hiện tại

Đã có:

- Download dữ liệu từ Hugging Face.
- Process dữ liệu Pháp điển.
- Process metadata và quan hệ văn bản VBPL.
- Schema cho `LegalDocument`, `LegalArticle`, `LegalEdge`.
- Tạo `legal_units.parquet`.
- Dense indexing lên Qdrant.

Đang cần triển khai tiếp:

- Các retriever: exact, BM25, sparse, ColBERT, graph.
- Fusion, reranker, context expansion.
- Các agent: supervisor, planner, evidence selector, reasoner, verifier, formatter.
- Validator và đóng gói submission.

## Cấu trúc repo

```text
configs/       Cấu hình data/model/retrieval/eval
scripts/       Script chạy pipeline
src/data/      Download, process, chuẩn hóa dữ liệu
src/schema/    Pydantic schema
src/indexing/  Build index
src/retrieval/ Retriever
src/agents/    Multi-agent workflow
src/eval/      Evaluation
src/submission Build, validate, zip kết quả
tests/         Unit tests
```

## Cài đặt

```bash
conda create -n legal_rag_agent python=3.11
conda activate legal_rag_agent
pip install -r requirements.txt
```

Nếu dùng Qdrant:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

## Chạy pipeline dữ liệu

Tải dữ liệu:

```bash
python scripts/01_download_data.py
```

Process Pháp điển:

```bash
python -m src.data.process_phapdien
```

Tạo legal units:

```bash
python -m src.data.build_legal_units
```

Process VBPL:

```bash
python -m src.data.process_vbpl
```

Build dense index:

```bash
python -m src.indexing.build_dense_qdrant
```

## Dữ liệu đầu ra chính

```text
data/processed/phapdien-moj-gov-vn.parquet
data/processed/legal_units.parquet
data/processed/documents.parquet
data/processed/legal_edges.parquet
```

## Test

```bash
pytest
```

Một số test yêu cầu đã có dữ liệu trong `data/processed/`.

## Ghi chú kỹ thuật

`src.data.build_legal_units` hiện cần xử lý thêm:

- `source_links` có thể là `numpy.ndarray`, cần convert sang `list` trước khi `json.dumps`.
- Cần import `Path` từ `pathlib`.

## Roadmap ngắn

1. Fix pipeline tạo `legal_units.parquet`.
2. Tạo `retrieval_corpus.parquet`.
3. Hoàn thiện exact/BM25/dense retriever.
4. Triển khai fusion + reranker + context expansion.
5. Hoàn thiện agent workflow.
6. Sinh, validate và zip `results.json`.
