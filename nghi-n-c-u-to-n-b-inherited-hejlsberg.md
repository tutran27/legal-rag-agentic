# Nghiên cứu & Đề xuất Cải thiện Retrieval - Legal Agent RAG

## Context
Hệ thống Legal Agent RAG là pipeline multi-agent hỏi đáp pháp luật tiếng Việt. Sau khi đọc toàn bộ source code (60+ Python files), đây là phân tích chi tiết các vấn đề retrieval và đề xuất cải thiện.

---

## Các Vấn Đề Tìm Thấy

### 🔴 Vấn đề nghiêm trọng (Correctness Bugs)

**1. BM25 vector dùng CRC32 hash → collision risk** (`src/common/bm25.py:24`)
- Token được hash bằng `zlib.crc32()` thay vì dùng sparse vector indices thực sự
- CRC32 chỉ 32-bit, có thể collision giữa các token khác nhau → BM25 scores sai
- Qdrant sparse vector cần indices là token IDs từ vocabulary, không phải CRC32 hash
- **Ảnh hưởng**: BM25 search trên Qdrant cho kết quả không chính xác, ảnh hưởng toàn bộ hybrid search và RRF fusion

**2. Hybrid retriever import sai module** (`src/retrieval/hybrid_retriever.py:6`)
- `from src.retrieval.bm25_retriever import colbert_rerank` — import `colbert_rerank` từ file `bm25_retriever.py`, nhưng file này KHÔNG phải BM25 retriever mà là ColBERT reranker
- Gây confusion về responsibility và có thể import cycle nếu code phát triển thêm

**3. `summary_search` chỉ trả về 1 chunk đầu tiên mỗi document** (`src/retrieval/summary_retriever.py:66-69`)
- `seen_docs` set bỏ qua tất cả chunk thuộc cùng doc_id sau chunk đầu tiên
- Nếu document có nhiều điều liên quan, chỉ lấy điều đầu tiên → mất thông tin quan trọng

**4. `graph_search` dùng `unit_id` nhưng graph lưu `doc_id`** (`src/retrieval/graph_retriever.py:28,91-95`)
- Graph lưu edges giữa `doc_id` (từ `legal_edges.parquet`), nhưng `graph_search` lọc kết quả bằng `unit_id` → không bao giờ match được với graph nodes
- Luôn trả về empty results từ graph

**5. `expand_context` không giữ lại seed candidates** (`src/retrieval/context_expander.py:26-38`)
- Chỉ trả về các chunk KHÔNG có trong seed, nhưng KHÔNG bao gồm seeds
- Sau đó `expand_context` results được fusion với `initial_candidates`, nên seeds bị duplicated

### 🟡 Vấn đề hiệu suất / chất lượng

**6. BM25 không chuẩn hóa đúng chuẩn** (`src/common/bm25.py:18-24`)
- `avg_length` mặc định là `None`, khi đó BM25 = raw term frequency (không phảo BM25 thật)
- Thiếu IDF component hoàn toàn → các term phổ biến (là, của, được,...) được weight cao

**7. Cross-encoder batch_size=2 quá nhỏ** (`src/retrieval/cross_encoder_rerank.py:12`)
- Với 60 candidates từ ColBERT, chỉ xử lý 2 samples/batch → 30 forward passes
- Nên dùng ít nhất 8-16 để tận dụng GPU

**8. ColBERT score dùng max-pooling rồi mean** (`src/retrieval/bm25_retriever.py:12`)
- `scores.max(axis=1).mean()` — max theo chiều document vectors, rồi mean theo query vectors
- Đây là MaxSim đúng chuẩn ColBERT, nhưng không có normalization theo query length

**9. Fusion không dedup text content** (`src/retrieval/fusion.py:8-40`)
- RRF fusion dùng `chunk_id` làm key, nhưng các chunk khác nhau có thể có nội dung text rất giống (overlap)
- Có thể trả về nhiều chunk duplicate content

**10. Không có query expansion cho tiếng Việt**
- Chỉ dùng original, legal_rewrite, keyword — không có synonym expansion hoặc stemming tiếng Việt
- Tiếng Việt có nhiều cách viết khác nhau cho cùng một khái niệm pháp lý

**11. Chunk boundary cắt ngang nội dung** (`src/chunking/build_retrieval_corpus.py:52-92`)
- `max_chars=1800` có thể cắt ngang câu/tiêu đề, mất ngữ cảnh
- Không có overlap giữa chunks → thông tin ở boundary bị mất

**12. Không rerank từ nhiều query riêng biệt**
- `run_hybrid` trong `05_run_inference.py` chạy 3 query riêng, nhưng kết quả chỉ được fusion bằng RRF chung
- Nên rerank riêng từng query rồi fusion, hoặc weighted fusion theo query type

### 🟢 Vấn đề kiến trúc / best practices

**13. Thiếu eval metrics cho retrieval**
- `src/eval/retrieval_eval.py` trống — không có cách đánh giá retrieval quality
- Không biếtrecall/precision thực tế trên test set

**14. Evidence selector không dùng context expansion evidence**
- Sau khi expand context, `expand_context` results được fusion nhưng chỉ dùng `final_candidates[:20]` cho evidence selection
- Context-expanded chunks bị thiệt

**15. Graph không có weighted edges**
- `build_graph.py` tạo unweighted MultiDiGraph → graph search không phân biệt được mức độ liên quan
- Tất cả edges có cùng weight = 1.0

---

## Đề Xuất Cải Thiện (Prioritized)

### Priority 1: Fix bugs nghiêm trọng

1. **Fix BM25 sparse vector** — thay CRC32 bằng token-to-index mapping đúng chuẩn, thêm IDF component
2. **Fix graph_search** — match `doc_id` thay vì `unit_id`, hoặc rebuild graph với unit_id
3. **Fix summary_retriever** — trả về tất cả relevant chunks thay vì chỉ 1 chunk/doc
4. **Fix import naming** — đổi tên `bm25_retriever.py` hoặc tách thành `colbert_reranker.py`

### Priority 2: Cải thiện chất lượng retrieval

5. **Thêm Vietnamese text normalization** — stemming/lemmatization tiếng Việt (underthesea, VnCoreNLP)
6. **Tăng cross-encoder batch_size** từ 2 lên 16
7. **Thêm chunk overlap** 200-300 chars giữa các chunks liền kề
8. **Dedup content** trong RRF fusion — loại bỏ chunks có text similarity > 0.9

### Priority 3: Kiến trúc & evaluation

9. **Triển khai retrieval evaluation** — precision@k, recall@k, MRR trên test set
10. **Weighted graph edges** gán weight theo relation type (GUIDES > REFERENCES > AMENDS)
11. **Multi-query rerank** — rerank riêng theo từng query type rồi weighted fusion
12. **Thêm metadata filtering** — filter theo `doc_type`, `status` trước khi rerank

---

## Verification
- Chạy `pytest tests/` để verify existing tests pass
- Chạy `python scripts/05_run_inference.py` để test end-to-end retrieval pipeline
- So sánh retrieval metrics trước/sau fix bằng golden set
