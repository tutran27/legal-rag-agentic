# CLAUDE_LATENCY_PLAN.md

> Phân tích latency toàn diện cho hệ thống Legal Agent RAG.
> Ngày phân tích: 2026-06-15.
> Trạng thái: Chỉ phân tích, **chưa sửa code**.

---

## 1. Tổng quan Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        InferencePipeline.run()                          │
│                                                                         │
│  ① LLM: Understanding + Query Planning (combined)     ~2048 max_tokens │
│     │                                                                   │
│  ② Hybrid Retrieval (Qdrant batch)                    1-3 calls        │
│     ├─ Dense embedding (SentenceTransformer)           batch encode     │
│     ├─ Sparse BM25 vector (per query)                 inline           │
│     └─ Qdrant query_batch_points                      gRPC/HTTP        │
│     │                                                                   │
│  ③ Exact Search (conditional)                         Parquet scan     │
│  ④ Summary Search (default: OFF)                      Parquet scan     │
│  ⑤ Weighted RRF Fusion                                in-memory        │
│     │                                                                   │
│  ⑥ Graph Expansion (conditional)                      pickle load      │
│  ⑦ Context Expansion (conditional)                    Parquet scan     │
│  ⑧ Expanded RRF Fusion                                in-memory        │
│     │                                                                   │
│  ⑨ ColBERT Rerank: 100 → 60                           batch_size=16    │
│  ⑩ Cross-Encoder Rerank: 60 → 40                     batch_size=16    │
│     │                                                                   │
│  ⑪ LLM: Evidence Selection + Sufficiency (combined)   ~1536 max_tokens │
│     │                                                                   │
│  ⑫ LLM: Reasoning                                     ~1536 max_tokens │
│     │                                                                   │
│  ⑬ LLM: Verification                                  ~1536 max_tokens │
│     │                                                                   │
│  ⑭ [If fail] LLM: Reasoner revision + Verification    +3072 tokens     │
│     │                                                                   │
│  ⑮ Submission Formatter + Validation                  no LLM           │
└─────────────────────────────────────────────────────────────────────────┘
```

**Số LLM calls mỗi câu hỏi:** 4 (best case) → 6 (worst case, verification fail)

---

## 2. Các Bottleneck Chi Tiết

### 🔴 BOTTLENECK #1: LLM Endpoint Latency
**Mức độ: RẤT LỚN | Ước tính: 60-80% tổng latency**

#### Vấn đề

| Hạng mục | File | Dòng | Chi tiết |
|----------|------|------|----------|
| Timeout quá lớn | `src/generation/endpoint.py` | 33 | `timeout=600s` mặc định, không retry |
| Không connection pooling | `src/generation/endpoint.py` | 34 | `requests.Session()` nhưng không config `HTTPAdapter` |
| max_new_tokens thừa | `src/agents/query_planner.py` | 199 | `max_new_tokens=2048` cho JSON nhỏ |
| max_new_tokens thừa | `src/agents/reasoner.py` | 111 | `max_new_tokens=1536` cho câu trả lời <250 từ |
| max_new_tokens thừa | `src/agents/evidence_selector.py` | 180 | `max_new_tokens=1536` cho JSON selection |
| max_new_tokens thừa | `src/agents/verifier.py` | 81 | `max_new_tokens=1536` cho JSON ngắn |
| Không caching | `src/generation/endpoint.py` | 96-107 | `call_llm_json()` không có cache layer |

#### Đề xuất

1. **Giảm `max_new_tokens`** phù hợp thực tế:
   - Query Planner: `2048 → 512` (chỉ cần JSON với 3 queries + filters)
   - Reasoner: `1536 → 512` (câu trả lời <250 từ ≈ 300-400 tokens)
   - Verifier: `1536 → 256` (chỉ cần JSON với boolean + short strings)
   - Evidence Selector: `1536 → 512` (JSON với selected/rejected lists)

2. **Thêm response caching**:
   ```python
   # Hash-based cache key: hash(question + system_prompt + temperature)
   # Lưu vào disk (JSON) hoặc in-memory (LRU)
   # Tiết kiệm 100% latency cho câu hỏi đã cached
   ```

3. **Connection pooling**:
   ```python
   adapter = requests.adapters.HTTPAdapter(
       pool_connections=4,
       pool_maxsize=8,
       max_retries=requests.adapters.Retry(total=2, backoff_factor=0.5),
   )
   session.mount("https://", adapter)
   session.mount("http://", adapter)
   ```

4. **Giảm timeout + retry**: `600s → 120s` với 2 retries, backoff 1s.

5. **Streaming + early exit** (nếu endpoint hỗ trợ): Parse JSON incrementally thay vì chờ full response.

---

### 🔴 BOTTLENECK #2: ColBERT Rerank
**Mức độ: LỚN | Ước tính: 15-25% tổng latency**

#### Vấn đề

| Hạng mục | File | Dòng | Chi tiết |
|----------|------|------|----------|
| Batch size nhỏ | `src/common/config.py` | 45 | `colbert_batch_size=16` |
| Encode lại mỗi query | `src/retrieval/colbert_reranker.py` | 26-32 | Encode query + 100 candidates mỗi lần |
| Model nặng | `src/common/embedding.py` | 42-47 | BGEM3FlagModel (BGE-M3) cho ColBERT vectors |
| Không pre-compute | `src/retrieval/colbert_reranker.py` | 26 | Document ColBERT vectors không được cache |

#### Đề xuất

1. **Tăng `colbert_batch_size`**: `16 → 32` hoặc `64` (tùy VRAM).
2. **Pre-compute ColBERT embeddings**: Lưu document ColBERT vectors vào Qdrant (multi-vector payload), chỉ encode query → dot product.
3. **Giản candidates trước ColBERT**: Chỉ lấy top 50 từ fusion thay vì 100.
4. **Dùng model nhỹ hơn**: ColBERT-lite hoặc distilled token encoder.

---

### 🔴 BOTTLENECK #3: Cross-Encoder Rerank
**Mức độ: LỚN | Ước tính: 10-20% tổng latency**

#### Vấn đề

| Hạng mục | File | Dòng | Chi tiết |
|----------|------|------|----------|
| Batch size nhỏ | `src/common/config.py` | 46 | `cross_encoder_batch_size=16` |
| Model 0.6B | `src/retrieval/cross_encoder_rerank.py` | 6 | Qwen3-Reranker-0.6B |
| 40 pairs/query | `src/schema/agent_schemas.py` | 63 | `top_k_cross_encoder=40` |

#### Đề xuất

1. **Tăng `cross_encoder_batch_size`**: `16 → 32-64`.
2. **Dùng model nhỹ hơn**: `ms-marco-MiniLM-L-6-v2` (90MB, nhanh hơn 5-10x) hoặc `flash-reranker`.
3. **Giảm `top_k_cross_encoder`**: `40 → 20-25` (nếu chất lượng không giảm đáng kể).

---

### 🟡 BOTTLENECK #4: Model Loading (One-time)
**Mức độ: TRUNG BÌNH | Chỉ 1 lần khi khởi tạo**

#### Vấn đề

| Hạng mục | File | Dòng | Chi tiết |
|----------|------|------|----------|
| Download mỗi lần | `src/common/embedding.py` | 15-19 | `snapshot_download()` gọi HF Hub mỗi init |
| Load 3 models | `src/pipeline/inference_pipeline.py` | 76-81 | Dense + ColBERT + Cross-encoder |
| Không lazy load | `src/pipeline/inference_pipeline.py` | 76-81 | Load dù chưa cần |

#### Đề xuất

1. **Cache model local**: Kiểm tra local path trước, chỉ download khi thiếu.
2. **Lazy load**: Chỉ load model khi flag `use_*` = True trong retrieval plan.
3. **Warmup ở background thread** khi server start.

---

### 🟡 BOTTLENECK #5: Qdrant Client Per Request
**Mức độ: TRUNG BÌNH | Ước tính: 100-500ms mỗi query**

#### Vấn đề

| Hạng mục | File | Dòng | Chi tiết |
|----------|------|------|----------|
| Tạo client mới | `src/retrieval/hybrid_retriever.py` | 74-78 | `QdrantClient()` mỗi lần search |
| Đóng client | `src/retrieval/hybrid_retriever.py` | 86 | `client.close()` mỗi lần |
| gRPC wasted | `src/retrieval/hybrid_retriever.py` | 76 | `prefer_grpc=True` nhưng connection mới |
| 3 calls fallback | `src/pipeline/inference_pipeline.py` | 126-158 | Có thể gọi hybrid 3 lần |

#### Đề xuất

1. **Shared QdrantClient**: Tạo 1 client ở `InferencePipeline.__init__()`, inject vào `hybrid_search_batch`.
2. **Connection reuse**: gRPC channel giữ nguyên giữa các requests.
3. **Giảm fallback**: Cải thiện filter logic hoặc dùng OR conditions.

---

### 🟡 BOTTLENECK #6: Graph Search — Pickle Load Mỗi Query
**Mức độ: TRUNG BÌNH | Ước tính: 50-200ms mỗi query**

#### Vấn đề

| Hạng mục | File | Dòng | Chi tiết |
|----------|------|------|----------|
| Load từ disk | `src/retrieval/graph_retriever.py` | 43-44 | `pickle.load(file)` mỗi query |
| NetworkX overhead | `src/retrieval/graph_retriever.py` | 69-86 | `out_edges` + `in_edges` traversal |

#### Đề xuất

1. **Cache graph trong memory**: Load 1 lần khi init pipeline.
2. **Dùng adjacency dict** thay vì NetworkX nếu chỉ cần edge traversal.
3. **Pre-compute related docs** cho mỗi doc_id phổ biến.

---

### 🟡 BOTTLENECK #7: Context Expansion — Parquet Scan Mỗi Query
**Mức độ: TRUNG BÌNH | Ước tính: 50-200ms mỗi query**

#### Vấn đề

| Hạng mục | File | Dòng | Chi tiết |
|----------|------|------|----------|
| Scan mỗi query | `src/retrieval/context_expander.py` | 23-26 | `ds.dataset()` + `to_table(filter=...)` |
| Không index | `src/retrieval/context_expander.py` | 25 | `unit_id` không có index trong parquet |
| Full columns | `src/retrieval/context_expander.py` | 26 | `to_pylist()` trả về tất cả columns |

#### Đề xuất

1. **Dùng DuckDB với index** thay vì PyArrow dataset scan.
2. **Cache dataset reference** thay vì tạo mới.
3. **Chỉ select columns cần thiết**.

---

### 🟢 BOTTLENECK #8: Sequential LLM Calls
**Mức độ: TRUNG BÌNH | Ước tính: Có thể giảm 20-30% nếu parallelize được**

#### Vấn đề

| Hạng mục | File | Dòng | Chi tiết |
|----------|------|------|----------|
| Tuần tự hoàn toàn | `src/pipeline/inference_pipeline.py` | 294-383 | 4-6 LLM calls chạy tuần tự |
| Phụ thuộc dữ liệu | `src/pipeline/inference_pipeline.py` | 294-383 | Mỗi call cần kết quả call trước |

#### Đề xuất

1. **Batch nhiều câu hỏi**: Encode + gọi LLM cho nhiều queries cùng lúc.
2. **Async LLM calls**: Dùng `asyncio.gather()` cho các batch câu hỏi độc lập.
3. **Speculative decoding**: Chạy Reasoner với top-2 evidence sets cùng lúc.

---

### 🟢 BOTTLENECK #9: Redundant Fallback Retrieval
**Mức độ: NHỸ-TRUNG BÌNH | 0-2x hybrid latency**

#### Vấn đề

| Hạng mục | File | Dòng | Chi tiết |
|----------|------|------|----------|
| 3 lần gọi hybrid | `src/pipeline/inference_pipeline.py` | 126-158 | Fallback: full filter → no taxonomy → is_current only |
| Mỗi lần encode lại | `src/retrieval/hybrid_retriever.py` | 24-28 | `embed_dense(queries)` mỗi lần gọi |

#### Đề xuất

1. **Gộp thành 1 query** với filter linh hoạt (OR conditions).
2. **Cache dense vectors** của queries đã encode.
3. **Bỏ fallback** nếu data đã được index đúng.

---

### 🟢 BOTTLENECK #10: Batch Inference — Chạy Từng Câu
**Mức độ: NHỈ | Ảnh hưởng throughput, không ảnh hưởng latency per-query**

#### Vấn đề

| Hạng mục | File | Dòng | Chi tiết |
|----------|------|------|----------|
| Tuần tự | `scripts/03_run_inference.py` | 60-95 | `for item in questions: pipeline.run()` |
| Không batch embedding | `scripts/03_run_inference.py` | 69 | Encode 1 query/lần |

#### Đ�ề xuất

1. **Batch embedding**: Encode N queries cùng lúc → 1 Qdrant batch call.
2. **ThreadPoolExecutor**: Chạy N câu hỏi song song (giới hạn concurrency).
3. **Async pipeline**: Dùng `asyncio` cho non-blocking I/O.

---

## 3. Ma trận Ưu Tiên

| Ưu tiên | Bottnerneck | Tiết kiệm ước tính | Độ khó | Tác động |
|---------|-------------|-------------------|--------|----------|
| 🥇 #1 | Giảm max_new_tokens LLM | 30-50% LLM time | Dễ | Cao |
| 🥇 #2 | Shared QdrantClient | 100-500ms/query | Dễ | Cao |
| 🥇 #3 | Cache graph trong memory | 50-200ms/query | Dễ | Cao |
| 🥈 #4 | Tăng batch_size (ColBERT + CE) | 20-40% rerank time | Dễ | Trung bình |
| 🥈 #5 | LLM response caching | 100% cho cache hit | Trung bình | Cao |
| 🥈 #6 | Pre-compute ColBERT embeddings | 50-70% ColBERT time | Khó | Cao |
| 🥉 #7 | Dùng reranker nhỹ hơn | 30-50% CE time | Trung bình | Trung bình |
| 🥉 #8 | Batch inference + async | 2-5x throughput | Trung bình | Cao (throughput) |
| 🥉 #9 | Giảm fallback retrieval | 0-2x hybrid time | Dễ | Thấp |
| 🥉 #10 | Streaming LLM + early exit | 10-30% LLM time | Trung bình | Trung bình |

---

## 4. Quick Wins (Dễ làm, tác động lớn)

### 4.1 Giảm max_new_tokens

| Agent | File | Hiện tại | Đề xuất | Giải thích |
|-------|------|----------|---------|------------|
| Query Planner | `src/agents/query_planner.py:199` | 2048 | **512** | JSON với 3 queries + filters |
| Evidence Selector | `src/agents/evidence_selector.py:180` | 1536 | **512** | JSON selected/rejected |
| Reasoner | `src/agents/reasoner.py:111` | 1536 | **512** | Câu trả lời <250 từ |
| Verifier | `src/agents/verifier.py:81` | 1536 | **256** | JSON ngắn, boolean |

### 4.2 Shared QdrantClient

```python
# InferencePipeline.__init__()
self.qdrant_client = QdrantClient(
    url=settings.qdrant_url,
    prefer_grpc=True,
    timeout=settings.qdrant_timeout,
)

# Inject vào hybrid_search_batch(client=self.qdrant_client)
# Không cần tạo/đóng client mỗi query
```

### 4.3 Cache Graph trong Memory

```python
# InferencePipeline.__init__()
with Path(graph_path).open("rb") as f:
    self.graph = pickle.load(f)

# Truyền self.graph vào graph_search(graph=self.graph)
# Không cần load từ disk mỗi query
```

### 4.4 Tăng batch_size

| Config | File | Hiện tại | Đề xuật | Điều kiện |
|--------|------|----------|---------|-----------|
| `colbert_batch_size` | `src/common/config.py:45` | 16 | **32** | VRAM ≥ 8GB |
| `cross_encoder_batch_size` | `src/common/config.py:46` | 16 | **32** | VRAM ≥ 8GB |

### 4.5 Connection Pooling cho LLM Client

```python
# EndpointLLMClient.__init__()
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

retry = Retry(total=2, backoff_factor=0.5, status_forcelist=[502, 503, 504])
adapter = HTTPAdapter(pool_connections=4, pool_maxsize=8, max_retries=retry)
self.session = requests.Session()
self.session.mount("https://", adapter)
self.session.mount("http://", adapter)
```

---

## 5. Kịch Bản Cải Thiện Theo Giai Đoạn

### Giai đoạn 1: Quick Wins (1-2 ngày)
- Giảm max_new_tokens
- Shared QdrantClient
- Cache graph trong memory
- Tăng batch_size
- Connection pooling + retry cho LLM client

**Ước tính cải thiện: 40-60% latency**

### Giai đoạn 2: Caching Layer (2-3 ngày)
- LLM response cache (disk-based, hash key)
- Dense embedding cache cho queries phổ biến
- Parquet dataset cache (DuckDB index)

**Ước tính cải thiện: 50-70% latency (với cache hit)**

### Giai đoạn 3: Model Optimization (3-5 ngày)
- Pre-compute ColBERT embeddings vào Qdrant
- Thay cross-encoder nhỹ hơn
- Lazy model loading

**Ước tính cải thiện: 60-80% latency**

### Giai đoạn 4: Async + Batch (3-5 ngày)
- Async LLM client (aiohttp)
- Batch inference cho nhiều câu hỏi
- Streaming LLM response

**Ước tính cải thiện: 3-5x throughput, 70-85% latency per-query**

---

## 6. Các Metrics Cần Đo

Để đánh giá hiệu quả sau khi cải thiện, cần đo:

| Metric | Cách đo | Mục tiêu |
|--------|---------|----------|
| **Per-query latency** | `time.perf_counter()` tổng | Giảm 50%+ |
| **LLM call latency** | Tổng thời gian 4-6 LLM calls | Giảm 40%+ |
| **Rerank latency** | ColBERT + Cross-encoder time | Giảm 30%+ |
| **Retrieval latency** | Qdrant + exact + graph + context | Giảm 20%+ |
| **Throughput** | Số câu hỏi/giờ (batch mode) | Tăng 3x+ |
| **Cache hit rate** | % queries trong cache | >30% cho repeated queries |
| **P50/P95 latency** | Phân phối latency | P95 < 2x P50 |

---

## 7. Lưu Ý Quan Trọng

1. **Không thay đổi kết quả**: Mọi cải thiện latency không được làm giảm chất lượng câu trả lời.
2. **Test sau mỗi thay đổi**: Chạy `pytest -q` để đảm bảo không break logic.
3. **Benchmark trước/sau**: Ghi lại latency trước khi bắt đầu để so sánh.
4. **GPU memory**: Tăng batch_size cần kiểm tra VRAM usage.
5. **Modal endpoint**: Một số cải thiện (streaming, timeout) phụ thuộc vào endpoint capability.

---

*Phân tích được thực hiện bởi OWL — dựa trên đọc toàn bộ source code, không sửa đổi bất kỳ file nào.*
