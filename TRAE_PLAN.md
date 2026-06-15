# TRAE Plan: Giam Latency He Thong Legal Agent RAG

## 1. Muc tieu

Tai lieu nay tong hop phan tich latency cua he thong theo source hien tai va de
xuat huong toi uu theo thu tu uu tien. Pham vi cua tai lieu la inference online,
khong bao gom ingest hay rebuild index.

Muc tieu:

- Giam thoi gian tra loi moi cau hoi.
- Giam tail latency, dac biet cac cau bi retry verification.
- Giu thay doi nho, de kiem tra, phu hop voi pipeline hien tai.
- Khong thay doi schema hoac thuat toan retrieval neu chua can thiet.

## 2. Entry Point Va Luong Chay Thuc Te

Entry point inference dang dung:

- `scripts/03_run_inference.py`

Pipeline online thuc te:

1. Khoi tao `InferencePipeline`
2. `QueryPlannerAgent.run_combined(question)`
3. `_retrieve(...)`
   - hybrid retrieval
   - exact retrieval
   - summary retrieval neu bat
   - initial fusion
4. `_expand(...)`
   - graph expansion
   - context expansion
   - expanded fusion
5. `_rerank(...)`
   - ColBERT rerank
   - cross-encoder rerank
6. `EvidenceSelectorAgent.run_with_sufficiency(...)`
7. `ReasonerAgent.run(...)`
8. `VerificationAgent.run(...)`
9. Neu verification fail:
   - reasoner revision
   - final verification
10. `SubmissionFormatterAgent().run(...)`

## 3. Tong Quan Nhan Dinh

Latency cao hien tai chu yeu den tu 3 nhom:

1. Chuoi goi LLM noi tiep qua endpoint remote.
2. Rerank nang bang ColBERT va cross-encoder tren nhieu candidate.
3. I/O lap lai tren parquet va pickle cho graph/context/exact/summary retrieval.

Ngoai ra con co fan-out retrieval rat lon do retrieval plan mac dinh dang bat gan
nhu toan bo nhanh.

## 4. Phan Ra Bottleneck Theo Stage

### 4.1. Model Load

File chinh:

- `src/pipeline/inference_pipeline.py`
- `src/common/embedding.py`

Hien trang:

- Khoi tao `InferencePipeline` load 3 model: dense, ColBERT, cross-encoder.
- Neu chay batch thi model duoc load 1 lan va tai su dung.
- Neu chay single query thi cold start rat dat.

Tac dong:

- Anh huong cao voi lan chay dau.
- Anh huong thap hon voi batch warm run.

Danh gia:

- Khong phai bottleneck online lon nhat sau khi process da warm.
- Van can toi uu neu muon phuc vu real-time hoac service da tien trinh ngan.

### 4.2. LLM Remote Calls

File chinh:

- `src/generation/endpoint.py`
- `src/agents/query_planner.py`
- `src/agents/evidence_selector.py`
- `src/agents/reasoner.py`
- `src/agents/verifier.py`
- `src/pipeline/inference_pipeline.py`

Hien trang:

- Tat ca buoc LLM deu goi qua HTTP endpoint remote.
- So lan goi toi thieu tren moi cau:
  - 1 lan planning
  - 1 lan evidence selection + sufficiency
  - 1 lan reasoning
  - 1 lan verification
- Neu verification fail thi co them 2 lan:
  - answer revision
  - final verification

Tac dong:

- Day la bottleneck lon nhat hoac top 2 lon nhat tuy tung query.
- Tail latency bi doi rat manh khi roi vao nhanh retry.

Rui ro hien tai:

- Phu thuoc network.
- Phu thuoc do on dinh cua endpoint remote.
- Prompt va payload dai lam tang token processing time.

### 4.3. Retrieval Fan-out

File chinh:

- `src/agents/query_planner.py`
- `src/schema/agent_schemas.py`
- `src/pipeline/inference_pipeline.py`

Hien trang:

- Prompt planner mac dinh bat:
  - `use_exact = true`
  - `use_bm25 = true`
  - `use_dense = true`
  - `use_sparse = true`
  - `use_colbert = true`
  - `use_cross_encoder = true`
  - `use_graph = true`
  - `use_context = true`
- Top-k cung dang cao:
  - exact 50
  - bm25/dense/sparse 100
  - colbert 60
  - cross-encoder 40
  - graph 50

Tac dong:

- Moi cau hoi mac dinh di qua rat nhieu nhanh.
- So candidate truoc rerank lon, keo theo compute va payload lon.

Danh gia:

- Day la diem co the giam latency nhanh nhat voi it rui ro neu tinh chinh hop ly.

### 4.4. Hybrid Retrieval + Qdrant

File chinh:

- `src/retrieval/hybrid_retriever.py`
- `src/common/embedding.py`

Hien trang:

- Moi query planner tao 3 query variants:
  - original
  - legal_rewrite
  - keyword
- Hybrid retrieval embed dense cho toan bo variants.
- Sau do goi `query_batch_points` vao Qdrant.
- Dang tra `with_payload=True` ngay o giai doan retrieval.
- Qdrant client duoc tao va dong trong tung lan goi.

Tac dong:

- Embed query khong qua lon, nhung tong retrieval payload co the lon.
- Overhead tao/dong client khong phai lon nhat, nhung la chi phi thua.

Rui ro:

- Payload lon gay ton network/serialization.
- Neu Qdrant tu xa thi chi phi cang ro.

### 4.5. ColBERT Rerank

File chinh:

- `src/retrieval/colbert_reranker.py`

Hien trang:

- Encode `[query + tat ca candidate.text]`.
- Tinh token-level similarity cho tung candidate.
- Chi sau khi cham xong moi cat `top_k`.

Tac dong:

- La hotspot compute rat nang.
- Chi phi tang manh khi so candidate lon va text dai.

Danh gia:

- Day la diem toi uu co ROI rat cao.

### 4.6. Cross-encoder Rerank

File chinh:

- `src/retrieval/cross_encoder_rerank.py`

Hien trang:

- Chay `model.predict([(query, candidate.text) ...])`.
- Score pairwise tren toan bo output tu ColBERT.
- Sau do moi cat `top_k`.

Tac dong:

- Thuong la compute hotspot cung cap hoac lon hon ColBERT tuy model/device.
- Dac biet dat khi chay CPU hoac batch size chua toi uu.

Danh gia:

- Nen coi day la buoc co dieu kien, khong nen bat mac dinh cho moi query.

### 4.7. Graph Expansion

File chinh:

- `src/retrieval/graph_retriever.py`

Hien trang:

- Moi query mo file `legal_graph.pkl`.
- Dung `pyarrow.dataset` scan `retrieval_corpus.parquet`.
- Sau do loc va materialize rows lien quan.

Tac dong:

- I/O lap lai tren moi query.
- Ton chi phi doc graph va quet parquet.

Danh gia:

- Neu graph khong giup nhieu cho da so cau hoi, day la nhanh nen gating.

### 4.8. Context Expansion

File chinh:

- `src/retrieval/context_expander.py`

Hien trang:

- Moi query tao `ds.dataset(...)` tu parquet.
- Loc theo `unit_id`.
- Lay chunk truoc/sau dua tren `part_index`.

Tac dong:

- Them mot lan scan parquet nua tren hot path.
- Co gia tri cho quality, nhung latency tang ro neu data nam tren dia chua cham.

Danh gia:

- Nen bat co dieu kien, khong nen coi la mandatory.

### 4.9. Exact Retrieval

File chinh:

- `src/retrieval/exact_retriever.py`

Hien trang:

- Scan parquet theo `doc_code/article`.
- Gia tri cao khi query neu ro so hieu van ban hoac dieu luat.
- Gia tri thap khi query mo ho.

Danh gia:

- Nen la nhanh uu tien theo dieu kien, khong can bat vo toi va.

### 4.10. Summary Retrieval

File chinh:

- `src/retrieval/summary_retriever.py`

Hien trang:

- Doc `documents.parquet`, sau do scan `retrieval_corpus.parquet`.
- Mac dinh hien tai thuong tat.

Danh gia:

- Khong phai bottleneck hien tai neu van tat.
- Neu bat trong tuong lai, can xem xet cache vi co doc file lon.

### 4.11. Evidence Selection + Sufficiency

File chinh:

- `src/agents/evidence_selector.py`

Hien trang:

- Da gom 2 buoc thanh 1 lan LLM, day la diem tot.
- Van gui candidate text kha dai vao LLM.

Danh gia:

- Chua phai diem te nhat, nhung payload input van co the toi uu.

### 4.12. Reasoning + Verification

File chinh:

- `src/agents/reasoner.py`
- `src/agents/verifier.py`

Hien trang:

- Reasoner nhan selected evidence full text.
- Verifier lai nhan answer + evidence full text.
- Neu verifier fail thi lap them 2 buoc nua.

Tac dong:

- Rat ton do vi vua co network vua co token processing.
- Tail latency doi len manh khi verifier hay fail.

## 5. Xep Hang Uu Tien Bottleneck

Theo danh gia tac dong thuc te:

### Muc 1: Can uu tien xu ly som nhat

1. So lan goi LLM qua endpoint remote.
2. ColBERT rerank tren nhieu candidate.
3. Cross-encoder rerank tren nhieu candidate.
4. Retrieval plan mac dinh bat qua nhieu nhanh.

### Muc 2: Tac dong lon, nen xu ly tiep theo

5. Graph expansion mo pickle + scan parquet moi query.
6. Context expansion scan parquet moi query.
7. Payload retrieval va payload gui sang LLM qua lon.

### Muc 3: Tac dong trung binh

8. Tao/dong Qdrant client tung query.
9. Cold start model load.

## 6. Nguyen Tac Toi Uu De Xuat

Khi toi uu nen theo cac nguyen tac sau:

- Giam fan-out truoc, sau do moi toi uu compute.
- Gating theo loai cau hoi truoc khi bat full pipeline.
- Cache doc du lieu lap lai trong process.
- Chi dung rerank nang cho shortlist nho.
- Giu schema va pipeline chinh on dinh.
- Do latency theo stage truoc va sau moi thay doi.

## 7. Quick Wins De Xuat

Day la nhom thay doi co kha nang giam latency ro nhat va rui ro thap nhat.

### 7.1. Giam top-k mac dinh

De xuat:

- Giam `top_k_bm25`, `top_k_dense`, `top_k_sparse`
- Giam `top_k_colbert`
- Giam `top_k_cross_encoder`
- Giam `top_k_graph`

Ly do:

- Dang lay candidate qua rong so voi nhu cau cua buoc cuoi.
- Moi buoc rerank va expansion deu tang chi phi theo so candidate.

Ky vong:

- Giam compute va giam payload rat nhanh.

Rui ro:

- Neu giam qua manh co the giam recall.
- Nen tune theo metrics thay vi cat cung.

### 7.2. Tat mac dinh `cross_encoder` cho query don gian

Chi bat khi:

- Top hybrid sau fusion sat diem nhau.
- Cau hoi mo, nhieu nghia, can precision cao.
- Hybrid va exact cho tin hieu mau thuan.

Khong bat khi:

- Query co `doc_code`.
- Query co `Dieu X`.
- Exact retrieval da tra ve ket qua ro.
- Top 1 va top 2 chenh diem lon.

Ky vong:

- Cat duoc 1 compute hotspot lon.

### 7.3. Tat mac dinh `graph` va `context` cho da so query

Chi bat khi:

- Cau hoi can van ban lien quan, van ban huong dan, van ban sua doi.
- Hybrid top dau thieu can cu hoac phan tan qua nhieu van ban.

Ky vong:

- Cat duoc nhieu I/O tren parquet va pickle.

### 7.4. Rut gon payload gui vao LLM

De xuat:

- Selection chi gui text ngan hon.
- Reasoning chi gui phan can thiet nhat cua evidence.
- Verification chi gui evidence da duoc cat gon va co tinh lien quan cao.

Ky vong:

- Giam token latency.
- Giam network payload.

### 7.5. Reuse Qdrant client theo vong doi pipeline

De xuat:

- Tao Qdrant client 1 lan trong pipeline hoặc service scope.
- Dong client khi ket thuc process, khong phai moi query.

Ky vong:

- Giam connection overhead.

## 8. Thay Doi Trung Han Nen Lam

### 8.1. Cache graph va parquet handles trong RAM

De xuat:

- Load `legal_graph.pkl` 1 lan.
- Tai su dung `pyarrow.dataset` hoac preloaded structures.
- Co cache cho summary documents.

Ky vong:

- Cat I/O lap lai moi query.

Luu y:

- Can kiem soat vong doi process va memory.

### 8.2. Gating retrieval theo loai query

Phan loai query:

- Query exact lookup
- Query legal condition/procedure
- Query mo ho / can mo rong context

De xuat hanh vi:

- Exact lookup: uu tien exact + hybrid nhe, bo graph/context/cross-encoder.
- Condition question ro rang: hybrid + colbert nho, co the bo graph.
- Query mo ho: moi bat graph/context va cross-encoder.

Ky vong:

- Giam thoi gian trung binh ma van giu chat luong cho query kho.

### 8.3. Rerank cascade that chat

De xuat:

- Fusion -> top nho -> ColBERT -> top nho hon nua -> cross-encoder.

Thay vi:

- Fusion -> ColBERT tren tap lon -> cross-encoder tren tap con lon.

Ky vong:

- Giam compute rat dang ke.

### 8.4. Tach model LLM theo tac vu

De xuat:

- Planning, selection, verification dung model nhanh hon.
- Reasoning moi dung model manh hon neu can.

Ky vong:

- Giam tong thoi gian remote inference.

Rui ro:

- Can benchmark chat luong tung agent.

## 9. Thay Doi Kien Truc Neu Can

Chi nen lam sau khi da thu quick wins va trung han.

### 9.1. Bo hoac giam verification full LLM

Huong:

- Chi verification bang LLM khi answer co dau hieu rui ro.
- Cau tra loi ngan, cau hoi don gian co the dung rule/check nhe hon truoc.

### 9.2. Precompute hoac index them cho graph/context

Huong:

- Tao map phuc vu tra hang xom theo `unit_id`, `doc_id`, `part_index`.
- Tranh scan parquet moi query.

### 9.3. Dieu phoi retrieval song song neu co service mode

Huong:

- O cap kien truc service, co the overlap mot so buoc khong phu thuoc truc tiep.
- Tuy nhien can can nhac do phuc tap va tinh de debug.

## 10. Thu Tu Uu Tien Trien Khai De Xuat

### Pha 1: It rui ro, ROI cao

1. Do latency theo tung stage cho tap cau hoi dai dien.
2. Giam top-k mac dinh.
3. Tat mac dinh `cross_encoder` cho query don gian.
4. Tat mac dinh `graph/context` cho phan lon query.
5. Rut gon payload gui sang LLM.

### Pha 2: Toi uu he thong nhung van giu contract

6. Cache graph.
7. Cache parquet dataset/documents.
8. Reuse Qdrant client.
9. Rerank cascade that chat hon.

### Pha 3: Toi uu chat luong/latency nang cao

10. Tach model LLM theo tac vu.
11. Gating verification.
12. Xay index phu cho graph/context.

## 11. Uoc Luong Tac Dong

Uoc luong dinh tinh:

- Giam top-k: tac dong cao, rui ro thap den trung binh.
- Tat `cross_encoder` co dieu kien: tac dong rat cao, rui ro trung binh.
- Tat `graph/context` co dieu kien: tac dong cao, rui ro trung binh.
- Cache graph/parquet: tac dong trung binh den cao, rui ro thap.
- Reuse Qdrant client: tac dong thap den trung binh, rui ro thap.
- Tach model LLM: tac dong cao, rui ro trung binh den cao.
- Gating verification: tac dong cao voi tail latency, rui ro trung binh.

## 12. Ke Hoach Benchmark De Xuat

Can do truoc khi sua code:

- `model_load_latency`
- `Understanding + query planning`
- `Hybrid retrieval`
- `Exact retrieval`
- `Summary retrieval`
- `Initial fusion`
- `Graph expansion`
- `Context expansion`
- `Expanded fusion`
- `ColBERT rerank`
- `Cross-encoder rerank`
- `Evidence selection + sufficiency`
- `Reasoning`
- `Verification`
- `Answer revision`
- `Final verification`
- `Total query`

Tap benchmark nen tach:

- Query exact co so hieu van ban
- Query co dieu luat cu the
- Query dieu kien/thu tuc ro rang
- Query mo ho can tong hop nhieu can cu
- Query de verifier hay fail

Can bao cao:

- p50
- p95
- p99 neu co du mau
- So query roi vao nhanh retry verification
- So query thuc su can graph/context/cross-encoder de giu quality

## 13. De Xuat Hanh Dong Cu The Nhat

Neu chi duoc chon 3 viec de lam truoc, nen uu tien:

1. Giam fan-out retrieval va rerank.
2. Gating `cross_encoder`, `graph`, `context` theo do kho query.
3. Cache graph/parquet va toi uu payload LLM.

Neu chi duoc chon 1 viec de thu nghiem dau tien:

- Hay bat dau bang viec giam top-k va tat `cross_encoder` cho query don gian,
  vi day thuong la thay doi de kiem tra, an toan va cho ket qua nhanh nhat.

## 14. Cac Diem Khong Phai Uu Tien Luc Nay

- `src/agents/legal_understanding.py` hien khong nam tren hot path inference
  chinh.
- `src/agents/sufficiency_checker.py` hien khong nam tren hot path vi pipeline
  dang dung `run_with_sufficiency()` trong evidence selector.
- Ingest, build hybrid, build graph khong phai nguon gay cham cho inference
  online hien tai.

## 15. Ket Luan

He thong hien tai cham khong phai vi mot diem duy nhat, ma vi su cong don cua:

- qua nhieu nhanh retrieval bat mac dinh,
- qua nhieu candidate di vao rerank nang,
- qua nhieu buoc LLM noi tiep,
- va I/O lap lai tren hot path.

Huong toi uu dung nhat la:

1. Giam fan-out.
2. Chi bat buoc nang khi that su can.
3. Cache moi thu doc lap lai.
4. Giam payload qua mang va qua LLM.

Theo thu tu uu tien, day la cach co kha nang giam latency lon nhat trong khi van
giu thay doi nho va tuong thich voi pipeline hien tai.
