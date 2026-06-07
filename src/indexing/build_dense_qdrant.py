import pandas as pd
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer


def build_dense_qdrant(
    corpus_path="data/processed/retrieval_corpus.parquet",
    collection_name="legal_articles_dense",
    model_name="tutran27/vietnamese-legal-phapdien-embedding-v1",
    qdrant_url="http://localhost:6333",
    batch_size=32,
):
    df = pd.read_parquet(corpus_path)

    model = SentenceTransformer(model_name)
    client = QdrantClient(url=qdrant_url)

    dim = model.get_sentence_embedding_dimension()

    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=dim,
            distance=Distance.COSINE,
        ),
    )

    point_id = 0

    for start in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[start:start + batch_size]
        texts = batch["text"].tolist()

        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )

        points = []

        for i, (_, row) in enumerate(batch.iterrows()):
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vectors[i].tolist(),
                    payload={
                        "chunk_id": row["chunk_id"],
                        "unit_id": row["unit_id"],
                        "chunk_type": row["chunk_type"],
                        "text": row["text"],
                        "doc_code": row.get("doc_code"),
                        "doc_type": row.get("doc_type"),
                        "doc_title_submission": row.get("doc_title_submission"),
                        "doc_name_for_submission": row.get("doc_name_for_submission"),
                        "article": row.get("article"),
                        "article_title": row.get("article_title"),
                        "parent_path": row.get("parent_path"),
                        "domain": row.get("domain"),
                        "status": row.get("status"),
                        "source_url": row.get("source_url"),
                    },
                )
            )
            point_id += 1

        client.upsert(
            collection_name=collection_name,
            points=points,
        )

    print("Dense Qdrant index built")
