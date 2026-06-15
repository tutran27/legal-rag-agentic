from types import SimpleNamespace

import pytest
from qdrant_client import models

from src.indexing.qdrant_collection import (
    PAYLOAD_INDEXES,
    create_collection,
    create_payload_indexes,
    validate_collection,
)


class FakeClient:
    def __init__(self, exists=False, dense_dim=1024, modifier=models.Modifier.IDF):
        self.created = None
        self.deleted = None
        self.exists = exists
        self.dense_dim = dense_dim
        self.modifier = modifier
        self.payload_indexes = []

    def collection_exists(self, collection_name):
        return self.exists

    def delete_collection(self, collection_name):
        self.deleted = collection_name

    def create_collection(self, **kwargs):
        self.created = kwargs

    def get_collection(self, collection_name):
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors={
                        "dense": models.VectorParams(
                            size=self.dense_dim,
                            distance=models.Distance.COSINE,
                        )
                    },
                    sparse_vectors={
                        "sparse": models.SparseVectorParams(
                            modifier=self.modifier,
                        )
                    },
                )
            )
        )

    def create_payload_index(self, **kwargs):
        self.payload_indexes.append(kwargs)


def test_collection_uses_sparse_idf():
    client = FakeClient()

    create_collection(client, "legal-idf", dense_dim=1024)

    sparse = client.created["sparse_vectors_config"]["sparse"]
    dense = client.created["vectors_config"]["dense"]
    assert sparse.modifier == models.Modifier.IDF
    assert sparse.index.on_disk is True
    assert dense.size == 1024
    assert dense.on_disk is True
    assert client.created["hnsw_config"].m == 0
    assert client.created["optimizers_config"].indexing_threshold == 0


def test_existing_collection_must_use_idf():
    client = FakeClient(exists=True, modifier=models.Modifier.NONE)

    with pytest.raises(RuntimeError, match="chưa bật sparse IDF"):
        validate_collection(client, "legal-old", dense_dim=1024)


def test_existing_collection_must_match_dense_dimension():
    client = FakeClient(exists=True, dense_dim=768)

    with pytest.raises(RuntimeError, match="1024 chiều"):
        validate_collection(client, "legal-old", dense_dim=1024)


def test_create_all_required_payload_indexes():
    client = FakeClient()

    create_payload_indexes(client, "legal-idf")

    fields = {
        item["field_name"]: item["field_schema"]
        for item in client.payload_indexes
    }
    assert fields == PAYLOAD_INDEXES
