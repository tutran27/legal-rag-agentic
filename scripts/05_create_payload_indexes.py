import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qdrant_client import QdrantClient

from src.common.config import settings
from src.indexing.qdrant_collection import (
    PAYLOAD_INDEXES,
    create_payload_indexes,
)


def main() -> None:
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        prefer_grpc=True,
        timeout=settings.qdrant_timeout,
    )
    try:
        for field_name, field_schema in PAYLOAD_INDEXES.items():
            print(f"Creating payload index: {field_name} ({field_schema.value})")
        create_payload_indexes(client, settings.collection_name)

        schema = client.get_collection(
            settings.collection_name
        ).payload_schema
        print("Payload indexes:", ", ".join(sorted(schema)))

    finally:
        client.close()


if __name__ == "__main__":
    main()
