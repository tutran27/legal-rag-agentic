import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qdrant_client import QdrantClient, models

from src.common.config import settings


PAYLOAD_INDEXES = {
    "is_current": models.PayloadSchemaType.BOOL,
    "doc_code": models.PayloadSchemaType.KEYWORD,
    "doc_type": models.PayloadSchemaType.KEYWORD,
    "domain": models.PayloadSchemaType.KEYWORD,
    "sector": models.PayloadSchemaType.KEYWORD,
}


def main() -> None:
    client = QdrantClient(
        url=settings.qdrant_url,
        prefer_grpc=True,
        timeout=settings.qdrant_timeout,
    )
    try:
        for field_name, field_schema in PAYLOAD_INDEXES.items():
            print(f"Creating payload index: {field_name} ({field_schema.value})")
            client.create_payload_index(
                collection_name=settings.collection_name,
                field_name=field_name,
                field_schema=field_schema,
                wait=True,
                timeout=600,
            )

        schema = client.get_collection(
            settings.collection_name
        ).payload_schema
        print("Payload indexes:", ", ".join(sorted(schema)))
    finally:
        client.close()


if __name__ == "__main__":
    main()
