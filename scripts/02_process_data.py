from src.chunking.build_retrieval_corpus import build_retrieval_corpus
from src.data.build_legal_units import build_legal_units
from src.data.build_submission_mapping import build_submission_mapping
from src.data.process_phapdien import process_phapdien_article
from src.data.process_vbpl import (
    build_documents,
    build_legal_edges,
    build_vbpl_articles,
)


def main():
    build_documents()
    build_vbpl_articles()
    build_legal_edges()
    process_phapdien_article()
    build_legal_units()
    build_retrieval_corpus()
    build_submission_mapping()


if __name__ == "__main__":
    main()
