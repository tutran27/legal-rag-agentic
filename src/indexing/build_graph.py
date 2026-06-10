import pickle
from pathlib import Path

import networkx as nx
import pandas as pd

def build_graph_index(
    edges_path: str | Path = "data/processed/legal_edges.parquet",
    output_path: str | Path = "data/indexes/graph/legal_graph.pkl",
):
    edges = pd.read_parquet(str(edges_path))
    required_columns = {"source_doc_id", "target_doc_id", "relation_type"}
    missing_columns = required_columns - set(edges.columns)
    if missing_columns:
        raise ValueError(
            f"Missing columns in edges parquet: {sorted(missing_columns)}. "
            f"Found columns: {edges.columns.tolist()}"
        )

    graph = nx.MultiDiGraph()

    for _, row in edges.iterrows():
        source_doc_id = row["source_doc_id"]
        target_doc_id = row["target_doc_id"]
        if pd.isna(source_doc_id) or pd.isna(target_doc_id):
            continue

        relation_type = row["relation_type"]
        if pd.isna(relation_type):
            relation_type = "RELATED"

        graph.add_edge(
            str(source_doc_id),
            str(target_doc_id),
            relation_type=str(relation_type),
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Graph nodes={graph.number_of_nodes()}, edges={graph.number_of_edges()}")


if __name__ == "__main__":
    build_graph_index()
