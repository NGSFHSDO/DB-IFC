from __future__ import annotations

"""
Reconstruct embeddings from the FAISS vector store and export them to CSV.

This script scans the stockAnalysisReport parquet file, matches each row with
its corresponding vector in artifacts/vectorstore/, and writes the full table
to data/kospidaq_embeddings.csv with columns date,ticker,embedding.

python export_kospidaq_embeddings.py
"""

import pickle
from pathlib import Path
from typing import Dict, Tuple

import faiss  # type: ignore
import numpy as np
import pandas as pd

PARQUET_PATH = Path("back\data\stockAnalysisReport_2025-10-10_2025-10-10.parquet")
VECTORSTORE_DIR = Path("artifacts/vectorstore")
OUTPUT_PATH = Path("back/data/kospidaq_embeddings.csv")


def _format_embedding(vec: np.ndarray) -> str:
    return "[" + " ".join(map(str, vec.tolist())) + "]"


def _load_docstore_mappings() -> Tuple[Dict[int, str], Dict[str, int]]:
    with open(VECTORSTORE_DIR / "index.pkl", "rb") as f:
        docstore, index_to_id = pickle.load(f)
    source_to_doc: Dict[int, str] = {}
    for doc_id, document in docstore._dict.items():
        src_idx = document.metadata.get("source_index")
        if isinstance(src_idx, int) and src_idx not in source_to_doc:
            source_to_doc[src_idx] = doc_id
    id_to_position = {doc_id: pos for pos, doc_id in index_to_id.items()}
    return source_to_doc, id_to_position


def main() -> None:
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"Parquet file not found: {PARQUET_PATH}")
    if not (VECTORSTORE_DIR / "index.faiss").exists():
        raise FileNotFoundError(f"FAISS index not found at {VECTORSTORE_DIR}")

    source_to_doc, id_to_position = _load_docstore_mappings()
    index = faiss.read_index(str(VECTORSTORE_DIR / "index.faiss"))
    df = pd.read_parquet(PARQUET_PATH)

    records = []
    for idx, row in df.iterrows():
        doc_id = source_to_doc.get(idx)
        if doc_id is None:
            raise KeyError(f"Missing embedding for source_index={idx}")
        pos = id_to_position[doc_id]
        vector = index.reconstruct(pos)
        date_val = pd.to_datetime(row["date"]).date().isoformat()
        ticker = row.get("ticker", "")
        records.append(
            {
                "date": date_val,
                "ticker": str(ticker),
                "embedding": _format_embedding(vector),
            }
        )

    out_df = pd.DataFrame(records, columns=["date", "ticker", "embedding"])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(out_df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
