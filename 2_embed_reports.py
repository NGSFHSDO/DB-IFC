#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from tqdm import tqdm

DEFAULT_INPUT_FILE = "data/stockAnalysisReport_2020-01-01_2025-10-10.parquet"
DEFAULT_OUTPUT_XLSX = "data/kospidaq_embeddings.xlsx"
DEFAULT_MAX_ROWS = None
DEFAULT_SAVE_EVERY = 1000


def normalize_reports(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ("broker", "content", "industry", "ticker", "equity", "title"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    return df


def _records_to_df(records: List[dict], embedding_dim: int) -> pd.DataFrame:
    embedding_cols = [f"embedding_{i}" for i in range(1, embedding_dim + 1)]
    return pd.DataFrame(
        records,
        columns=["date", "ticker", "broker", "target_price", "rating", "content", *embedding_cols],
    )


def _write_excel(existing_df: pd.DataFrame | None, records: List[dict], embedding_dim: int, output_path: Path) -> pd.DataFrame:
    if not records or not embedding_dim:
        return existing_df if existing_df is not None else pd.DataFrame()
    new_df = _records_to_df(records, embedding_dim)
    if existing_df is not None and not existing_df.empty:
        out_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        out_df = new_df
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_excel(output_path, index=False)
    return out_df


def _infer_embedding_dim(df: pd.DataFrame) -> int:
    max_idx = 0
    for col in df.columns:
        if not col.startswith("embedding_"):
            continue
        suffix = col.split("_", 1)[-1]
        if suffix.isdigit():
            max_idx = max(max_idx, int(suffix))
    return max_idx


def build_embedding_records(
    df: pd.DataFrame,
    max_rows: int | None,
    save_every: int,
    output_path: Path,
    existing_df: pd.DataFrame | None,
    existing_embedding_dim: int,
) -> Tuple[List[dict], int, pd.DataFrame | None]:
    embedder = OpenAIEmbeddings()
    records: List[dict] = []
    total = len(df)
    embedding_dim = existing_embedding_dim
    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=total, desc="Embedding", unit="row"), start=1):
        broker = row.get("broker", "")
        ticker = row.get("ticker", "")
        date_val = row.get("date")
        date_str = ""
        if pd.notna(date_val):
            date_str = pd.Timestamp(date_val).date().isoformat()
        target_price = row.get("target_price", "")
        rating = row.get("rating", "")
        content = row.get("content", row.get("body", ""))
        text = str(content).strip()
        vector = embedder.embed_documents([text])[0]
        if not embedding_dim:
            embedding_dim = len(vector)
        embedding_cols = {f"embedding_{j+1}": float(v) for j, v in enumerate(vector)}
        records.append(
            {
                "date": date_str,
                "ticker": str(ticker),
                "broker": broker,
                "target_price": str(target_price),
                "rating": str(rating),
                "content": str(content),
                **embedding_cols,
            }
        )
        if i % 100 == 0 or i == total:
            print(f"Embedded {i}/{total} documents ({i/total:.1%})")
        if save_every and i % save_every == 0:
            existing_df = _write_excel(existing_df, records, embedding_dim, output_path)
            print(f"Saved {i} rows to {output_path}")
            records = []
        if max_rows and i >= max_rows:
            break
    return records, embedding_dim, existing_df


def main() -> None:
    load_dotenv()
    input_path = Path(DEFAULT_INPUT_FILE).resolve()
    output_path = Path(DEFAULT_OUTPUT_XLSX).resolve()
    max_rows = DEFAULT_MAX_ROWS if DEFAULT_MAX_ROWS is None else int(DEFAULT_MAX_ROWS)
    save_every = int(DEFAULT_SAVE_EVERY)

    if not input_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {input_path}")

    df = normalize_reports(pd.read_parquet(input_path))
    if df.empty:
        raise RuntimeError("No rows found in input parquet file.")

    existing_df = None
    embedding_dim = 0
    if output_path.exists():
        existing_df = pd.read_excel(output_path)
        embedding_dim = _infer_embedding_dim(existing_df)
        processed_count = len(existing_df)
        if processed_count >= len(df):
            print(f"All {processed_count} rows already embedded. Nothing to do.")
            return
        df = df.iloc[processed_count:].reset_index(drop=True)

    records, embedding_dim, existing_df = build_embedding_records(
        df, max_rows, save_every, output_path, existing_df, embedding_dim
    )
    out_df = _write_excel(existing_df, records, embedding_dim, output_path)
    print(f"Wrote {len(out_df)} rows to {output_path}")


if __name__ == "__main__":
    main()
