#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

DEFAULT_INPUT_FILE = "data/stockAnalysisReport_2020-01-01_2025-10-10.parquet"
DEFAULT_OUTPUT_XLSX = "data/kospidaq_embeddings_KR_finBERT.xlsx"
DEFAULT_MAX_ROWS = 4
DEFAULT_SAVE_EVERY = 100
DEFAULT_BATCH_SIZE = 8
DEFAULT_MODEL_NAME = "snunlp/KR-FinBERT"


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


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    masked = last_hidden * mask
    summed = masked.sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom


def _embed_texts(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
) -> List[List[float]]:
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded)
        pooled = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
    return pooled.cpu().numpy().tolist()


def build_embedding_records(
    df: pd.DataFrame,
    max_rows: int | None,
    save_every: int,
    output_path: Path,
    existing_df: pd.DataFrame | None,
    existing_embedding_dim: int,
    batch_size: int,
    model_name: str,
) -> Tuple[List[dict], int, pd.DataFrame | None]:
    device = _get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    records: List[dict] = []
    total = len(df)
    embedding_dim = existing_embedding_dim
    processed = 0

    for start in tqdm(range(0, total, batch_size), desc="Embedding", unit="batch"):
        batch = df.iloc[start : start + batch_size]
        texts = [str(c).strip() for c in batch.get("content", batch.get("body", "")).tolist()]
        vectors = _embed_texts(texts, tokenizer, model, device)
        if not embedding_dim and vectors:
            embedding_dim = len(vectors[0])
        for row, vector in zip(batch.itertuples(index=False), vectors):
            date_val = getattr(row, "date", None)
            date_str = ""
            if pd.notna(date_val):
                date_str = pd.Timestamp(date_val).date().isoformat()
            embedding_cols = {f"embedding_{j+1}": float(v) for j, v in enumerate(vector)}
            records.append(
                {
                    "date": date_str,
                    "ticker": str(getattr(row, "ticker", "")),
                    "broker": str(getattr(row, "broker", "")),
                    "target_price": str(getattr(row, "target_price", "")),
                    "rating": str(getattr(row, "rating", "")),
                    "content": str(getattr(row, "content", getattr(row, "body", ""))),
                    **embedding_cols,
                }
            )
            processed += 1
            if save_every and processed % save_every == 0:
                existing_df = _write_excel(existing_df, records, embedding_dim, output_path)
                print(f"Saved {processed} rows to {output_path}")
                records = []
            if max_rows and processed >= max_rows:
                return records, embedding_dim, existing_df

    return records, embedding_dim, existing_df


def main() -> None:
    load_dotenv()
    input_path = Path(DEFAULT_INPUT_FILE).resolve()
    output_path = Path(DEFAULT_OUTPUT_XLSX).resolve()
    max_rows = DEFAULT_MAX_ROWS if DEFAULT_MAX_ROWS is None else int(DEFAULT_MAX_ROWS)
    save_every = int(DEFAULT_SAVE_EVERY)
    batch_size = int(DEFAULT_BATCH_SIZE)
    model_name = str(DEFAULT_MODEL_NAME)

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
        df,
        max_rows,
        save_every,
        output_path,
        existing_df,
        embedding_dim,
        batch_size,
        model_name,
    )
    out_df = _write_excel(existing_df, records, embedding_dim, output_path)
    print(f"Wrote {len(out_df)} rows to {output_path}")


if __name__ == "__main__":
    main()
