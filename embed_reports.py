#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build a local FAISS vector store for report documents using OpenAI embeddings.

Usage
-----
python embed_reports.py --input back\data\ --output artifacts/vectorstore

Requirements
------------
pip install langchain-openai langchain-community faiss-cpu pandas pyarrow
OPENAI_API_KEY must be available in the environment (or .env loaded by caller).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

REPORT_TYPES = ("stock", "industry")


def load_parquet_reports(input_dir: Path) -> pd.DataFrame:
    paths: List[Path] = []
    for suffix in REPORT_TYPES:
        paths.extend(sorted(input_dir.glob(f"**/*{suffix}*.parquet")))
    if not paths:
        raise FileNotFoundError(f"No parquet files found under {input_dir}")
    frames = [pd.read_parquet(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    return df


def normalize_reports(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ("broker", "content", "industry", "ticker", "equity", "title"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    return df


MAX_CONTENT_CHARS = int(os.environ.get("EMBED_MAX_CHARS", "12000"))


def build_documents(df: pd.DataFrame) -> Iterable[Document]:
    for idx, row in df.iterrows():
        view_type = row.get("view_type") or ("stock" if row.get("ticker") else "industry")
        broker = row.get("broker", "")
        primary = row.get("ticker") or row.get("industry") or row.get("equity") or f"{view_type}_{idx}"
        date_val = row.get("date")
        date_str = ""
        if pd.notna(date_val):
            date_str = pd.Timestamp(date_val).date().isoformat()
        title = row.get("title", "")
        content = row.get("content", row.get("body", ""))
        lines = [
            f"Type: {view_type}",
            f"Primary: {primary}",
            f"Broker: {broker}",
            f"Date: {date_str}",
        ]
        if title:
            lines.append(f"Title: {title}")
        lines.append("")
        lines.append(str(content))
        text = "\n".join(lines).strip()
        if MAX_CONTENT_CHARS and len(text) > MAX_CONTENT_CHARS:
            text = text[:MAX_CONTENT_CHARS]
        metadata = {
            "type": view_type,
            "primary": primary,
            "broker": broker,
            "date": date_str,
            "source_index": int(idx),
        }
        yield Document(page_content=text, metadata=metadata)


def build_vector_store(documents: List[Document]) -> FAISS:
    embedder = OpenAIEmbeddings()
    texts: List[str] = []
    embeddings: List[List[float]] = []
    metadatas: List[dict] = []
    ids: List[str] = []
    total = len(documents)
    for i, doc in enumerate(documents, start=1):
        text = doc.page_content
        vector = embedder.embed_documents([text])[0]
        texts.append(text)
        embeddings.append(vector)
        metadatas.append(doc.metadata)
        base_id = doc.metadata.get("primary") or "doc"
        ids.append(f"{base_id}_{i}")
        if i % 100 == 0 or i == total:
            print(f"Embedded {i}/{total} documents ({i/total:.1%})")
    text_embedding_pairs = list(zip(texts, embeddings))
    return FAISS.from_embeddings(text_embedding_pairs, embedder, metadatas=metadatas, ids=ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed reports into a local FAISS store.")
    parser.add_argument("--input", required=True, help="Directory containing report parquet files.")
    parser.add_argument("--output", required=True, help="Directory to persist FAISS index.")
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    df = normalize_reports(load_parquet_reports(input_dir))
    docs = list(build_documents(df))
    if not docs:
        raise RuntimeError("No documents produced from input parquet files.")

    vectorstore = build_vector_store(docs)
    vectorstore.save_local(str(output_dir))
    print(f"Saved FAISS index to {output_dir} with {len(docs)} documents.")


if __name__ == "__main__":
    main()
