"""LanceDB vector store setup, backed by OpenAI embeddings via LangChain."""
from __future__ import annotations

import lancedb
from langchain_community.vectorstores import LanceDB
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src.config import settings


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.embedding_model, api_key=settings.openai_api_key
    )


def get_or_create_vectorstore(
    db_path: str, table_name: str, overwrite: bool = False
) -> LanceDB:
    """Connect to (or create) a LanceDB table wrapped as a LangChain vector store."""
    db = lancedb.connect(db_path)
    embeddings = get_embeddings()

    existing_tables = db.table_names()
    table = None
    mode = "overwrite" if overwrite else "append"

    if table_name in existing_tables and not overwrite:
        table = db.open_table(table_name)

    vs = LanceDB(
        connection=db,
        embedding=embeddings,
        table_name=table_name,
        table=table,
        mode=mode,
    )
    return vs


def add_documents(vs: LanceDB, documents: list[Document], batch_size: int = 64) -> int:
    """Add documents to the vector store in batches. Returns count added."""
    total = 0
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        vs.add_documents(batch)
        total += len(batch)
    return total


def load_vectorstore(db_path: str, table_name: str) -> LanceDB:
    """Open an existing LanceDB table for querying (raises if it doesn't exist)."""
    db = lancedb.connect(db_path)
    if table_name not in db.table_names():
        raise RuntimeError(
            f"Table '{table_name}' not found in {db_path}. Run `ingest` first."
        )
    table = db.open_table(table_name)
    embeddings = get_embeddings()
    return LanceDB(connection=db, embedding=embeddings, table_name=table_name, table=table)
