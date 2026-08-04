"""CLI entrypoint for the Code RAG pipeline.

    python main.py ingest   --path ./data/sample_code --db ./lancedb_store --table code_chunks
    python main.py ask      --db ./lancedb_store --table code_chunks --question "..."
    python main.py evaluate --db ./lancedb_store --table code_chunks --testset ./eval/testset.json
"""
from __future__ import annotations

import argparse

from src.chunking import chunk_directory
from src.config import settings
from src.rag_chain import CodeRagChain
from src.vectorstore import add_documents, get_or_create_vectorstore, load_vectorstore


def cmd_ingest(args):
    settings.validate()
    print(f"Chunking Python files under {args.path} (split_methods={not args.no_split_methods})...")
    documents = chunk_directory(args.path, split_methods=not args.no_split_methods)
    print(f"Produced {len(documents)} structural chunks.")

    if not documents:
        print("No chunks produced — nothing to ingest.")
        return

    vs = get_or_create_vectorstore(args.db, args.table, overwrite=args.overwrite)
    added = add_documents(vs, documents)
    print(f"Ingested {added} chunks into LanceDB table '{args.table}' at {args.db}")


def cmd_ask(args):
    settings.validate()
    vs = load_vectorstore(args.db, args.table)
    chain = CodeRagChain(vs, top_k=args.top_k)
    result = chain.ask(args.question)

    print("\n=== Answer ===")
    print(result.answer)

    print("\n=== Sources ===")
    for d in result.source_documents:
        m = d.metadata
        print(f"- {m.get('file_path')} :: {m.get('qualified_name')} "
              f"(lines {m.get('start_line')}-{m.get('end_line')}, type={m.get('chunk_type')})")


def cmd_evaluate(args):
    settings.validate()
    # Deferred import: eval deps (ragas/datasets) are only needed for this subcommand.
    from eval.evaluate_ragas import run_evaluation

    run_evaluation(args.db, args.table, args.testset, args.out)


def main():
    parser = argparse.ArgumentParser(description="Structured-chunking Code RAG over LanceDB")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Chunk and embed a directory of Python files")
    p_ingest.add_argument("--path", required=True, help="Directory to walk for .py files")
    p_ingest.add_argument("--db", required=True, help="LanceDB directory (created if missing)")
    p_ingest.add_argument("--table", required=True, help="LanceDB table name")
    p_ingest.add_argument("--overwrite", action="store_true", help="Wipe table before ingesting")
    p_ingest.add_argument(
        "--no-split-methods",
        action="store_true",
        help="Keep each class as one chunk instead of one chunk per method",
    )
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="Ask a question against the ingested codebase")
    p_ask.add_argument("--db", required=True)
    p_ask.add_argument("--table", required=True)
    p_ask.add_argument("--question", required=True)
    p_ask.add_argument("--top-k", type=int, default=None)
    p_ask.set_defaults(func=cmd_ask)

    p_eval = sub.add_parser("evaluate", help="Run RAGAS evaluation against a test set")
    p_eval.add_argument("--db", required=True)
    p_eval.add_argument("--table", required=True)
    p_eval.add_argument("--testset", default="./eval/testset.json")
    p_eval.add_argument("--out", default="./eval/results.json")
    p_eval.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
