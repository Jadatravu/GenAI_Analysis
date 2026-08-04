# Code RAG with LanceDB + LangChain + RAGAS

A Retrieval-Augmented Generation (RAG) pipeline purpose-built for **Python source
code**. Instead of naive fixed-size text splitting, code is parsed with Python's
`ast` module and chunked along *structural* boundaries (module docstring, imports,
top-level functions, classes, and methods). Each chunk carries rich metadata
(file path, chunk type, qualified name, parent class, line ranges, docstring)
which is embedded alongside the code and used for filtering/citation.

## Stack

| Layer            | Tool                                   |
|------------------|-----------------------------------------|
| Chunking         | Python `ast` (structure-aware) + fallback `RecursiveCharacterTextSplitter` |
| Embeddings       | OpenAI (`text-embedding-3-small`, configurable) |
| Vector store     | [LanceDB](https://lancedb.github.io/lancedb/) via `langchain-community` |
| LLM              | OpenAI Chat models via `langchain-openai` |
| Orchestration    | LangChain (LCEL) |
| Evaluation       | [RAGAS](https://docs.ragas.io/) (faithfulness, answer relevancy, context precision/recall) |

## Project layout

```
code-rag-lancedb/
├── README.md
├── requirements.txt
├── .env.example
├── main.py                     # CLI entrypoint: ingest / ask / evaluate
├── src/
│   ├── config.py                # env & settings
│   ├── chunking.py               # AST-based structured chunker for .py files
│   ├── vectorstore.py            # LanceDB vector store builder
│   └── rag_chain.py              # LCEL RAG chain (retriever + prompt + LLM)
├── eval/
│   ├── testset.json              # sample question/ground-truth eval set
│   └── evaluate_ragas.py         # runs the RAG chain over testset.json + scores with RAGAS
└── data/
    └── sample_code/              # a couple of sample .py files to try ingestion on
        ├── example1.py
        └── example2.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set OPENAI_API_KEY
```

## 1. Ingest a codebase

```bash
python main.py ingest --path ./data/sample_code --db ./lancedb_store --table code_chunks
```

This will:
1. Walk `--path` recursively for `*.py` files.
2. Parse each file with `ast`, emitting one chunk per module docstring/import
   block, top-level function, class, and method (see `src/chunking.py`).
3. Embed each chunk with OpenAI embeddings.
4. Upsert into a LanceDB table at `--db` (created if missing).

Re-running ingest on the same path is idempotent-ish: it appends new rows keyed
by a deterministic chunk id (`file_path::qualified_name::start_line`), so you
can safely add more directories over time. Use `--overwrite` to wipe the table
first.

## 2. Ask questions

```bash
python main.py ask --db ./lancedb_store --table code_chunks  --question "Where is retry logic implemented and how does backoff work?"
```

Prints the answer plus the source chunks (file, name, line range) used as
context, so answers are traceable back to exact functions/classes.

## 3. Evaluate with RAGAS

```bash
python main.py evaluate --db ./lancedb_store --table code_chunks  --testset ./eval/testset.json --out ./eval/results.json
```

This runs every question in `eval/testset.json` through the RAG chain,
collects `(question, answer, retrieved_contexts, ground_truth)` and scores it
with RAGAS metrics:

- **faithfulness** – is the answer grounded in retrieved context?
- **answer_relevancy** – does the answer address the question?
- **context_precision** – are retrieved chunks actually relevant?
- **context_recall** – did retrieval find what was needed to answer correctly?

Results (per-question + aggregate averages) are written to `--out` and printed
as a table.

## Customizing chunking granularity

`src/chunking.py` exposes `chunk_python_file(path, split_methods=True)`.
Set `split_methods=False` to keep each class as a single chunk (methods
folded in) instead of splitting every method into its own chunk — useful for
smaller classes where you want more context per retrieval hit.

## Notes

- Swap `OPENAI_EMBEDDING_MODEL` / `OPENAI_CHAT_MODEL` in `.env` to change
  models; any OpenAI-compatible chat/embedding model works via
  `langchain-openai`.
- LanceDB is file-based (no server to run) — the `--db` path is just a
  directory on disk.
- The RAGAS test set (`eval/testset.json`) is a **starting point** with a
  handful of example Q/A pairs against `data/sample_code`. Replace it with
  questions and ground truths relevant to your own codebase before trusting
  the scores.
