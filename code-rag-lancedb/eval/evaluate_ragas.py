"""
Evaluate Code RAG using RAGAS 0.4.x
Compatible with:
    ragas==0.4.3
    langchain==1.x
    langchain-openai==1.x
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import Dataset

# Import metrics only
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    ContextPrecision,
    ContextRecall,
)

from ragas.evaluation import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from src.rag_chain import CodeRagChain
from src.vectorstore import load_vectorstore


llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

ragas_llm = LangchainLLMWrapper(llm)
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)


METRICS = [
    Faithfulness(),
    ResponseRelevancy(),
    ContextPrecision(),
    ContextRecall(),
]


def build_dataset(chain, testset):

    rows = []

    for item in testset:

        result = chain.ask(item["question"])

        rows.append(
            {
                "question": item["question"],
                "answer": result.answer,
                "contexts": [
                    d.page_content
                    for d in result.source_documents
                ],
                "ground_truth": item["ground_truth"],
            }
        )

    return Dataset.from_list(rows)


def run_evaluation(db_path, table_name, testset_path, out_path):

    with open(testset_path, encoding="utf-8") as f:
        testset = json.load(f)

    vectorstore = load_vectorstore(db_path, table_name)

    chain = CodeRagChain(vectorstore)

    dataset = build_dataset(chain, testset)

    result = evaluate(
        dataset=dataset,
        metrics=METRICS,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    df = result.to_pandas()

    df.to_json(
        out_path,
        orient="records",
        indent=2,
    )

    print(df)

    print("\nAverage Scores")

    for c in df.columns:
        if c != "question":
            try:
                print(f"{c:25} {df[c].mean():.3f}")
            except:
                pass


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--db", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--testset", required=True)
    parser.add_argument("--out", required=True)

    args = parser.parse_args()

    run_evaluation(
        args.db,
        args.table,
        args.testset,
        args.out,
    )


if __name__ == "__main__":
    main()