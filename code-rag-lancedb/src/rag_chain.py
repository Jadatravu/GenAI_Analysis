"""LangChain LCEL RAG chain for question-answering over chunked Python code."""
from __future__ import annotations

from dataclasses import dataclass

from langchain_community.vectorstores import LanceDB
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

from src.config import settings

SYSTEM_PROMPT = """You are a senior software engineer answering questions about a \
Python codebase. You are given retrieved code chunks (functions, classes, methods, \
or module headers) as context. Answer the question using ONLY the given context.

Rules:
- Cite the specific file path and function/class/method name for any claim you make.
- If the context does not contain enough information to answer, say so explicitly \
instead of guessing.
- Keep the answer concise and technical; include short code snippets only when they \
clarify the answer.

Context:
{context}
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


def _format_docs(docs: list[Document]) -> str:
    blocks = []
    for d in docs:
        m = d.metadata
        header = f"# {m.get('file_path')} :: {m.get('qualified_name')} " \
                  f"(lines {m.get('start_line')}-{m.get('end_line')}, type={m.get('chunk_type')})"
        blocks.append(f"{header}\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)


@dataclass
class RagResult:
    question: str
    answer: str
    source_documents: list[Document]


class CodeRagChain:
    """Retriever + LLM chain that also exposes the raw retrieved contexts,
    which is needed for RAGAS evaluation."""

    def __init__(self, vectorstore: LanceDB, top_k: int | None = None):
        self.vectorstore = vectorstore
        self.top_k = top_k or settings.retriever_top_k
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": self.top_k})
        self.llm = ChatOpenAI(
            model=settings.chat_model, api_key=settings.openai_api_key, temperature=0
        )

        self.chain = (
            {
                "context": self.retriever | RunnableLambda(_format_docs),
                "question": RunnablePassthrough(),
            }
            | PROMPT
            | self.llm
            | StrOutputParser()
        )

    def retrieve(self, question: str) -> list[Document]:
        return self.retriever.invoke(question)

    def ask(self, question: str) -> RagResult:
        docs = self.retrieve(question)
        answer = (
            {
                "context": RunnableLambda(lambda _: _format_docs(docs)),
                "question": RunnablePassthrough(),
            }
            | PROMPT
            | self.llm
            | StrOutputParser()
        ).invoke(question)
        return RagResult(question=question, answer=answer, source_documents=docs)
