"""Structure-aware chunking for Python source code.

Instead of splitting code by a fixed character/token window (which cuts
functions in half and separates a function from its docstring), we parse each
file with the `ast` module and emit one chunk per meaningful structural unit:

- a "module" chunk: module docstring + top-level import statements
- one chunk per top-level function
- one chunk per top-level class
    - by default, one chunk per method inside that class (``split_methods=True``)
    - or, if ``split_methods=False``, the whole class body as a single chunk

Any file that fails to parse (syntax errors, non-UTF8, etc.) falls back to a
plain recursive character splitter so ingestion never hard-fails on a single
bad file.

Each chunk is returned as a LangChain ``Document`` with metadata:

    file_path        str   - path to the source file
    chunk_type       str   - "module" | "function" | "class" | "method"
    name             str   - function/class/method name ("module" for module chunk)
    qualified_name   str   - e.g. "MyClass.my_method"
    parent           str   - enclosing class name, "" if none
    start_line       int
    end_line         int
    docstring        str   - extracted docstring, "" if none
    chunk_id         str   - deterministic id: f"{file_path}::{qualified_name}::{start_line}"
"""
from __future__ import annotations

import ast
import hashlib
import os
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

FALLBACK_CHUNK_SIZE = 1500
FALLBACK_CHUNK_OVERLAP = 200


@dataclass
class CodeChunk:
    text: str
    chunk_type: str
    name: str
    qualified_name: str
    parent: str
    start_line: int
    end_line: int
    docstring: str = ""
    file_path: str = ""

    def chunk_id(self) -> str:
        raw = f"{self.file_path}::{self.qualified_name}::{self.start_line}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    def to_document(self) -> Document:
        return Document(
            page_content=self.text,
            metadata={
                "file_path": self.file_path,
                "chunk_type": self.chunk_type,
                "name": self.name,
                "qualified_name": self.qualified_name,
                "parent": self.parent,
                "start_line": self.start_line,
                "end_line": self.end_line,
                "docstring": self.docstring or "",
                "chunk_id": self.chunk_id(),
            },
        )


def _get_source_segment(source_lines: list[str], node: ast.AST) -> str:
    start = node.lineno - 1
    end = getattr(node, "end_lineno", node.lineno)
    return "\n".join(source_lines[start:end])


def _extract_module_header(tree: ast.Module, source_lines: list[str]) -> str | None:
    """Grab module docstring + leading import statements as one chunk."""
    header_nodes = []
    docstring = ast.get_docstring(tree)
    body = list(tree.body)

    # Skip the docstring expression node itself if present, we render it separately.
    if body and isinstance(body[0], ast.Expr) and isinstance(
        getattr(body[0], "value", None), (ast.Constant,)
    ) and isinstance(body[0].value.value, str):
        body = body[1:]

    for node in body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            header_nodes.append(node)
        else:
            break  # stop at first non-import top-level statement

    parts = []
    if docstring:
        parts.append(f'"""{docstring}"""')
    for node in header_nodes:
        parts.append(_get_source_segment(source_lines, node))

    if not parts:
        return None
    return "\n".join(parts)


def _function_chunk(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    source_lines: list[str],
    file_path: str,
    parent: str = "",
) -> CodeChunk:
    text = _get_source_segment(source_lines, node)
    qualified = f"{parent}.{node.name}" if parent else node.name
    return CodeChunk(
        text=text,
        chunk_type="method" if parent else "function",
        name=node.name,
        qualified_name=qualified,
        parent=parent,
        start_line=node.lineno,
        end_line=getattr(node, "end_lineno", node.lineno),
        docstring=ast.get_docstring(node) or "",
        file_path=file_path,
    )


def _class_chunk(
    node: ast.ClassDef, source_lines: list[str], file_path: str
) -> CodeChunk:
    text = _get_source_segment(source_lines, node)
    return CodeChunk(
        text=text,
        chunk_type="class",
        name=node.name,
        qualified_name=node.name,
        parent="",
        start_line=node.lineno,
        end_line=getattr(node, "end_lineno", node.lineno),
        docstring=ast.get_docstring(node) or "",
        file_path=file_path,
    )


def chunk_python_source(
    source: str, file_path: str, split_methods: bool = True
) -> list[CodeChunk]:
    """Parse Python source text into structural CodeChunks."""
    tree = ast.parse(source)
    source_lines = source.splitlines()
    chunks: list[CodeChunk] = []

    header = _extract_module_header(tree, source_lines)
    if header:
        chunks.append(
            CodeChunk(
                text=header,
                chunk_type="module",
                name="module",
                qualified_name="module",
                parent="",
                start_line=1,
                end_line=1,
                docstring=ast.get_docstring(tree) or "",
                file_path=file_path,
            )
        )

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunks.append(_function_chunk(node, source_lines, file_path))

        elif isinstance(node, ast.ClassDef):
            if not split_methods:
                chunks.append(_class_chunk(node, source_lines, file_path))
                continue

            # Emit a small "class header" chunk (class line + docstring + class
            # attributes) plus one chunk per method, so retrieval can land on
            # a single method without pulling in unrelated siblings.
            method_nodes = [
                n
                for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            if method_nodes:
                first_method_line = method_nodes[0].lineno
                header_lines = source_lines[node.lineno - 1 : first_method_line - 1]
                header_text = "\n".join(header_lines).rstrip()
                if header_text.strip():
                    chunks.append(
                        CodeChunk(
                            text=header_text,
                            chunk_type="class",
                            name=node.name,
                            qualified_name=node.name,
                            parent="",
                            start_line=node.lineno,
                            end_line=first_method_line - 1,
                            docstring=ast.get_docstring(node) or "",
                            file_path=file_path,
                        )
                    )
                for method in method_nodes:
                    chunks.append(
                        _function_chunk(method, source_lines, file_path, parent=node.name)
                    )
            else:
                # class with no methods (e.g. dataclass / constants only)
                chunks.append(_class_chunk(node, source_lines, file_path))

    return chunks


def _fallback_chunks(source: str, file_path: str) -> list[CodeChunk]:
    """Used when a file fails to parse with ast (syntax errors, etc.)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=FALLBACK_CHUNK_SIZE,
        chunk_overlap=FALLBACK_CHUNK_OVERLAP,
        separators=["\ndef ", "\nclass ", "\n\n", "\n", " "],
    )
    pieces = splitter.split_text(source)
    chunks = []
    for i, piece in enumerate(pieces):
        chunks.append(
            CodeChunk(
                text=piece,
                chunk_type="raw_fallback",
                name=f"fragment_{i}",
                qualified_name=f"fragment_{i}",
                parent="",
                start_line=0,
                end_line=0,
                docstring="",
                file_path=file_path,
            )
        )
    return chunks


def chunk_python_file(path: str, split_methods: bool = True) -> list[Document]:
    """Read + chunk a single .py file on disk. Returns LangChain Documents."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        source = f.read()

    try:
        code_chunks = chunk_python_source(source, path, split_methods=split_methods)
        if not code_chunks:
            # e.g. an empty file or one with only unhandled top-level statements
            code_chunks = _fallback_chunks(source, path)
    except SyntaxError:
        code_chunks = _fallback_chunks(source, path)

    return [c.to_document() for c in code_chunks]


def chunk_directory(root: str, split_methods: bool = True) -> list[Document]:
    """Walk `root` recursively, chunking every .py file found."""
    documents: list[Document] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if "/.venv" in dirpath or "/node_modules" in dirpath or "/.git" in dirpath:
            continue
        for fname in filenames:
            if fname.endswith(".py"):
                full_path = os.path.join(dirpath, fname)
                try:
                    documents.extend(chunk_python_file(full_path, split_methods=split_methods))
                except Exception as e:  # noqa: BLE001 - keep ingestion resilient
                    print(f"[chunking] skipped {full_path}: {e}")
    return documents
