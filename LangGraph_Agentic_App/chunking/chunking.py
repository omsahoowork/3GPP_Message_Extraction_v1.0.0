from __future__ import annotations

import glob
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional
import joblib

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import argparse

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import EMBED_MODEL, MODEL_CACHE_DIR, CHUNKS_DIR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_LEVEL_KEYS = ["section-L1", "section-L2", "section-L3", "section-L4", "section-L5", "section-L6", "section-L7", "section-L8", "section-L9", "section-L10"]
_DEFAULT_MIN_CHUNK_SIZE = 4000
_DEFAULT_CHUNK_SIZE = 5000
_DEFAULT_CHUNK_OVERLAP = 500


# ---------------------------------------------------------------------------
# Markdown normalisation
# ---------------------------------------------------------------------------

def convert_setext_to_atx(text: str) -> str:
    lines = text.split("\n")
    new_lines: list[str] = []
    i = 0
    while i < len(lines):
        if i + 1 < len(lines):
            line = lines[i].strip()
            next_line = lines[i + 1].strip()
            if set(next_line) == {"="} and re.match(r"^[=-]+$", lines[i + 1]):
                new_lines.append("# " + line)
                i += 2
                continue
            elif set(next_line) == {"-"} and re.match(r"^[=-]+$", lines[i + 1]):
                new_lines.append("## " + line)
                i += 2
                continue
        new_lines.append(lines[i])
        i += 1
    return "\n".join(new_lines)


def normalize_heading(line: str):
    match = re.match(r"^(#+)\s*(\d+(?:\.\d+)*)\s+(.*)", line)
    if not match:
        return line

    _, numbering, title = match.groups()
    level = numbering.count(".") + 1
    level = min(level, 10)
    return f"{'#' * level} {numbering} {title}"


# ---------------------------------------------------------------------------
# TP-parent chunking logic
# ---------------------------------------------------------------------------
def split_by_tp_parent(text: str, filename: str) -> list[Document]:
    lines = text.split("\n")

    heading_pattern = re.compile(r"^(#+)\s+(.*)")
    tp_pattern = re.compile(
        r"\bTest\s+Purpose\s*\(TP\)\s*(\{#.*?\})?",
        re.IGNORECASE,
    )
    # Matches the specific section to be removed
    specific_content_pattern = re.compile(r"Specific\s+message\s+contents", re.IGNORECASE)
    
    # Matches headings whose first non-space token after the hashes is a
    # dotted-number such as "8.1.1.1.2".
    numbered_heading_re = re.compile(r"^(#+)\s*\d+(?:\.\d+)*\s+")

    heading_stack: list[tuple[int, str]] = []   # (level, title)
    chunks: list[Document] = []

    current_chunk: Document | None = None
    current_parent_level: int | None = None
    current_parent_path: str | None = None
    
    # --- NEW STATE VARIABLE ---
    skip_specific_content = False

    for line in lines:
        match = heading_pattern.match(line)

        if match:
            level = len(match.group(1))
            title = match.group(2).strip()

            # 1. Check if we should STOP skipping
            # If we see a numbered heading, we are entering a new formal section
            if numbered_heading_re.match(line):
                skip_specific_content = False

            # 2. Check if we should START skipping
            if specific_content_pattern.search(title):
                skip_specific_content = True
                continue  # Skip this heading line immediately

            # Always keep the heading stack up-to-date.
            heading_stack = heading_stack[: level - 1]
            heading_stack.append((level, title))

            if not skip_specific_content:
                if tp_pattern.search(title):
                    parent_level = level - 1
                    parent_stack = heading_stack[:parent_level]
                    parent_path = " > ".join(t for _, t in parent_stack)

                    if current_chunk is None or current_parent_path != parent_path:
                        if current_chunk is not None:
                            chunks.append(current_chunk)

                        metadata: dict = {"filename": filename}
                        for i, (_, t) in enumerate(parent_stack):
                            if i < len(_LEVEL_KEYS):
                                metadata[_LEVEL_KEYS[i]] = t

                        current_chunk = Document(page_content="", metadata=metadata)
                        current_parent_level = parent_level
                        current_parent_path = parent_path

                    current_chunk.page_content += line + "\n"

                elif current_chunk is not None:
                    if numbered_heading_re.match(line) and level <= current_parent_level:
                        chunks.append(current_chunk)
                        current_chunk = None
                        current_parent_level = None
                        current_parent_path = None
                    else:
                        current_chunk.page_content += line + "\n"

        else:
            # Non-heading line: only append if we aren't in a skip zone  
            if current_chunk is not None and not skip_specific_content:
                current_chunk.page_content += line + "\n"

    if current_chunk is not None:
        chunks.append(current_chunk)

    return chunks

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_breadcrumb(metadata: dict) -> str:
    parts = [metadata[k] for k in _LEVEL_KEYS if metadata.get(k)]
    return " > ".join(parts)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def split_markdown_files(
    data_dir: str | Path,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:

    data_dir = Path(data_dir)
    files = glob.glob(str(data_dir / "*.md"))
    if not files:
        raise FileNotFoundError(f"No .md files found in {data_dir}")

    all_header_splits: list[Document] = []

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Step 1: convert setext headings to ATX
        upd_content = convert_setext_to_atx(content)

        # Step 2: normalise heading levels by dotted-number depth
        lines = upd_content.splitlines()
        normalized_lines = [normalize_heading(line) for line in lines]
        normalized_content = "\n".join(normalized_lines)

        # Step 3: TP-parent chunking (operates on already-normalised text)
        tp_chunks = split_by_tp_parent(normalized_content, Path(file_path).name)

        print(f"{file_path} → {len(tp_chunks)} TP-parent chunks")
        all_header_splits.extend(tp_chunks)

    print(f"After TP-parent split: {len(all_header_splits)} chunks")

    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split_chunks: list[Document] = char_splitter.split_documents(all_header_splits)
    print(f"After size guard split: {len(split_chunks)} chunks")


    # Step 4: enrich metadata with breadcrumb
    final_chunks: list[Document] = []
    for idx, chunk in enumerate(split_chunks):
        breadcrumb = _build_breadcrumb(chunk.metadata)
        enriched_text = (
            f"[{breadcrumb}]\n\n{chunk.page_content}" if breadcrumb else chunk.page_content
        )
        enriched_metadata = {
            **chunk.metadata,
            "breadcrumb": f"{chunk.metadata.get('filename', '')}>{breadcrumb}",
            "chunk_index": idx,
            "char_count": len(chunk.page_content),
        }
        final_chunks.append(Document(page_content=enriched_text, metadata=enriched_metadata))

    print(f"After enriching metadata: {len(final_chunks)} chunks")
    return final_chunks

def build_vectorstore(
    data_dir: str | Path,
    collection_name: str,
    vectorstore_dir: str | Path,
    embed_model: Optional[str] = None,
    model_cache_dir: Optional[str] = None,
) -> Chroma:
    """Full ingestion pipeline: chunk ``*.md`` files → embed → store in Chroma.

    Parameters
    ----------
    data_dir:
        Directory containing source ``.md`` files.
    collection_name:
        Chroma collection name (e.g. ``"multiple_spec"``).
    vectorstore_dir:
        Directory where Chroma persists its data.
    embed_model:
        HuggingFace model name. Defaults to ``config.EMBED_MODEL``.
    model_cache_dir:
        Local cache directory for the embedding model. Defaults to
        ``config.MODEL_CACHE_DIR``.

    Returns
    -------
    Chroma
        The populated vectorstore instance.
    """
    

    embed_model = EMBED_MODEL
    model_cache_dir = str(model_cache_dir or MODEL_CACHE_DIR)

    final_chunks = split_markdown_files(
        data_dir=data_dir,
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=embed_model,
        cache_folder=model_cache_dir,
        show_progress=True,
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma.from_documents(
        documents=final_chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(vectorstore_dir),
    )

    print(f"\nStored {len(final_chunks)} chunks in collection '{collection_name}'")
    print(f"Vectorstore path: {vectorstore_dir}")
    if final_chunks:
        print("\nSample chunk metadata:")
        print(final_chunks[0].metadata)
        print("\nSample chunk text (first 300 chars):")
        print(final_chunks[0].page_content[:300])
        joblib.dump(final_chunks, f"{CHUNKS_DIR}/chunks.pkl")

    return vectorstore


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk .md files and ingest into Chroma.")
    parser.add_argument("--data-dir", default=str(Path(__file__).parent.parent / "data_523"), required=False, help="Directory with .md source files")
    parser.add_argument("--collection", default="multiple_spec", help="Chroma collection name")
    parser.add_argument(
        "--vectorstore-dir",
        default=str(Path(__file__).parent.parent / "vectorstore"),
        help="Chroma persist directory",
    )
    parser.add_argument("--min-chunk-size", type=int, default=_DEFAULT_MIN_CHUNK_SIZE)
    parser.add_argument("--chunk-size", type=int, default=_DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=_DEFAULT_CHUNK_OVERLAP)
    args = parser.parse_args()

    build_vectorstore(
        data_dir=args.data_dir,
        collection_name=args.collection,
        vectorstore_dir=args.vectorstore_dir,
    )
