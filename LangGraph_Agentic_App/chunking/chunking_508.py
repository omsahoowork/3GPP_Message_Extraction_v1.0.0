from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path
from typing import Any, Optional
import joblib
from langchain_core.documents import Document

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    CHUNKS_508_DIR,
    COLLECTION_NAME_508,
    DATA_508_DIR,
    EMBED_MODEL,
    MODEL_CACHE_DIR,
    VECTORSTORE_508_DIR,
)

_LEVEL_KEYS = [
    "section-L1",
    "section-L2",
    "section-L3",
    "section-L4",
    "section-L5",
    "section-L6",
]
_HEADING_RE = re.compile(r"^(#+)\s+(.*)")
_TABLE_RE = re.compile(r"^Table\s+([^:]+):\s*(.*)$", re.IGNORECASE)
_LTE_TRANSITION_RE = re.compile(
    r"\(\s*state\s+([^\)]+?)\s+to\s+state\s+([^\)]+?)\s*\)",
    re.IGNORECASE,
)
_NR_STATE_TOKEN_RE = re.compile(r"^[0-9][A-Z]?-[A-Z]$")


def _extract_table_id(caption: str) -> str:
    match = _TABLE_RE.match(str(caption).strip())
    if not match:
        return ""
    return match.group(1).strip()


def _extract_table_family(table_id: str) -> str:
    # Example: 4.5.2.2-1 -> 4.5.2.2
    return str(table_id).split("-", 1)[0].strip()


def convert_setext_to_atx(text: str) -> str:
    lines = text.split("\n")
    new_lines: list[str] = []
    index = 0
    while index < len(lines):
        if index + 1 < len(lines):
            line = lines[index].strip()
            next_line = lines[index + 1].strip()
            if set(next_line) == {"="} and re.match(r"^[=-]+$", lines[index + 1]):
                new_lines.append("# " + line)
                index += 2
                continue
            if set(next_line) == {"-"} and re.match(r"^[=-]+$", lines[index + 1]):
                new_lines.append("## " + line)
                index += 2
                continue
        new_lines.append(lines[index])
        index += 1
    return "\n".join(new_lines)


def _normalise_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalise_state_token(token: str) -> str:
    cleaned = re.sub(r"\bSTATE\b", "", token, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", "", cleaned).upper()
    return cleaned.replace("RF", "RF")


def _normalise_rrc_state(value: str) -> str | None:
    cleaned = value.upper().replace("\\", "")
    cleaned = re.sub(r"[_\s]+", " ", cleaned).strip()
    if "RRC IDLE" in cleaned:
        return "RRC_IDLE"
    if "RRC INACTIVE" in cleaned:
        return "RRC_INACTIVE"
    if "RRC CONNECTED" in cleaned:
        return "RRC_CONNECTED"
    return None


def _build_breadcrumb(metadata: dict) -> str:
    parts = [metadata[key] for key in _LEVEL_KEYS if metadata.get(key)]
    return " > ".join(parts)


def _iter_sections(text: str, filename: str) -> list[Document]:
    lines = convert_setext_to_atx(text).splitlines()
    sections: list[Document] = []
    heading_stack: list[tuple[int, str]] = []
    current_lines: list[str] = []
    current_metadata: dict | None = None

    def flush() -> None:
        nonlocal current_lines, current_metadata
        if not current_lines or current_metadata is None:
            return
        breadcrumb = _build_breadcrumb(current_metadata)
        sections.append(
            Document(
                page_content="\n".join(current_lines).strip(),
                metadata={
                    **current_metadata,
                    "filename": filename,
                    "breadcrumb": f"{filename}>{breadcrumb}" if breadcrumb else filename,
                },
            )
        )
        current_lines = []
        current_metadata = None

    for line in lines:
        match = _HEADING_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            heading_stack = heading_stack[: level - 1]
            heading_stack.append((level, title))
            metadata = {"filename": filename}
            for index, (_, section_title) in enumerate(heading_stack[: len(_LEVEL_KEYS)]):
                metadata[_LEVEL_KEYS[index]] = section_title
            current_metadata = metadata
            current_lines = [line]
            continue
        if current_metadata is not None:
            current_lines.append(line)

    flush()
    return sections


def _iter_table_blocks(section: Document) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    current_caption: str | None = None
    current_lines: list[str] = []

    for line in section.page_content.splitlines():
        if _TABLE_RE.match(line.strip()):
            if current_caption is not None:
                blocks.append((current_caption, "\n".join(current_lines).strip()))
            current_caption = line.strip()
            current_lines = [line]
            continue
        if current_caption is not None:
            current_lines.append(line)

    if current_caption is not None:
        blocks.append((current_caption, "\n".join(current_lines).strip()))
    return blocks


def _extract_lte_transition_chunks(section: Document) -> list[Document]:
    chunks: list[Document] = []
    for caption, block in _iter_table_blocks(section):
        transition_match = _LTE_TRANSITION_RE.search(_normalise_whitespace(block))
        if not transition_match:
            continue
        from_state = _normalise_state_token(transition_match.group(1))
        to_state = _normalise_state_token(transition_match.group(2))
        table_id = _extract_table_id(caption)
        metadata = {
            **section.metadata,
            "rat": "lte",
            "chunk_kind": "lte_transition_sequence",
            "transition_from": from_state,
            "transition_to": to_state,
            "sequence_key": f"{from_state}->{to_state}",
            "table_caption": caption,
            "table_id": table_id,
            "table_family": _extract_table_family(table_id),
        }
        chunks.append(
            Document(
                page_content=block,
                metadata=metadata,
            )
        )
    return chunks


def _extract_nr_state_definition_chunks(text: str, filename: str) -> list[Document]:
    lines = convert_setext_to_atx(text).splitlines()
    current_caption = ""
    chunks: list[Document] = []

    for raw_line in lines:
        line = raw_line.strip()
        if line.lower().startswith("table 4.4a.2-"):
            current_caption = line
            continue
        parts = re.split(r"\s{2,}", line)
        if len(parts) < 3 or not _NR_STATE_TOKEN_RE.match(parts[0]):
            continue
        rrc_state = _normalise_rrc_state(parts[2])
        if not rrc_state:
            continue
        state_id = _normalise_state_token(parts[0])
        connectivity = parts[1].strip()
        chunks.append(
            Document(
                page_content=f"{current_caption}\n{line}".strip(),
                metadata={
                    "filename": filename,
                    "breadcrumb": f"{filename}>{current_caption}" if current_caption else filename,
                    "rat": "nr",
                    "chunk_kind": "nr_state_definition",
                    "state_id": state_id,
                    "rrc_state": rrc_state,
                    "connectivity": connectivity,
                    "table_caption": current_caption,
                },
            )
        )
    return chunks


def _extract_nr_rrc_sequence_chunks(section: Document) -> list[Document]:
    chunks: list[Document] = []
    section_title = str(section.metadata.get("section-L2") or section.metadata.get("section-L1") or "")
    section_rrc_state = _normalise_rrc_state(section_title)

    if section_rrc_state:
        chunks.append(
            Document(
                page_content=section.page_content,
                metadata={
                    **section.metadata,
                    "rat": "nr",
                    "chunk_kind": "nr_rrc_section",
                    "rrc_state": section_rrc_state,
                },
            )
        )

    for caption, block in _iter_table_blocks(section):
        title_text = _normalise_whitespace(f"{caption} {block[:200]}")
        rrc_state = _normalise_rrc_state(title_text)
        if not rrc_state or "NR" not in title_text.upper():
            continue
        table_id = _extract_table_id(caption)
        chunks.append(
            Document(
                page_content=block,
                metadata={
                    **section.metadata,
                    "rat": "nr",
                    "chunk_kind": "nr_rrc_sequence_table",
                    "rrc_state": rrc_state,
                    "table_caption": caption,
                    "table_id": table_id,
                    "table_family": _extract_table_family(table_id),
                },
            )
        )
    return chunks


def split_508_markdown_files(data_dir: str | Path) -> list[Document]:
    data_dir = Path(data_dir)
    files = sorted(glob.glob(str(data_dir / "*.md")))
    if not files:
        raise FileNotFoundError(f"No .md files found in {data_dir}")

    all_chunks: list[Document] = []

    for file_path in files:
        path = Path(file_path)
        content = path.read_text(encoding="utf-8")
        sections = _iter_sections(content, path.name)

        if path.name.startswith("36508"):
            for section in sections:
                all_chunks.extend(_extract_lte_transition_chunks(section))
            continue

        if path.name.startswith("38508"):
            all_chunks.extend(_extract_nr_state_definition_chunks(content, path.name))
            for section in sections:
                all_chunks.extend(_extract_nr_rrc_sequence_chunks(section))

    for index, chunk in enumerate(all_chunks):
        chunk.metadata["chunk_index"] = index
        chunk.metadata["char_count"] = len(chunk.page_content)

    return all_chunks


def build_508_vectorstore(
    data_dir: str | Path = DATA_508_DIR,
    collection_name: str = COLLECTION_NAME_508,
    vectorstore_dir: str | Path = VECTORSTORE_508_DIR,
    embed_model: Optional[str] = None,
    model_cache_dir: Optional[str] = None,
) -> Any:
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    embed_model = embed_model or EMBED_MODEL
    model_cache_dir = str(model_cache_dir or MODEL_CACHE_DIR)
    final_chunks = split_508_markdown_files(data_dir=data_dir)

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

    Path(CHUNKS_508_DIR).mkdir(parents=True, exist_ok=True)
    joblib.dump(final_chunks, str(Path(CHUNKS_508_DIR) / "chunks.pkl"))
    return vectorstore


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk 36.508/38.508 markdown and ingest into Chroma.")
    parser.add_argument("--data-dir", default=DATA_508_DIR, required=False)
    parser.add_argument("--collection", default=COLLECTION_NAME_508)
    parser.add_argument("--vectorstore-dir", default=VECTORSTORE_508_DIR)
    args = parser.parse_args()

    build_508_vectorstore(
        data_dir=args.data_dir,
        collection_name=args.collection,
        vectorstore_dir=args.vectorstore_dir,
    )