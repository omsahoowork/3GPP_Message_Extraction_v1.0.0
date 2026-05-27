# tools/__init__.py
"""LangGraph-compatible tool functions.

Each tool is decorated with ``@tool`` (LangChain) and can be passed directly
to a LangGraph ``ToolNode`` or bound to an LLM via ``llm.bind_tools([...])``.

Import from here so the graph definition stays decoupled from tool internals.
"""

from tools.retrieval_tool import retrieve_rag_context, combine_context
from tools.retrieval_508_tool import retrieve_508_context
from tools.extraction_tool import (
    extract_messages_rag,
    generate_context_fields_json,
)

__all__ = [
    "retrieve_rag_context",
    "retrieve_508_context",
    "combine_context",
    "extract_messages_rag",
    "generate_context_fields_json",
]
