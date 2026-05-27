# langgraph_pipeline/graph.py
"""Graph definition — wires nodes and edges into a compiled LangGraph pipeline.

HOW TO MODIFY THE GRAPH
────────────────────────
Adding a new node
  1. Write the node function in nodes.py.
  2. Call ``graph.add_node("node_name", my_node)`` here.
  3. Add edges: ``graph.add_edge("predecessor", "node_name")`` or a
     conditional edge with ``graph.add_conditional_edges(...)``.

Swapping a node
  Replace the node function reference in ``graph.add_node`` — the edges stay
  the same as long as the new function reads/writes the same state keys.

Running the pipeline
  Use run_pipeline() — it handles LangSmith run_id capture automatically.
  Or call pipeline.invoke() directly for simple use without tracing metadata.
"""
from __future__ import annotations

import os
import uuid
from typing import Optional

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from langgraph_pipeline.state import PipelineState
from langgraph_pipeline.nodes import (
    load_config_node,
    enhance_query_node,
    retrieve_context_options_node,
    shortlist_contexts_node,
    select_context_node,
    select_final_sibling_section_node,
    finalize_section_selection_node,
    select_test_purpose_node,
    extract_sib_messages_node,
    extract_ue_transition_messages_node,
    extract_procedure_messages_node,
    concatenate_messages_node,
    generate_sequence_diagram_node,
)
from langgraph.errors import GraphInterrupt 

# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Construct and return the *uncompiled* StateGraph.

    Keeping build separate from compile lets you inspect or modify the graph
    (e.g. in tests) before calling ``.compile()``.
    """
    graph = StateGraph(PipelineState)

    # ──────────────────────────────── Register nodes (single RAG path) ─────────────────────────────
    graph.add_node("load_config", load_config_node)
    graph.add_node("enhance_query", enhance_query_node)
    graph.add_node("retrieve_context_options", retrieve_context_options_node)
    graph.add_node("shortlist_contexts", shortlist_contexts_node)
    graph.add_node("select_context", select_context_node)
    graph.add_node("select_final_sibling_section", select_final_sibling_section_node)
    graph.add_node("finalize_section_selection", finalize_section_selection_node)
    graph.add_node("select_test_purpose", select_test_purpose_node)
    graph.add_node("extract_sib_messages", extract_sib_messages_node)
    graph.add_node("extract_ue_transition_messages", extract_ue_transition_messages_node)
    graph.add_node("extract_procedure_messages", extract_procedure_messages_node)
    graph.add_node("concatenate_messages", concatenate_messages_node)
    graph.add_node("generate_sequence_diagram", generate_sequence_diagram_node)



    # ───────────────────────────────────────────── Graph edges ─────────────────────────────────────
    graph.add_edge(START, "load_config")
    graph.add_edge("load_config", "enhance_query")
    graph.add_edge("enhance_query", "retrieve_context_options")
    graph.add_edge("retrieve_context_options", "shortlist_contexts")
    graph.add_edge("shortlist_contexts", "select_context")

    # Sibling section selection flow
    graph.add_edge("select_context", "select_final_sibling_section")
    graph.add_edge("select_final_sibling_section", "finalize_section_selection")
    graph.add_edge("finalize_section_selection", "select_test_purpose")

    # Parallel fan-out after final context and test-purpose selection
    graph.add_edge("select_test_purpose", "extract_sib_messages")
    graph.add_edge("select_test_purpose", "extract_ue_transition_messages")
    graph.add_edge("select_test_purpose", "extract_procedure_messages")

    # Fan-in join node for final ordered message list
    graph.add_edge("extract_sib_messages", "concatenate_messages")
    graph.add_edge("extract_ue_transition_messages", "concatenate_messages")
    graph.add_edge("extract_procedure_messages", "concatenate_messages")

    # Generate sequence diagram from final message list
    graph.add_edge("concatenate_messages", "generate_sequence_diagram")

    graph.add_edge("generate_sequence_diagram", END)

    return graph


# ---------------------------------------------------------------------------
# Compiled pipeline — import this for production use
# ---------------------------------------------------------------------------

_checkpointer = MemorySaver()
pipeline = build_graph().compile(checkpointer=_checkpointer)


# ---------------------------------------------------------------------------
# run_pipeline() — preferred entry point with LangSmith metadata
# ---------------------------------------------------------------------------

def run_pipeline(
    query_config_path: str,
    initial_state: Optional[dict] = None,
    thread_id: Optional[str] = None,
    run_name: Optional[str] = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[dict] = None,
) -> PipelineState:
    """Invoke the compiled single-path RAG pipeline with LangSmith tracing support.

        This is the **recommended** way to run the pipeline. It:
    - Assigns a deterministic ``run_id`` (UUID4) so you can find this exact
      trace in LangSmith before the call even returns.
    - Passes ``run_name``, ``tags``, and ``metadata`` to LangSmith so traces
      are easy to filter and annotate in the UI.
    - Writes ``run_id`` and ``langsmith_url`` back into the returned state dict
      so callers can surface the URL in logs or downstream tooling.

    Parameters
    ----------
    query_config_path:
        Path to the JSON config consumed by ``load_config_node``.
    run_name:
        Human-readable label shown as the trace title in LangSmith.
        Defaults to "rag: <query_config_path>".
    tags:
        List of tag strings attached to the LangSmith trace.  Use these to
        filter runs (e.g. ["experiment-1", "rag"]).
    metadata:
        Arbitrary key-value pairs attached to the trace.  Visible in the
        LangSmith run detail panel.

    Returns
    -------
    PipelineState
        The final graph state, augmented with:
          result["run_id"]         — UUID string of the LangSmith trace
          result["langsmith_url"]  — direct browser URL (empty if tracing off)

    Examples
    --------
    >>> from langgraph_pipeline.graph import run_pipeline
    >>> result = run_pipeline(
    ...     query_config_path="query_config.json",
    ...     tags=["experiment-1"],
    ...     metadata={"dataset": "38.331-h60"},
    ... )
    >>> print(result["message_sequence"])
    >>> print(result["langsmith_url"])   # open in browser
    """
    # Generate a run_id we control so we can build the URL before invoke()
    thread_id = thread_id or str(uuid.uuid4())
    run_id = str(uuid.uuid4())

    effective_run_name = run_name or f"rag: {query_config_path}"
    effective_tags = tags or ["rag"]
    effective_metadata = metadata or {}

    config: RunnableConfig = {
        "run_name": effective_run_name,
        "tags": effective_tags,
        "metadata": effective_metadata,
        "run_id": run_id,
        "configurable": {"thread_id": thread_id},
    }

    input_state: dict = {
        "query_config_path": query_config_path,
    }
    if initial_state:
        input_state.update(initial_state)

    try:
        result: PipelineState = pipeline.invoke(input_state, config=config)
    except GraphInterrupt:
        # Expected for human-in-the-loop interrupt nodes (e.g., select_context).
        # Recover checkpointed partial state so caller can present options and resume.
        snapshot = pipeline.get_state(config)
        result = dict(snapshot.values or {})
    result["thread_id"] = thread_id

    # Build LangSmith URL if tracing is enabled
    tracing_on = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    project = os.getenv("LANGCHAIN_PROJECT", "3gpp-pipeline")
    langsmith_url = (
        f"https://smith.langchain.com/projects/{project}?traceId={run_id}"
        if tracing_on else ""
    )

    result["run_id"] = run_id
    result["langsmith_url"] = langsmith_url

    if langsmith_url:
        print(f"[LangSmith] Trace → {langsmith_url}")

    return result


# ---------------------------------------------------------------------------
# resume_pipeline() — continue a paused run after user context selection
# ---------------------------------------------------------------------------

def resume_pipeline(
    thread_id: str,
    selected_context_index: int,
) -> PipelineState:
    """Resume an interrupted pipeline run with the user's selection.

    Uses the same ``thread_id`` as the original run so LangSmith records
    everything under a single trace.  Catches GraphInterrupt so that
    subsequent interrupt nodes (e.g. sibling or test-purpose selection)
    are handled the same way as the initial run_pipeline interrupt.
    """
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
    }
    try:
        result: PipelineState = pipeline.invoke(
            Command(resume=selected_context_index), config=config
        )
    except GraphInterrupt:
        snapshot = pipeline.get_state(config)
        result = dict(snapshot.values or {})
    result["thread_id"] = thread_id
    return result


def is_awaiting_selection(thread_id: str) -> bool:
    """Return True if the pipeline is paused waiting for any user selection."""
    config = {"configurable": {"thread_id": thread_id}}
    state = pipeline.get_state(config)
    return bool(state.next)


def get_pending_interrupt_payload(thread_id: str):
    """Return the first pending interrupt payload for the given thread, if any."""
    config = {"configurable": {"thread_id": thread_id}}
    state = pipeline.get_state(config)

    tasks = getattr(state, "tasks", None) or []
    for task in tasks:
        interrupts = getattr(task, "interrupts", None) or []
        for intr in interrupts:
            if hasattr(intr, "value"):
                return intr.value
    return None


def get_pending_interrupt_node(thread_id: str) -> Optional[str]:
    """Return the node name currently waiting on interrupt, if available."""
    config = {"configurable": {"thread_id": thread_id}}
    state = pipeline.get_state(config)
    next_nodes = list(getattr(state, "next", ()) or ())
    return next_nodes[0] if next_nodes else None

