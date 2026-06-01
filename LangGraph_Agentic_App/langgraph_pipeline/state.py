# langgraph_pipeline/state.py
"""Graph state definition.

``PipelineState`` is the single shared data structure that flows through every
node in the graph.  LangGraph passes it as a dict between nodes; each node
reads the fields it needs and writes back only the fields it produces.

HOW TO EXTEND
─────────────
- Add a new field here whenever a new node needs to pass data to a downstream node.
- Use ``Optional[...]`` (defaulting to ``None``) for fields that may not be set
  on every execution path — this lets you run partial pipelines without errors.
- If a field holds a list that should be *appended to* across nodes (e.g. a log),
  annotate it with ``Annotated[list, operator.add]`` so LangGraph merges updates
  instead of overwriting.

FIELD REFERENCE
───────────────
question        Input question supplied by the caller.  Every node reads this.

pipeline_mode   Controls which execution path the router selects:
                  "rag"  → retrieve_rag_node → extract_rag_node
                  "kg"   → retrieve_kg_node  → extract_kg_node
                  "llm"  → extract_llm_node  (no retrieval)

subgraph        Serialised KG subgraph string produced by an upstream KG lookup
                step (outside this pipeline, or injected by the caller).  Only
                used when pipeline_mode == "kg".

rag_context     Output of retrieve_rag_node / retrieve_kg_node.  A dict with:
                  question   – echoed question
                  source_doc – list of spec references  (e.g. ["38.331>4.2"])
                  context    – concatenated chunk texts ("[CHUNK 0]\n…")
                  subgraph   – present only in the KG variant

source_doc      Convenience shortcut: list of spec references extracted from
                rag_context.  Populated by the retrieval nodes.

message_sequence  Final output: ordered list of 3GPP signalling message names
                  produced by the active extraction node.

run_id          UUID string of the LangSmith trace run for this invocation.
                Populated by run_pipeline() in graph.py after the call returns.
                Use it to build a direct link to the trace in the LangSmith UI:
                  https://smith.langchain.com/o/<org>/projects/p/<project>/r/<run_id>

error           Non-None if any node catches an exception; the graph routes to
                END immediately when this field is set.
"""
from __future__ import annotations

from typing import Literal, Optional
from typing_extensions import TypedDict


class PipelineState(TypedDict, total=False):
    # ── Inputs ────────────────────────────────────────────────────────────────
    query_config_path: str
    """Path to the user-supplied query config JSON file."""

    query_config: dict
    """Parsed contents of the config JSON file, populated by a load_config node."""

    spec_series_filter: Optional[str]
    """Target spec family for retrieval filtering: "36" (LTE) or "38" (NR)."""

    question: str
    """The 3GPP procedure question supplied by the caller.""" 

    openai_api_key: Optional[str]
    """Session-provided OpenAI API key used for explicit ChatOpenAI(api_key=...) calls."""

    ground_truth_messages: Optional[list[dict]]
    """Ground-truth signaling sequence as a list of {name, direction} dicts."""

    source_doc: list[str]
    """Spec references top-k extracted after searching (e.g. ['38.331>4.2.1'])."""

    rag_context: str
    """Full context str returned by the retrieval tool."""

    user_choice: str
    """User's choice of context from the retrieved options, if applicable."""

    enhanced_contexts: Optional[list[dict]]
    """User-facing JSON summaries for each retrieved context option."""

    raw_contexts: Optional[list[str]]
    """Raw retrieved context strings, cached so resume uses the same contexts as the original run."""

    raw_source_docs: Optional[list[str]]
    """Source docs corresponding to raw_contexts."""

    shortlisted_raw_indices: Optional[list[int]]
    """Original raw_contexts indices retained after LLM relevance filtering."""

    shortlisted_enhanced_contexts: Optional[list[dict]]
    """Shortlisted options shown to the user, re-indexed to 0..N-1 for selection."""

    shortlisted_comparison_json: Optional[list[dict]]
    """Structured comparison JSON for shortlisted options with table-equivalent fields."""

    shortlisted_display_options: Optional[list[dict]]
    """Lightweight context options for GUI interrupt cards (heading + summary only)."""

    selected_context_index: Optional[int]
    """The index the user chose — recorded after resume for reference."""

    selected_raw_context_index: Optional[int]
    """Mapped index in raw_contexts corresponding to selected_context_index."""

    selected_context_json: Optional[dict]
    """Schema-extracted context JSON for the option selected by the user."""

    test_purpose_index: Optional[int]
    """User-selected test purpose index stored using 1-based indexing."""

    sibling_section_options: Optional[list[dict]]
    """List of sibling sections (including self) with summaries: 
    [{"section_id": "38.331>4.2.1", "summary": "...", "raw_index": 0}, ...]"""

    sibling_sections_available: Optional[bool]
    """True if sibling sections were found and user selection is required."""

    sibling_final_selection_index: Optional[int]
    """Index in sibling_section_options that the user finally selected."""

    sibling_display_options: Optional[list[dict]]
    """Lightweight sibling options for GUI interrupt cards (heading + summary only)."""

    thread_id: Optional[str]
    """LangGraph thread ID for this run — used to resume after interrupt."""

    sib_message_sequence: Optional[list[dict]]
    """System Information Block (SIB) messages from system_information_combinations.
    Each entry: {"name": "SIB2", "direction": "GNB_TO_UE", "layer": "SYSTEM"}."""

    ue_state_loop_path: Optional[str]
    """Resolved UE-state loop path (from lte_states.md/nr_states.md) used for transition retrieval."""

    ue_transition_message_sequence: Optional[list[dict]]
    """Messages from 36.508/38.508 state-transition tables.
    Each entry: {"name": "RRCSetup", "direction": "GNB_TO_UE", "layer": "RRC"}."""

    ue_transition_table_contexts: Optional[list[str]]
    """Per-step table_context chunks used by extract_ue_transition_messages_node."""

    ue_transition_state_messages: Optional[list[dict]]
    """Per-loop-state UE transition message groups.
    Format: [{"state": "STATE A", "messages": [{"name": "...", "direction": "...", "cell_id": "...", "layer": "..."}, ...]}, ...]."""

    ue_transition_complete: Optional[bool]
    """True when UE transition extraction completed for all expected loop steps."""

    rat: Optional[str]
    """RAT tag from 508 retrieval payload ("nr" or "lte")."""

    loop_path: Optional[list[str]]
    """Normalized loop-path states returned by 508 retrieval."""

    sequence_chunks: Optional[list[dict]]
    """LTE per-transition retrieval chunks returned by 508 retrieval."""

    state_chunks: Optional[list[dict]]
    """NR per-state retrieval chunks returned by 508 retrieval."""

    combined_context: Optional[str]
    """Combined 508 retrieval context built from sequence/state chunks."""

    missing_transitions: Optional[list[str]]
    """LTE loop transitions that were expected but not found in 508 chunks."""

    missing_states: Optional[list[str]]
    """NR state IDs that were expected but not found in 508 chunks."""

    procedure_message_sequence: Optional[list[dict]]
    """Scenario-specific procedure messages extracted from selected RAG context.
    Each entry: {"name": "RRCSetupRequest", "direction": "UE_TO_GNB", "layer": "RRC", "message_parameters": ["param1", ...]}.
    message_parameters is currently populated for procedure extraction only."""

    initial_mandate_messages: Optional[list[dict]]
    """Initial mandatory messages always prepended (e.g. MIB, SIB1).
    Each entry: {"name": "MIB", "direction": "GNB_TO_UE", "layer": "SYSTEM"}."""












    """ ** Generated by copilot below ** """

    

    pipeline_mode: Literal["rag", "kg", "llm"]
    """Which pipeline variant to execute.  Set by the caller or an upstream node."""

    subgraph: Optional[str]
    """Serialised KG subgraph (required when pipeline_mode == 'kg')."""

    # ── Retrieval outputs ─────────────────────────────────────────────────────
    rag_context: str
    """Full context str returned by the retrieval tool."""

    source_doc: Optional[list[str]]
    """Spec references extracted from rag_context (e.g. ['38.331>4.2.1'])."""

    # ── Extraction output ─────────────────────────────────────────────────────
    message_sequence: Optional[list[dict]]
    """Final ordered message sequence — the pipeline output.
    Each entry: {"name": "RRCSetup", "direction": "GNB_TO_UE", "cell_id": "Cell1", "layer": "RRC", "message_parameters": []}.
    """

    llm_direct_message_sequence: Optional[list[dict]]
    """Direct LLM-only answer branch output as list of {name, direction, cell_id, layer} dicts."""

    # ── Visualization output ──────────────────────────────────────────────────
    sequence_diagram_file: Optional[str]
    """Path to generated Mermaid sequence diagram file (`.mmd`).
    Generated from the final message_sequence."""

    sequence_diagram_text: Optional[str]
    """Mermaid sequence diagram markup as plain text."""

    sequence_csv_file: Optional[str]
    """Path to generated sequence CSV file.
    Columns: serial no., cell, direction, layer, message name, message parameters, category."""

    ue_transition_references: Optional[list[str]]
    """UE transition reference lines formatted as '<state>: <spec_id> $ <section_id>'."""

    procedure_references: Optional[list[str]]
    """Procedure reference lines formatted as '<label>: <spec_id> $ <section_id>'."""

    final_references_text: Optional[str]
    """Combined references block appended to final result display."""

    # ── LangSmith tracing ─────────────────────────────────────────────────────
    run_id: Optional[str]
    """UUID of the LangSmith trace for this invocation.
    Populated by run_pipeline() after invoke() returns.
    Build a UI link:  https://smith.langchain.com/o/<org>/projects/p/<project>/r/<run_id>
    """

    langsmith_url: Optional[str]
    """Direct browser URL to this run in the LangSmith UI.
    Populated by run_pipeline() when tracing is enabled.
    """

    # ── Error handling ────────────────────────────────────────────────────────
    sib_extraction_error: Optional[str]
    """Branch-local error captured by extract_sib_messages_node."""

    ue_transition_error: Optional[str]
    """Branch-local error captured by extract_ue_transition_messages_node."""

    procedure_extraction_error: Optional[str]
    """Branch-local error captured by extract_procedure_messages_node."""

    error: Optional[str]
    """Set to an error description string if any node fails.
    The conditional edge in graph.py routes to END when this is non-None."""
