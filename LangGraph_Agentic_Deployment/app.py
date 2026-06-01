from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import streamlit as st

from services.config_builder import build_query_config
from services.feedback_service import write_feedback
from services.pipeline_service import (
    DEPLOYMENT_DIR,
    PipelineSnapshot,
    continue_pipeline,
    has_required_openai_key,
    node_status_text,
    start_pipeline,
    write_runtime_query_config,
)


st.set_page_config(page_title="LangGraph Agentic Pipeline", page_icon="LG", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: #0f1115;
        color: #f3f4f6;
        font-family: "Inter", "Segoe UI", sans-serif;
    }
    [data-testid="stAppViewContainer"] > .main .block-container,
    [data-testid="stMainBlockContainer"] {
        max-width: 750px;
        width: 100%;
        margin-left: auto;
        margin-right: auto;
        padding-top: 2rem;
        padding-left: 0;
        padding-right: 0;
    }
    h1, h2, h3, h4, h5, h6, p, div, span, label {
        font-family: "Inter", "Segoe UI", sans-serif !important;
    }
    [data-testid="stSidebar"] {
        background: #111318;
    }
    div.stButton > button {
        background: #1f2937;
        color: #f9fafb;
        border: 1px solid #374151;
        border-radius: 10px;
        padding: 0.45rem 0.9rem;
        font-weight: 600;
    }
    div.stButton > button:hover {
        border-color: #9ca3af;
        background: #111827;
    }
    .card {
        background: #161a22;
        border: 1px solid #2b313c;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.75rem;
    }
    .muted {
        color: #9ca3af;
        font-size: 0.9rem;
    }
    .chip {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 999px;
        border: 1px solid #3f4754;
        font-size: 0.75rem;
        color: #e5e7eb;
        margin-right: 0.4rem;
        margin-bottom: 0.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _init_state() -> None:
    defaults: dict[str, Any] = {
        "snapshot": None,
        "current_input": {},
        "query_counter": 0,
        "last_feedback_key": "",
        "decision_trail": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()

st.title("AI Based 3GPP Conformance Test Message Sequence Generator")
st.caption("Describe a test scenario and let the AI generate the expected message sequence along with a sequence diagram.")

if not has_required_openai_key():
    st.error(
        "OPENAI_API_KEY is missing. Set it in KG_Generation_Pipeline/RAG_KG_Integration/.env "
        "or Streamlit secrets, then restart the app."
    )

with st.form("query_form", clear_on_submit=False):
    test_description = st.text_area(
        "Test Description",
        placeholder="Describe the test scenario. This is required for every query.",
        height=130,
    )
    rat = st.radio("RAT", options=["NR", "LTE"], horizontal=True)
    additional_prompt = ""
    submitted = st.form_submit_button("Start Query", type="primary")

if submitted:
    if not str(test_description).strip():
        st.error("Test Description is required.")
    elif not has_required_openai_key():
        st.error("Cannot start pipeline without OPENAI_API_KEY.")
    else:
        st.session_state.query_counter += 1
        query_id = f"q{st.session_state.query_counter}"
        query_config = build_query_config(
            test_description=test_description,
            rat=rat,
            additional_prompt=additional_prompt,
        )
        config_path = write_runtime_query_config(query_config, run_id=query_id)

        with st.spinner("Running pipeline: retrieval and objective-based shortlisting..."):
            snapshot = start_pipeline(
                config_path=config_path,
                tags=["streamlit", "agentic-app", rat.lower()],
                metadata={
                    "query_id": query_id,
                    "rat": rat,
                },
            )

        st.session_state.snapshot = snapshot
        st.session_state.current_input = {
            "test_description": test_description,
            "rat": rat,
            "additional_prompt": additional_prompt,
            "query_id": query_id,
            "config_path": str(config_path),
        }
        st.session_state.last_feedback_key = ""
        st.session_state.decision_trail = [
            {
                "step": "query_input",
                "label": f"Query started ({rat})",
                "details": str(test_description).strip(),
            }
        ]

snapshot = st.session_state.snapshot
if snapshot is not None and snapshot.awaiting_selection:
    st.markdown("---")
    st.subheader("Selection Required")
    st.markdown(f"<div class='muted'>{node_status_text(snapshot.pending_node)}</div>", unsafe_allow_html=True)

    options = snapshot.pending_options
    if not options:
        st.error("Pipeline is waiting for input but no options were returned.")
    else:
        for option in options:
            option_index = int(option.get("option_index", 0))
            is_test_purpose_step = snapshot.pending_node == "select_test_purpose"
            heading = (
                str(option.get("test_heading", "")).strip()
                or str(option.get("test_purpose", "")).strip()
                or str(option.get("section_id", "")).strip()
                or f"Option {option_index}"
            )
            objective = str(option.get("objective", "")).strip() or str(option.get("summary", "")).strip()

            with st.container(border=True):
                st.markdown(f"**{heading}**")
                if objective and not is_test_purpose_step:
                    st.markdown(f"<div class='muted'><b>Objective:</b> {objective}</div>", unsafe_allow_html=True)
                elif not is_test_purpose_step:
                    preview = dict(option)
                    preview.pop("option_index", None)
                    st.json(preview)

                button_key = f"select_{snapshot.result.get('thread_id')}_{snapshot.pending_node}_{option_index}"
                button_text = f"Choose Option {option_index}" if not is_test_purpose_step else "Select Test Purpose"
                if st.button(button_text, key=button_key):
                    trail_step = snapshot.pending_node or "selection"
                    st.session_state.decision_trail.append(
                        {
                            "step": trail_step,
                            "label": heading,
                            "details": objective or f"Selected option {option_index}",
                        }
                    )
                    spinner_text = "Applying selection and continuing pipeline..."
                    if snapshot.pending_node == "select_context":
                        spinner_text = "Applying context selection and preparing sibling options..."
                    elif snapshot.pending_node == "select_final_sibling_section":
                        spinner_text = "Applying sibling selection and preparing test-purpose options..."
                    elif snapshot.pending_node == "select_test_purpose":
                        spinner_text = "Applying test purpose and running extraction plus sequence generation..."

                    with st.spinner(spinner_text):
                        next_snapshot = continue_pipeline(snapshot.thread_id or "", option_index)
                    st.session_state.snapshot = next_snapshot
                    st.rerun()

if snapshot is not None:
    result = snapshot.result
    st.markdown("---")
    st.subheader("Decision Trail")
    decisions = st.session_state.get("decision_trail", [])
    if decisions:
        for i, item in enumerate(decisions, start=1):
            st.markdown(
                f"<div class='card'><b>{i}. {item.get('label', '-')}</b><br/>"
                f"<span class='muted'>{item.get('step', '-')}: {item.get('details', '-')}</span></div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("No decisions recorded yet.")

    with st.expander("Pipeline State Snapshot", expanded=False):
        if result.get("error"):
            st.error(str(result.get("error")))
        loop_path = result.get("ue_state_loop_path")
        loop_path_text = ""
        if isinstance(loop_path, list):
            loop_path_text = " -> ".join(str(item) for item in loop_path)
        else:
            loop_path_text = str(loop_path or "")
        snapshot_payload = {
            "pending_node": snapshot.pending_node or "Completed",
            "awaiting_selection": snapshot.awaiting_selection,
            "selected_context_index": result.get("selected_context_index"),
            "test_purpose_index": result.get("test_purpose_index"),
            "ue_state_loop_path": loop_path_text,
            "sequence_diagram_file": result.get("sequence_diagram_file"),
            "sequence_csv_file": result.get("sequence_csv_file"),
        }
        st.code(json.dumps(snapshot_payload, indent=2, ensure_ascii=False), language="json")

if snapshot is not None and not snapshot.awaiting_selection:
    result = snapshot.result
    if result.get("error"):
        st.error(str(result.get("error")))
    elif result.get("message_sequence"):
        st.success("Pipeline completed.")

        # st.markdown("### Final Message Sequence")
        messages = result.get("message_sequence") or []

        # if isinstance(messages, list) and messages:
        #     st.code(json.dumps(messages, indent=2, ensure_ascii=False), language="json")
        # else:
        #     st.info("No message sequence available.")

        seq_diagram = str(result.get("sequence_diagram_file", "")).strip()
        if seq_diagram and Path(seq_diagram).exists():
            st.markdown("### Sequence Diagram")
            st.image(seq_diagram, use_container_width=True)

        # Prefer streaming `message_sequence` to the client as CSV (built from in-memory result)
        message_sequence = result.get("message_sequence", []) or []
        if message_sequence:
            import io
            import csv
            import json

            output = io.StringIO()
            writer = csv.writer(output)
            # header row
            writer.writerow(["cell_id", "direction", "layer", "name", "message_parameters"])

            for item in message_sequence:
                cell_id = item.get("cell_id", "")
                direction = item.get("direction", "")
                layer = item.get("layer", "")
                name = item.get("name", "")
                params = item.get("message_parameters", [])
                params_str = json.dumps(params, ensure_ascii=False)
                writer.writerow([cell_id, direction, layer, name, params_str])

            csv_bytes = output.getvalue().encode("utf-8")
            suggested_name = f"{result.get('run_id','sequence')}_message_sequence.csv"
            st.download_button(
                label="Download CSV",
                data=csv_bytes,
                file_name=suggested_name,
                mime="text/csv",
            )
        else:
            # Fallback: if a CSV file was produced on disk, offer it
            seq_csv = str(result.get("sequence_csv_file", "")).strip()
            if seq_csv and Path(seq_csv).exists():
                with open(seq_csv, "rb") as f:
                    csv_bytes = f.read()
                st.download_button(
                    label="Download CSV",
                    data=csv_bytes,
                    file_name=Path(seq_csv).name,
                    mime="text/csv",
                )
        references = str(result.get("final_references_text", "")).strip()
        if references:
            st.markdown("### References")
            st.code(references)

        st.markdown("### Feedback")
        feedback_key = f"{result.get('run_id', '')}_{result.get('thread_id', '')}"
        if st.session_state.last_feedback_key == feedback_key:
            st.info("Feedback already submitted for this query.")
        else:
            fb_col1, fb_col2 = st.columns([1, 2])
            with fb_col1:
                feedback_value = st.radio("Rating", options=["thumbs_up", "thumbs_down"], horizontal=True)
            with fb_col2:
                feedback_comment = st.text_input("Short feedback", placeholder="Optional short note")

            if st.button("Submit Feedback", type="primary"):
                current_input = st.session_state.current_input or {}
                success, message = write_feedback(
                    {
                        "run_id": result.get("run_id", ""),
                        "thread_id": result.get("thread_id", ""),
                        "rat": current_input.get("rat", ""),
                        "test_description": current_input.get("test_description", ""),
                        "feedback": feedback_value,
                        "feedback_comment": feedback_comment,
                        "sequence_diagram_file": result.get("sequence_diagram_file", ""),
                        "sequence_csv_file": result.get("sequence_csv_file", ""),
                        "langsmith_url": result.get("langsmith_url", ""),
                    },
                    fallback_file=DEPLOYMENT_DIR / "feedback" / "feedback_fallback.jsonl",
                )
                if success:
                    st.success(message)
                else:
                    st.warning(message)
                st.session_state.last_feedback_key = feedback_key

st.markdown("---")
st.caption(
    "Each new query requires Test Description and RAT. Choices are tracked in Decision Trail."
)
st.markdown("---")
st.caption(
    "*AI generated message sequence can make mistakes. Keep the latency in mind. Response generation may take some time.*"
)
