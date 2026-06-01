from __future__ import annotations

import json
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Make LangGraph_Agentic_App importable from deployment folder.
DEPLOYMENT_DIR = Path(__file__).resolve().parents[1]
RAG_ROOT_DIR = DEPLOYMENT_DIR.parent
AGENTIC_APP_DIR = RAG_ROOT_DIR / "LangGraph_Agentic_App"


def _load_environment_like_agentic_app() -> None:
    """Mirror API-key and LangSmith env initialization from Agentic_App/config.py.

    Priority:
    1. Existing process env
    2. .env at workspace root (same convention as Agentic_App)
    3. Streamlit secrets (for cloud deployment)
    """
    try:
        import dotenv

        # Load from most likely locations first.
        candidate_env_paths = [
            RAG_ROOT_DIR / ".env",  # Actual project location in this workspace.
            DEPLOYMENT_DIR / ".env",
            DEPLOYMENT_DIR.parent.parent.parent / ".env",  # Agentic_App pattern fallback.
            DEPLOYMENT_DIR.parent.parent / ".env",  # Agentic_App pattern fallback.
            DEPLOYMENT_DIR.parent / ".env",  # Agentic_App pattern fallback.
        ]
        seen: set[str] = set()
        for env_path in candidate_env_paths:
            path_key = str(env_path.resolve())
            if path_key in seen:
                continue
            seen.add(path_key)
            if env_path.exists():
                dotenv.load_dotenv(dotenv_path=env_path, override=False)
    except Exception:
        # Keep startup resilient when python-dotenv is unavailable.
        pass

    secrets: dict[str, Any] = {}
    try:
        import streamlit as st

        secrets = dict(st.secrets)
    except Exception:
        secrets = {}

    def _pick(key: str, default: str = "") -> str:
        env_value = str(os.getenv(key, "")).strip()
        if env_value:
            return env_value
        secret_value = secrets.get(key)
        return str(secret_value).strip() if secret_value is not None else default

    def _pick_alias(keys: list[str], default: str = "") -> str:
        for key in keys:
            value = _pick(key, "")
            if value:
                return value
        return default

    # API keys used by Agentic_App/config.py
    os.environ["ANTHROPIC_API_KEY"] = _pick("ANTHROPIC_API_KEY", "")
    os.environ["OLLAMA_API_KEY"] = _pick("OLLAMA_API_KEY", "")

    # LangSmith / LangChain tracing variables and aliases.
    tracing_value = _pick_alias(["LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING"], "true")
    endpoint_value = _pick_alias(
        ["LANGCHAIN_ENDPOINT", "LANGSMITH_ENDPOINT"],
        "https://api.smith.langchain.com",
    )
    api_key_value = _pick_alias(["LANGCHAIN_API_KEY", "LANGSMITH_API_KEY"], "")
    project_value = _pick_alias(["LANGCHAIN_PROJECT", "LANGSMITH_PROJECT"], "3gpp-pipeline")

    os.environ["LANGCHAIN_TRACING_V2"] = tracing_value
    os.environ["LANGCHAIN_ENDPOINT"] = endpoint_value
    os.environ["LANGCHAIN_API_KEY"] = api_key_value
    os.environ["LANGCHAIN_PROJECT"] = project_value

    # Guard: hosted LangSmith tracing requires API key.
    if not os.environ["LANGCHAIN_API_KEY"].strip():
        os.environ["LANGCHAIN_TRACING_V2"] = "false"

    os.environ["LANGSMITH_TRACING"] = os.environ["LANGCHAIN_TRACING_V2"]
    os.environ["LANGSMITH_ENDPOINT"] = os.environ["LANGCHAIN_ENDPOINT"]
    os.environ["LANGSMITH_API_KEY"] = os.environ["LANGCHAIN_API_KEY"]
    os.environ["LANGSMITH_PROJECT"] = os.environ["LANGCHAIN_PROJECT"]


_load_environment_like_agentic_app()

if str(AGENTIC_APP_DIR) not in sys.path:
    sys.path.insert(0, str(AGENTIC_APP_DIR))

def _get_langgraph_api() -> dict[str, Any]:
    """Import LangGraph modules lazily so app UI can render before API key entry."""
    from langgraph_pipeline import run_pipeline, resume_pipeline  # noqa: E402
    from langgraph_pipeline.graph import (  # noqa: E402
        get_pending_interrupt_node,
        get_pending_interrupt_payload,
        is_awaiting_selection,
    )

    return {
        "run_pipeline": run_pipeline,
        "resume_pipeline": resume_pipeline,
        "get_pending_interrupt_node": get_pending_interrupt_node,
        "get_pending_interrupt_payload": get_pending_interrupt_payload,
        "is_awaiting_selection": is_awaiting_selection,
    }


@dataclass
class PipelineSnapshot:
    thread_id: str | None
    result: dict[str, Any]
    awaiting_selection: bool
    pending_node: str | None
    pending_options: list[dict[str, Any]]


def _normalize_options(payload: object) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        return []
    options: list[dict[str, Any]] = []
    for idx, item in enumerate(payload):
        if isinstance(item, dict):
            option = dict(item)
            option.setdefault("option_index", idx)
            options.append(option)
        else:
            options.append({"option_index": idx, "label": str(item)})
    return options


def write_runtime_query_config(query_config: dict[str, Any], *, run_id: str | None = None) -> Path:
    runtime_dir = DEPLOYMENT_DIR / "runtime_configs"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    suffix = run_id or str(uuid.uuid4())
    target = runtime_dir / f"query_config_{suffix}.json"
    with target.open("w", encoding="utf-8") as handle:
        json.dump(query_config, handle, indent=2, ensure_ascii=False)
    return target


def start_pipeline(
    config_path: Path,
    *,
    api_key: str,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> PipelineSnapshot:
    api = _get_langgraph_api()
    result = api["run_pipeline"](
        query_config_path=str(config_path),
        initial_state={"openai_api_key": str(api_key or "").strip()},
        tags=tags or ["streamlit", "agentic-app"],
        metadata=metadata or {},
    )
    return capture_snapshot(result)


def continue_pipeline(thread_id: str, selected_index: int) -> PipelineSnapshot:
    api = _get_langgraph_api()
    result = api["resume_pipeline"](thread_id, int(selected_index))
    return capture_snapshot(result)


def capture_snapshot(result: dict[str, Any]) -> PipelineSnapshot:
    api = _get_langgraph_api()
    thread_id = str(result.get("thread_id", "")).strip() or None
    awaiting = bool(thread_id and api["is_awaiting_selection"](thread_id))
    pending_node = api["get_pending_interrupt_node"](thread_id) if awaiting and thread_id else None
    pending_payload = api["get_pending_interrupt_payload"](thread_id) if awaiting and thread_id else []
    pending_options = _normalize_options(pending_payload)

    return PipelineSnapshot(
        thread_id=thread_id,
        result=result,
        awaiting_selection=awaiting,
        pending_node=pending_node,
        pending_options=pending_options,
    )


def node_status_text(node_name: str | None) -> str:
    mapping = {
        "select_context": "Shortlisted candidate tests are ready. Choose one to continue.",
        "select_final_sibling_section": "Sibling sections are ready. Pick the final section.",
        "select_test_purpose": "Test purpose options are ready. Pick one to launch extraction.",
    }
    if not node_name:
        return "Pipeline running or completed."
    return mapping.get(node_name, f"Waiting for selection at node: {node_name}")
