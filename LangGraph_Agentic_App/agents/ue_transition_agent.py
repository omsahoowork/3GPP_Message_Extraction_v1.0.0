"""UE state transition message extraction agent."""
from __future__ import annotations

from functools import lru_cache
import json
import re
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from core.llm import get_llm
from core.prompts import UE_TRANSITION_AGENT_SYSTEM_PROMPT, NR_RRC_TABLE_MESSAGE_PROMPT, LTE_TRANSITION_TABLE_MESSAGE_PROMPT
from tools.state_tools import get_ue_state_loop_path, retrieve_state_transition_context


@lru_cache(maxsize=8)
def create_ue_transition_agent(llm_provider: str, llm_model: str):
    """Create a ReAct agent for UE state transition extraction."""
    llm = get_llm(provider=llm_provider, model=llm_model, temperature=0.1)

    @tool
    def extract_messages_from_context(
        rat: Annotated[str, "RAT type: 'nr' or 'lte'"],
        state_or_transition: Annotated[str, "State ID (NR) or transition string (LTE)"],
        context: Annotated[str, "Table context to extract messages from"],
    ) -> dict:
        """Extract signalling messages from a state/transition context."""
        extraction_llm = get_llm(provider=llm_provider, model=llm_model, temperature=0.1)

        if rat == "nr":
            prompt = NR_RRC_TABLE_MESSAGE_PROMPT.format(
                rrc_lookup_query=state_or_transition,
                table_context=context,
            )
        else:
            prompt = LTE_TRANSITION_TABLE_MESSAGE_PROMPT.format(
                transition_query=state_or_transition,
                table_context=context,
            )

        raw = extraction_llm.invoke(prompt).content
        try:
            parsed = json.loads(raw)
            messages = parsed.get("message_sequence", []) if isinstance(parsed, dict) else []
            return {"message_sequence": messages if isinstance(messages, list) else []}
        except (json.JSONDecodeError, ValueError):
            return {"message_sequence": []}
    
    tools = [get_ue_state_loop_path, retrieve_state_transition_context, extract_messages_from_context]
    
    agent = create_react_agent(
        llm,
        tools,
        prompt=UE_TRANSITION_AGENT_SYSTEM_PROMPT,
    )
    return agent


def run_ue_transition_agent(
    ue_state: str,
    rat: str,
    llm_provider: str = "",
    llm_model: str = "",
    max_iterations: int = 20,
) -> dict:
    """Run the UE transition agent to extract state transition messages.

    Args:
        ue_state: UE state identifier
        rat: RAT type ('nr' or 'lte')
        max_iterations: Max reasoning steps

    Returns:
        Dict with keys expected by pipeline nodes:
        - transition_message_sequence
        - state_messages
        - loop_path
        - table_contexts
        - is_complete
        - error
    """
    prompt = f"""\
Extract messages for UE state transition:
- RAT: {rat}
- Target UE state: {ue_state}

Procedure:
1. Get the loop path for this UE state
2. For each state/transition in the loop path:
   a. Retrieve the context
   b. Extract messages from that context
3. Track which states are covered
4. Return all extracted messages

Validate that all states in the loop path are covered before finishing.
"""

    def _extract_text_fragments(content: object) -> list[str]:
        if isinstance(content, str):
            return [content]
        if isinstance(content, list):
            fragments: list[str] = []
            for item in content:
                if isinstance(item, str):
                    fragments.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content") or item.get("value")
                    if isinstance(text, str) and text.strip():
                        fragments.append(text)
            return fragments
        return []

    def _try_parse_json(text: str) -> object | None:
        raw = str(text or "").strip()
        if not raw:
            return None
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(line for line in lines if not line.strip().startswith("```"))
            raw = raw.strip()
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError, TypeError):
            return None

    def _coerce_loop_path(value: object) -> list[str]:
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        text = str(value or "").strip()
        if not text:
            return []
        return [part.strip() for part in re.split(r"->|→", text) if part.strip()]

    def _coerce_state_messages(value: object) -> list[dict]:
        if isinstance(value, list):
            rows: list[dict] = []
            for item in value:
                if not isinstance(item, dict):
                    continue
                rows.append({
                    "state": str(item.get("state", "")).strip(),
                    "messages": item.get("messages", []) if isinstance(item.get("messages", []), list) else [],
                })
            return rows
        if isinstance(value, dict):
            rows = []
            for state_name, messages in value.items():
                rows.append({
                    "state": str(state_name).strip(),
                    "messages": messages if isinstance(messages, list) else [],
                })
            return rows
        return []

    def _extract_from_markdown(fragment: str) -> tuple[list[dict], list[dict], list[str]]:
        """Best-effort fallback for markdown/table answers when JSON is absent."""
        text = str(fragment or "")
        if not text.strip():
            return [], [], []

        # Track current state by section/table headers such as "**1N-A**".
        current_state = ""
        state_map: dict[str, list[dict]] = {}
        flat_messages: list[dict] = []

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            # State title line examples:
            # | **1N‑A** | Step | Actor | ...
            # | **3N‑A** | Step | Actor | ...
            m_state = re.search(r"\*\*\s*([0-9][A-Z](?:[-‑][A-Z])?)\s*\*\*", line)
            if m_state and ("STATE" in line.upper() or "ACTOR" in line.upper() or line.startswith("|")):
                current_state = m_state.group(1).replace("‑", "-")
                state_map.setdefault(current_state, [])

            if "|" not in line:
                continue

            cells = [c.strip() for c in line.strip("|").split("|")]
            if not cells:
                continue

            # Skip header/divider rows.
            joined_upper = " ".join(cells).upper()
            if "ACTOR" in joined_upper and "MESSAGE" in joined_upper:
                continue
            if re.fullmatch(r"[-:\s]+", "".join(cells)):
                continue

            actor = ""
            for c in cells:
                token = c.strip().upper()
                if token in {"UE", "SS", "GNB", "ENB"}:
                    actor = token
                    break
            if not actor:
                continue

            # Prefer bold message text from table rows.
            bold_tokens = re.findall(r"\*\*([^*]+)\*\*", line)
            message_name = ""
            for token in bold_tokens:
                candidate = str(token).strip().replace("‑", "-")
                if not candidate or candidate in {"-", "–"}:
                    continue
                # Ignore state-like tokens captured in bold.
                if re.fullmatch(r"[0-9][A-Z](?:-[A-Z])?", candidate):
                    continue
                message_name = candidate
                break

            if not message_name:
                continue

            direction = "UE_TO_GNB" if actor == "UE" else "GNB_TO_UE"
            row = {"name": message_name, "direction": direction}
            flat_messages.append(row)
            if current_state:
                state_map.setdefault(current_state, []).append(row)

        state_rows = [
            {"state": state_name, "messages": messages}
            for state_name, messages in state_map.items()
        ]
        loop_states = [row["state"] for row in state_rows if row.get("state")]
        return flat_messages, state_rows, loop_states

    try:
        ue_transition_agent = create_ue_transition_agent(llm_provider, llm_model)
        result = ue_transition_agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config={"recursion_limit": max_iterations},
        )

        output = result.get("messages", []) if isinstance(result, dict) else []
        concatenated_messages: list[dict] = []
        state_messages: list[dict] = []
        loop_path: list[str] = []
        table_contexts: list[str] = []
        explicit_complete: bool | None = None
        error_reasons: list[str] = []
        parsed_any_json = False

        for msg in output:
            # Handles both dict-based messages and LangChain BaseMessage-like objects.
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            for fragment in _extract_text_fragments(content):
                parsed = _try_parse_json(fragment)
                if parsed is None:
                    md_messages, md_state_rows, md_loop_path = _extract_from_markdown(fragment)
                    if md_messages:
                        concatenated_messages.extend(md_messages)
                    if md_state_rows and not state_messages:
                        state_messages = md_state_rows
                    if md_loop_path and not loop_path:
                        loop_path = md_loop_path
                    continue
                parsed_any_json = True

                payloads = parsed if isinstance(parsed, list) else [parsed]
                for payload in payloads:
                    if not isinstance(payload, dict):
                        continue

                    for key in ("transition_message_sequence", "message_sequence", "ue_transition_message_sequence"):
                        messages = payload.get(key)
                        if isinstance(messages, list):
                            concatenated_messages.extend(item for item in messages if isinstance(item, dict))

                    candidate_state_rows = _coerce_state_messages(
                        payload.get("state_messages")
                        or payload.get("state_message_map")
                        or payload.get("ue_transition_state_messages")
                    )
                    if candidate_state_rows:
                        state_messages = candidate_state_rows

                    candidate_path = _coerce_loop_path(payload.get("loop_states") or payload.get("loop_path"))
                    if candidate_path:
                        loop_path = candidate_path

                    contexts = payload.get("table_contexts") or payload.get("ue_transition_table_contexts")
                    if isinstance(contexts, list):
                        table_contexts.extend(str(c).strip() for c in contexts if str(c).strip())

                    # Capture contexts directly from retrieval tool payloads.
                    for key in ("sequence_chunks", "state_chunks"):
                        chunks = payload.get(key)
                        if isinstance(chunks, list):
                            for chunk in chunks:
                                if isinstance(chunk, dict):
                                    ctx = str(chunk.get("context", "")).strip()
                                    if ctx:
                                        table_contexts.append(ctx)

                    if "is_complete" in payload and isinstance(payload.get("is_complete"), bool):
                        explicit_complete = payload.get("is_complete")

                    if payload.get("found") is False:
                        error_reasons.append("loop path not found")

                    payload_error = str(payload.get("error", "")).strip()
                    if payload_error:
                        error_reasons.append(payload_error)

                    missing_transitions = payload.get("missing_transitions")
                    if isinstance(missing_transitions, list) and missing_transitions:
                        error_reasons.append(
                            "missing LTE transition context for: "
                            + ", ".join(str(x) for x in missing_transitions)
                        )

                    missing_states = payload.get("missing_states")
                    if isinstance(missing_states, list) and missing_states:
                        error_reasons.append(
                            "missing NR state context for: "
                            + ", ".join(str(x) for x in missing_states)
                        )

        # De-duplicate while preserving order.
        dedup_contexts: list[str] = []
        for ctx in table_contexts:
            if ctx not in dedup_contexts:
                dedup_contexts.append(ctx)
        table_contexts = dedup_contexts

        dedup_errors: list[str] = []
        for reason in error_reasons:
            if reason and reason not in dedup_errors:
                dedup_errors.append(reason)

        if explicit_complete is not None:
            is_complete = explicit_complete
        elif dedup_errors:
            is_complete = False
        else:
            is_complete = bool(concatenated_messages)

        error_text = ""
        if not is_complete:
            if dedup_errors:
                error_text = " | ".join(dedup_errors)
            elif not parsed_any_json:
                error_text = "agent returned no parseable JSON payload"
            else:
                error_text = "no transition messages extracted"

        return {
            "transition_message_sequence": concatenated_messages,
            "state_messages": state_messages,
            "loop_path": loop_path,
            "table_contexts": table_contexts,
            "is_complete": is_complete,
            "error": error_text,
        }
    except Exception as e:
        return {
            "transition_message_sequence": [],
            "state_messages": [],
            "loop_path": [],
            "table_contexts": [],
            "is_complete": False,
            "error": f"UE transition agent error: {str(e)}",
        }
