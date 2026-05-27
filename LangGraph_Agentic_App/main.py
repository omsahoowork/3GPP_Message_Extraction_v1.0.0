from dotenv import load_dotenv
load_dotenv() 
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
from langgraph_pipeline import run_pipeline, resume_pipeline
from langgraph_pipeline.graph import (
    is_awaiting_selection,
    get_pending_interrupt_payload,
    get_pending_interrupt_node,
)


def _coerce_test_purpose_options(value):
    options = []

    def _append(option):
        text = ""
        if isinstance(option, dict):
            for key in ("test_purpose", "purpose", "name", "title", "text", "description"):
                candidate = str(option.get(key, "")).strip()
                if candidate:
                    text = candidate
                    break
        else:
            text = str(option or "").strip()

        if text:
            text = text.lstrip("-* ")
        if text and text not in options:
            options.append(text)

    if isinstance(value, list):
        for item in value:
            _append(item)
        return options

    text = str(value or "").strip()
    if not text or text in {"-", "none", "None", "[]"}:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            for item in parsed:
                _append(item)
            return options

    lines = [line.strip().lstrip("-* ") for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        for item in lines:
            _append(item)
        return options

    if ";" in text:
        for item in text.split(";"):
            _append(item)
        if options:
            return options

    marker_count = len(re.findall(r"(?:^|\s)\d+[.)]\s+", text))
    if marker_count >= 2:
        numbered_parts = [
            part.strip()
            for part in re.split(r"(?:^|\s)\d+[.)]\s+", text)
            if part.strip()
        ]
        for item in numbered_parts:
            _append(item)
        if len(options) > 1:
            return options

    _append(text)
    return options


def _get_test_purpose_value(context_json):
    if not isinstance(context_json, dict):
        return []
    for key in (
        "test_purposes",
        "test_purpose",
        "testPurpose",
        "test_purpose_list",
        "test purpose",
        "test purposes",
    ):
        if key in context_json and context_json.get(key) not in (None, ""):
            return context_json.get(key)
    return []

if __name__ == "__main__":
    result = run_pipeline(
        query_config_path=config.QUERY_CONFIG_PATH
    )

    thread_id = result.get("thread_id")

    while thread_id and is_awaiting_selection(thread_id):
        pending_node = get_pending_interrupt_node(thread_id)
        pending_payload = get_pending_interrupt_payload(thread_id)

        print("\n" + "=" * 80)
        if pending_node == "select_context":
            print("SHORTLISTED CONTEXT OPTIONS")
        elif pending_node == "select_final_sibling_section":
            print("SIBLING SECTION OPTIONS (Final Selection)")
        elif pending_node == "select_test_purpose":
            print("TEST PURPOSE OPTIONS")
        else:
            print(f"PENDING USER SELECTION ({pending_node or 'unknown node'})")
        print("=" * 80)

        options = pending_payload if isinstance(pending_payload, list) else []

        if not options and pending_node == "select_test_purpose":
            options = _coerce_test_purpose_options(
                _get_test_purpose_value(result.get("selected_context_json") or {})
            )
            options = [
                {"option_index": idx, "test_purpose": text}
                for idx, text in enumerate(options)
            ]

        if not options:
            print("ERROR: Pipeline is awaiting input, but no selectable options were found.")
            sys.exit(1)

        for idx, option in enumerate(options):
            print(f"\n--- Option {idx} ---")
            if isinstance(option, dict):
                print(json.dumps(option, indent=2, ensure_ascii=False))
            else:
                print(str(option))

        print("=" * 80)
        print("Enter the option index to continue extraction:")
        print("=" * 80 + "\n")
        sys.stdout.flush()
        try:
            selected_index = int(input("Option index: ").strip())
        except ValueError:
            print("ERROR: Invalid input. Must be an integer.")
            sys.exit(1)

        result = resume_pipeline(thread_id, selected_index)

    if result.get("error"):
        print(f"\nERROR: {result['error']}")
        sys.exit(1)

    print("\nFinal message sequence:")
    print(json.dumps(result.get("message_sequence", []), indent=2))
