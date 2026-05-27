from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _append_local_fallback(payload: dict[str, Any], fallback_file: Path) -> None:
    fallback_file.parent.mkdir(parents=True, exist_ok=True)
    with fallback_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_feedback(payload: dict[str, Any], *, fallback_file: Path | None = None) -> tuple[bool, str]:
    record = dict(payload)
    record.setdefault("timestamp_utc", _utc_now_iso())

    run_id = str(record.get("run_id", "")).strip()
    if not run_id:
        if fallback_file is not None:
            _append_local_fallback(record, fallback_file)
        return (False, "Missing run_id; cannot submit feedback to LangSmith.")

    feedback_raw = str(record.get("feedback", "")).strip().lower()
    score = 1.0 if feedback_raw == "thumbs_up" else 0.0
    comment = str(record.get("feedback_comment", "")).strip()

    try:
        from langsmith import Client

        client = Client()
        client.create_feedback(
            run_id=run_id,
            key="user_feedback",
            score=score,
            value={
                "feedback": feedback_raw,
                "rat": str(record.get("rat", "")),
                "test_description": str(record.get("test_description", "")),
                "sequence_diagram_file": str(record.get("sequence_diagram_file", "")),
                "sequence_csv_file": str(record.get("sequence_csv_file", "")),
                "timestamp_utc": str(record.get("timestamp_utc", "")),
            },
            comment=comment,
            source_info={
                "source": "streamlit-app",
                "thread_id": str(record.get("thread_id", "")),
            },
        )
        return (True, "Feedback submitted to LangSmith run.")
    except Exception as exc:
        if fallback_file is not None:
            record["langsmith_error"] = str(exc)
            _append_local_fallback(record, fallback_file)
            return (False, f"LangSmith feedback write failed; saved locally ({exc}).")
        return (False, f"LangSmith feedback write failed ({exc}).")
