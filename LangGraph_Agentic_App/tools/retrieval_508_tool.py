from __future__ import annotations

from typing import Annotated

from langchain_core.tools import tool

from core.retrieval_508 import retrieve_508_context as retrieve_508_context_impl


@tool
def retrieve_508_context(
    rat: Annotated[str, "Target radio access type: 'lte' or 'nr'"],
    loop_path: Annotated[str, "Loop path string such as '1N-A -> 2N-A -> 3N-A'"] = "",
) -> dict:
    """Retrieve exact 36.508/38.508 context for a resolved UE-state loop path.

    Parameters
    ----------
    rat:
        Radio access type, either "lte" or "nr".
    loop_path:
        Ordered state path string, e.g. "State 1 -> State 2 -> State 3" or
        "1N-A -> 2N-A -> 3N-A".

    Returns
    -------
    dict
        Structured context payload containing per-transition/per-state chunks
        and a combined_context string for downstream extraction.
    """
    return retrieve_508_context_impl(rat=rat, loop_path=loop_path)