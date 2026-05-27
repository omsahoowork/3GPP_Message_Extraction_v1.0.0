# langgraph_pipeline/__init__.py
"""Public API for the langgraph_pipeline package.

Preferred entry point
─────────────────────
``run_pipeline()`` is the recommended way to invoke the graph.  It handles
LangSmith run_id assignment, config wiring, and URL generation automatically.

    from langgraph_pipeline import run_pipeline

    result = run_pipeline(
        query_config_path="query_config.json",
        tags=["experiment-1"],
        metadata={"dataset": "38.331-h60"},
    )
    print(result["message_sequence"])
    print(result["langsmith_url"])    # open in browser

Direct pipeline access
──────────────────────
``pipeline`` is the compiled LangGraph instance.  Use it when you need direct
access (e.g. streaming, async invocation, or custom configs):

    from langgraph_pipeline import pipeline
    result = pipeline.invoke({"query_config_path": "query_config.json"})

Graph introspection
───────────────────
``build_graph()`` returns the *uncompiled* StateGraph.  Use it in tests or to
visualise the graph structure without triggering model loading:

    from langgraph_pipeline import build_graph
    g = build_graph()
    g.get_graph().print_ascii()
"""

from langgraph_pipeline.state import PipelineState
from langgraph_pipeline.graph import pipeline, build_graph, run_pipeline, resume_pipeline, is_awaiting_selection

__all__ = [
    "PipelineState",
    "pipeline",
    "build_graph",
    "run_pipeline",
    "resume_pipeline",
    "is_awaiting_selection",
]

