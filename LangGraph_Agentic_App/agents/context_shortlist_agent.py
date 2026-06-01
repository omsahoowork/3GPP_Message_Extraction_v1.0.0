"""Context selection and ranking agent."""
from __future__ import annotations

import json

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from config import LLM_MODEL, AGENT_MAX_RETRIES
from core.prompts import CONTEXT_SHORTLIST_AGENT_SYSTEM_PROMPT
from tools.extraction_tool import generate_context_fields_json


@tool
def rank_contexts(
    query_config_json: str,
    question: str,
    contexts_json: str,
    openai_api_key: str = "",
) -> dict:
    """Rank and select relevant contexts from a list of candidates.

    Args:
        query_config_json: JSON string of user config
        question: Enhanced query question
        contexts_json: JSON string of contexts with option_index

    Returns:
        Dict with selected_option_indices (list of ints)
    """
    from core.prompts import CONTEXT_OPTION_INDEX_SHORTLIST_PROMPT
    
    # llm = ChatOllama(model=LLM_MODEL, base_url="https://api.ollama.com", temperature=0.1)
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.1, api_key=str(openai_api_key or "").strip())
    
    prompt = CONTEXT_OPTION_INDEX_SHORTLIST_PROMPT.format(
        query_config=query_config_json,
        question=question,
        raw_contexts_json=contexts_json,
    )
    
    raw = llm.invoke(prompt).content
    try:
        parsed = json.loads(raw)
        selected = parsed.get("selected_option_indices", [])
        return {"selected_option_indices": selected if isinstance(selected, list) else []}
    except (json.JSONDecodeError, ValueError):
        return {"selected_option_indices": []}


def create_shortlist_agent(openai_api_key: str):
    """Create a ReAct agent for context shortlisting."""
    # llm = ChatOllama(model=LLM_MODEL, base_url="https://api.ollama.com", temperature=0.1)
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.1, api_key=str(openai_api_key or "").strip())
    # llm = ChatAnthropic(model=LLM_MODEL, temperature=0.1)

    @tool
    def rank_contexts_with_key(
        query_config_json: str,
        question: str,
        contexts_json: str,
    ) -> dict:
        from core.prompts import CONTEXT_OPTION_INDEX_SHORTLIST_PROMPT

        rank_llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0.1,
            api_key=str(openai_api_key or "").strip(),
        )
        prompt = CONTEXT_OPTION_INDEX_SHORTLIST_PROMPT.format(
            query_config=query_config_json,
            question=question,
            raw_contexts_json=contexts_json,
        )
        raw = rank_llm.invoke(prompt).content
        try:
            parsed = json.loads(raw)
            selected = parsed.get("selected_option_indices", [])
            return {"selected_option_indices": selected if isinstance(selected, list) else []}
        except (json.JSONDecodeError, ValueError):
            return {"selected_option_indices": []}
    
    tools = [rank_contexts_with_key]
    
    agent = create_react_agent(
        llm,
        tools,
        prompt=CONTEXT_SHORTLIST_AGENT_SYSTEM_PROMPT,
    )
    return agent


def run_shortlist_agent(
    raw_contexts: list[str],
    raw_source_docs: list[str],
    query_config: dict,
    question: str,
    openai_api_key: str,
    max_iterations: int = 10,
) -> dict:
    """Run the shortlist agent to filter and extract context fields.

    Args:
        raw_contexts: List of raw context strings
        raw_source_docs: List of source doc references
        query_config: User config dict
        question: Enhanced question
        max_iterations: Max reasoning steps

    Returns:
        Dict with:
        - shortlisted_raw_indices: list of selected indices in raw_contexts
        - shortlisted_enhanced_contexts: list of extracted context JSON dicts
        - shortlisted_comparison_json: list of display rows
    """
    if not raw_contexts:
        return {
            "shortlisted_raw_indices": [],
            "shortlisted_enhanced_contexts": [],
            "shortlisted_comparison_json": [],
        }

    # First, rank the contexts
    indexed_contexts = [
        {"option_index": i, "context": ctx}
        for i, ctx in enumerate(raw_contexts)
    ]
    
    prompt = f"""\
Rank these {len(raw_contexts)} contexts for relevance to the user query.

User config:
{json.dumps(query_config, indent=2)}

Question:
{question}

Candidate contexts:
{json.dumps(indexed_contexts, indent=2)}

Use the rank_contexts tool to select the relevant ones.
"""
    
    try:
        ranking_result = create_shortlist_agent(openai_api_key).invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config={"recursion_limit": max_iterations},
        )
        
        # Extract selected indices from agent result
        output = ranking_result.get("messages", [])
        selected_indices = []
        for msg in output:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str):
                    try:
                        parsed = json.loads(content)
                        selected = parsed.get("selected_option_indices", [])
                        if isinstance(selected, list):
                            selected_indices = selected
                            break
                    except (json.JSONDecodeError, ValueError):
                        pass
        
        if not selected_indices:
            selected_indices = list(range(min(5, len(raw_contexts))))  # Default fallback
    except Exception as e:
        print(f"Shortlist agent ranking error: {e}")
        selected_indices = list(range(min(5, len(raw_contexts))))

    # Now extract schema fields for each selected context
    shortlisted_enhanced_contexts = []
    formatted_options = []

    for display_idx, raw_idx in enumerate(selected_indices):
        if raw_idx >= len(raw_contexts):
            continue
            
        ctx = raw_contexts[raw_idx]
        source_doc = raw_source_docs[raw_idx] if raw_idx < len(raw_source_docs) else ""

        # Extract context JSON with retry
        context_json = _extract_context_with_retry(
            ctx,
            openai_api_key=openai_api_key,
            max_retries=AGENT_MAX_RETRIES,
        )

        enhanced = {
            "option_index": display_idx,
            "original_option_index": raw_idx,
            "source_doc": source_doc,
            "context_json": context_json,
        }
        shortlisted_enhanced_contexts.append(enhanced)

        # Build display row
        display_row: dict = {
            "option_index": display_idx,
            "original_option_index": raw_idx,
            "source_doc": source_doc,
        }
        for k, v in context_json.items():
            display_row[k] = v if v else "-"
        formatted_options.append(display_row)

    return {
        "shortlisted_raw_indices": selected_indices,
        "shortlisted_enhanced_contexts": shortlisted_enhanced_contexts,
        "shortlisted_comparison_json": formatted_options,
    }


def _extract_context_with_retry(context: str, *, openai_api_key: str, max_retries: int = 3) -> dict:
    """Extract context fields with JSON parse retry."""
    for attempt in range(max_retries):
        try:
            context_json = generate_context_fields_json.invoke(
                {
                    "context": context,
                    "openai_api_key": str(openai_api_key or "").strip(),
                }
            )
            if isinstance(context_json, dict):
                return context_json
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Context extraction failed after {max_retries} retries: {e}")
                return {}
            # Retry on error
            continue
    
    return {}
