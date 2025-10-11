from __future__ import annotations
from pathlib import Path
from typing import Optional
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


def _get_llm(model_name: str = "gemini-pro", temperature: float = 0) -> ChatGoogleGenerativeAI:
    """Get LLM instance with configuration.

    Args:
        model_name: Gemini model name to use
        temperature: LLM temperature setting

    Returns:
        ChatGoogleGenerativeAI instance
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")

    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=api_key,
    )


def supervisor_route(
    user_input: str,
    supervisor_prompt_path: Path,
    model_name: str = "gemini-pro",
    temperature: float = 0,
    history_context: str = ""
) -> str:
    """Route user query to appropriate mode using supervisor prompt.

    Args:
        user_input: User query
        supervisor_prompt_path: Path to supervisor prompt template file
        model_name: Gemini model name
        temperature: LLM temperature
        history_context: Previous conversation context

    Returns:
        Routing mode: "docs", "web", "all", or "direct"
    """
    routing_mode = "direct"  # default

    try:
        # Load prompt template from file
        prompt_text = supervisor_prompt_path.read_text(encoding="utf-8")
        supervisor_prompt = ChatPromptTemplate.from_template(prompt_text)

        msg = supervisor_prompt.format_messages(
            user_input=user_input,
            history_context=history_context or "No previous conversation"
        )
        decision = _get_llm(model_name, temperature).invoke(msg).content.strip().lower()
        
        # Debug logging
        print(f"DEBUG: Supervisor decision: '{decision}' for query: '{user_input[:50]}...'")

        # Map supervisor decision to internal routing
        if decision.startswith(("internal_doc", "rag", "docs")):
            routing_mode = "docs"
        elif decision.startswith(("web_search", "websearch", "web")):
            routing_mode = "web"
        elif decision.startswith("all"):
            routing_mode = "all"
        else:
            routing_mode = "direct"

    except Exception as e:
        print(f"ERROR: Supervisor routing failed: {e}")
        routing_mode = "direct"  # fallback

    return routing_mode
