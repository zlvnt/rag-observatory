from __future__ import annotations
from pathlib import Path
from typing import Optional
import json
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


def _get_llm(model_name: str = "gemini-pro", temperature: float = 0.7) -> ChatGoogleGenerativeAI:
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


def _load_personality_config(config_path: Optional[Path]) -> dict:
    """Load personality configuration from JSON file.

    Args:
        config_path: Path to personality config JSON file (optional)

    Returns:
        Personality configuration dictionary
    """
    if config_path is None or not config_path.exists():
        # Return simple default personality
        print("INFO: Using default personality config")
        return {
            "identity": {
                "name": "AI Assistant",
                "company": "Your Business"
            },
            "service_guidelines": [
                "Be helpful and professional",
                "Provide clear and accurate information",
                "Use a friendly tone"
            ],
            "reply_template": "Identity: {identity_name} from {company}\n\n{service_guidelines}\n\nUser Query: {query}\n\nContext: {context}\n\nResponse:"
        }

    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        print(f"INFO: Loaded personality config from {config_path}")
        return config
    except Exception as e:
        print(f"WARNING: Failed to load personality config from {config_path}: {e}")
        # Return default
        return _load_personality_config(None)


def generate_reply(
    query: str,
    context: str = "",
    personality_config_path: Optional[Path] = None,
    model_name: str = "gemini-pro",
    temperature: float = 0.7,
    verbose: bool = False
) -> str:
    """Generate reply using LLM with personality configuration.

    Args:
        query: User query/question
        context: Retrieved context from RAG system
        personality_config_path: Path to personality config JSON (optional)
        model_name: Gemini model name
        temperature: LLM temperature
        verbose: Print debug information

    Returns:
        Generated reply string
    """
    try:
        # Load personality config
        config = _load_personality_config(personality_config_path)

        # Extract identity and guidelines
        identity = config.get("identity", {})
        guidelines = config.get("service_guidelines", [])

        # Format guidelines
        guidelines_text = "\n".join([f"- {g}" for g in guidelines])

        # Get template
        template_str = config.get(
            "reply_template",
            "Identity: {identity_name} from {company}\n\n{service_guidelines}\n\nUser Query: {query}\n\nContext: {context}\n\nResponse:"
        )

        # Format template variables
        template_vars = {
            "query": query,
            "context": context or "No additional information available.",
            "identity_name": identity.get("name", "AI Assistant"),
            "company": identity.get("company", "Your Business"),
            "service_guidelines": guidelines_text
        }

        # Create prompt and invoke LLM
        prompt_template = ChatPromptTemplate.from_template(template_str)
        messages = prompt_template.format_messages(**template_vars)

        if verbose:
            print(f"{'='*60}")
            print("PROMPT TO LLM:")
            print(messages[0].content)
            print(f"{'='*60}")

        ai_msg = _get_llm(model_name, temperature).invoke(messages)
        reply = ai_msg.content.strip()
        print(f"INFO: Generated reply successfully")

    except Exception as e:
        print(f"ERROR: Reply generation failed - error: {e}")
        reply = "Sorry, I encountered an issue processing your message. Please try again."

    return reply
