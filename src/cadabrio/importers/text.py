"""Text importer for Cadabrio.

Handles text-based inputs: plain descriptions, prompts,
and structured specifications for AI-driven 3D generation.
"""

from dataclasses import dataclass


@dataclass
class TextInput:
    """A text input for AI-driven generation."""

    text: str
    intent: str = "generate"  # generate, modify, describe
    context: dict | None = None


def parse_text_input(text: str) -> TextInput:
    """Parse a text input and determine intent."""
    lower = text.lower().strip()

    if any(word in lower for word in ["modify", "change", "adjust", "edit", "update"]):
        intent = "modify"
    elif any(word in lower for word in ["describe", "what is", "explain"]):
        intent = "describe"
    else:
        intent = "generate"

    return TextInput(text=text, intent=intent)
