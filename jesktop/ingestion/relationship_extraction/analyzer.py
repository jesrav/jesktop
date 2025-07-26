"""Analysis functions for relationship strength and context extraction."""

import re


def calculate_relationship_strength(source_content: str, target_name: str) -> float:
    """Calculate relationship strength based on frequency and context.

    Args:
        source_content: Full content of the source note
        target_name: Name of the target note

    Returns:
        Relationship strength between 0.0 and 1.0
    """
    # Count occurrences of the target in the source
    occurrences = len(re.findall(re.escape(target_name), source_content, re.IGNORECASE))

    # Base strength on frequency, capped at 1.0
    base_strength = min(occurrences * 0.3, 1.0)

    # Boost if mentioned in headers
    header_mentions = len(
        re.findall(f"#{1, 6}.*{re.escape(target_name)}", source_content, re.IGNORECASE)
    )
    header_boost = header_mentions * 0.2

    return min(base_strength + header_boost, 1.0)


def extract_relationship_context(content: str, target_name: str, context_chars: int = 100) -> str:
    """Extract surrounding context for a relationship mention.

    Args:
        content: Full content of the note
        target_name: Name of the target being referenced
        context_chars: Number of characters before/after to include

    Returns:
        Context string around the first mention
    """
    # Find first mention of target
    match = re.search(re.escape(target_name), content, re.IGNORECASE)
    if not match:
        return ""

    start = max(0, match.start() - context_chars)
    end = min(len(content), match.end() + context_chars)

    context = content[start:end].strip()

    # Clean up context - remove newlines, extra spaces
    context = re.sub(r"\s+", " ", context)

    return context
