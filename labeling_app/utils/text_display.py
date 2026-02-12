"""Text display utilities for highlighting mentions in context."""

import re
import html
from typing import Optional


def highlight_mention(
    text: str,
    match_text: str,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> str:
    """
    Highlight the mention in the text.

    Args:
        text: The full text (paragraph or sentence)
        match_text: The text to highlight
        start: Optional character start position
        end: Optional character end position

    Returns:
        HTML string with highlighted mention
    """
    if not text:
        return ""

    # Escape HTML in the text
    escaped_text = html.escape(text)
    escaped_match = html.escape(match_text)

    # If we have exact positions, use them
    if start is not None and end is not None:
        try:
            before = html.escape(text[:start])
            mention = html.escape(text[start:end])
            after = html.escape(text[end:])
            return f'{before}<span class="mention-highlight">{mention}</span>{after}'
        except (IndexError, TypeError):
            pass

    # Fall back to regex-based highlighting
    pattern = re.compile(re.escape(escaped_match), re.IGNORECASE)
    highlighted = pattern.sub(
        f'<span class="mention-highlight">{escaped_match}</span>',
        escaped_text,
        count=1  # Only highlight first occurrence
    )

    return highlighted


def format_context_html(
    paragraph: str,
    sentence: str,
    match_text: str,
    start_in_sentence: Optional[int] = None,
    end_in_sentence: Optional[int] = None,
) -> str:
    """
    Format the context display with highlighted mention.

    Shows the sentence with the mention highlighted, with the full paragraph
    available for expansion.
    """
    # Highlight in sentence
    highlighted_sentence = highlight_mention(
        sentence, match_text, start_in_sentence, end_in_sentence
    )

    # Build HTML
    html_content = f"""
    <div class="mention-context">
        <p>{highlighted_sentence}</p>
    </div>
    """

    return html_content


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """Truncate text to a maximum length."""
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def format_speaker_info(
    speaker_canonical: Optional[str],
    speaker_bioguide: Optional[str],
    speaker_method: Optional[str],
    speaker_confidence: Optional[float],
) -> str:
    """Format speaker information for display."""
    if not speaker_canonical:
        return "Unknown Speaker"

    parts = [speaker_canonical]

    if speaker_bioguide:
        parts.append(f"({speaker_bioguide})")

    if speaker_confidence is not None:
        confidence_pct = int(speaker_confidence * 100)
        if confidence_pct < 70:
            parts.append(f"⚠️ {confidence_pct}% confidence")
        else:
            parts.append(f"✓ {confidence_pct}%")

    return " ".join(parts)


def format_match_type(is_acronym: bool, match_type: Optional[str] = None) -> str:
    """Format match type for display."""
    if is_acronym:
        return "🔤 Acronym"
    elif match_type:
        return f"📝 {match_type.title()}"
    else:
        return "📝 Name"


def get_context_window(paragraph: str, mention_start: int, mention_end: int, window_chars: int = 200) -> str:
    """
    Get a window of context around the mention.

    Args:
        paragraph: Full paragraph text
        mention_start: Start character position of mention
        mention_end: End character position of mention
        window_chars: Number of characters of context on each side

    Returns:
        Context window with mention highlighted
    """
    if not paragraph:
        return ""

    # Calculate window boundaries
    start = max(0, mention_start - window_chars)
    end = min(len(paragraph), mention_end + window_chars)

    # Adjust to word boundaries
    if start > 0:
        # Find the start of the word
        while start > 0 and paragraph[start - 1] not in ' \n\t':
            start -= 1
        prefix = "..."
    else:
        prefix = ""

    if end < len(paragraph):
        # Find the end of the word
        while end < len(paragraph) and paragraph[end] not in ' \n\t':
            end += 1
        suffix = "..."
    else:
        suffix = ""

    # Extract and format
    window = paragraph[start:end]
    mention_in_window_start = mention_start - start
    mention_in_window_end = mention_end - start

    highlighted = highlight_mention(
        window,
        paragraph[mention_start:mention_end],
        mention_in_window_start,
        mention_in_window_end
    )

    return f"{prefix}{highlighted}{suffix}"
