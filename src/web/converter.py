"""HTML to Markdown conversion."""

import re

from markdownify import markdownify


def html_to_markdown(html: str) -> str:
    """Convert an HTML string to clean Markdown."""
    if not html or not html.strip():
        return ""
    md = markdownify(html, heading_style="ATX", strip=["img", "script", "style"])
    # Collapse excessive blank lines
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()
