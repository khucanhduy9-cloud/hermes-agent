"""Telegram ``text_link`` expansion, kept out of the adapter facade.

A rich-text hyperlink arrives as visible anchor text plus a ``text_link`` entity carrying the
URL; the model only ever sees the anchor unless the URL is inlined (#31071).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

# Marker line prefixed to the text of a forwarded message. A forward carries someone else's words
# (or the user's own, re-shared), so the model and the stored transcript must not read it as
# something the user just typed. The origin's name is deliberately left out.
FORWARD_MARKER = "[Forwarded message]"


def _utf16_length(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def _code_point_index(text: str, utf16_offset: int) -> int | None:
    """Python index for a UTF-16 code-unit offset; None when it splits a surrogate pair."""
    try:
        return len(text.encode("utf-16-le")[: utf16_offset * 2].decode("utf-16-le"))
    except UnicodeDecodeError:
        return None


def expand_link_entities(message: Any) -> str:
    """Message text (or caption) with every hidden ``text_link`` URL inlined after its anchor.

    Entity offsets are UTF-16 code units (emoji before the anchor count twice), so they are mapped
    to code-point indices before slicing. Malformed entities are skipped; an anchor that already
    reads as its own URL, or text already carrying the inline form, is left untouched.
    """
    text = getattr(message, "text", None)
    if text:
        entities = getattr(message, "entities", None) or []
    else:
        text = getattr(message, "caption", None) or ""
        entities = getattr(message, "caption_entities", None) or []
    if not text or not entities:
        return text

    utf16_length = _utf16_length(text)
    links: list[tuple[int, int, str]] = []
    for entity in entities:
        entity_type = str(getattr(entity, "type", "")).split(".")[-1].lower()
        raw_url = getattr(entity, "url", None)
        url = raw_url.strip() if isinstance(raw_url, str) else ""
        if entity_type != "text_link" or not url:
            continue
        try:
            offset = int(getattr(entity, "offset", -1))
            length = int(getattr(entity, "length", 0))
        except (TypeError, ValueError):
            continue
        if offset < 0 or length <= 0 or offset + length > utf16_length:
            continue
        start, end = _code_point_index(text, offset), _code_point_index(text, offset + length)
        if start is None or end is None or end <= start:
            continue
        if text[start:end].strip() == url:
            continue
        links.append((start, end, url))

    expanded = text
    for _start, end, url in sorted(links, reverse=True):
        inline = f" ({url})"
        if expanded[end:].startswith(inline):
            continue
        expanded = f"{expanded[:end]}{inline}{expanded[end:]}"
    return expanded


def is_forwarded(message: Any) -> bool:
    """True when Telegram delivered ``message`` as a forward (or a channel auto-forward).

    Every forward carries the original send date: ``forward_origin.date`` on Bot API 7+ and the
    legacy ``forward_date`` before it, so a real ``datetime`` there is the signal. Checking the
    type (not mere truthiness) keeps attribute-auto-creating test doubles from reading as forwards.
    """
    if getattr(message, "is_automatic_forward", None) is True:
        return True
    origin = getattr(message, "forward_origin", None)
    if origin is not None and isinstance(getattr(origin, "date", None), datetime):
        return True
    return isinstance(getattr(message, "forward_date", None), datetime)


def inbound_text(message: Any) -> str:
    """Event text for an inbound message: ``expand_link_entities`` plus the forward marker line.

    A forward with no text or caption (bare photo, voice note) still gets the marker, so later
    enrichment (vision description, voice transcript) lands in a row that says it was forwarded.
    """
    text = expand_link_entities(message)
    if not is_forwarded(message):
        return text
    return f"{FORWARD_MARKER}\n\n{text}" if text else FORWARD_MARKER
