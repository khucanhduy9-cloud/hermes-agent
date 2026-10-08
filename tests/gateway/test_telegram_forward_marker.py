"""Tests for the forwarded-message marker on inbound Telegram text.

A forward carries someone else's words, so the event text (what the model reads and what the
session transcript stores) starts with ``[Forwarded message]``. The origin's name never appears.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.event import MessageType
from plugins.platforms.telegram.telegram_entities import FORWARD_MARKER, inbound_text, is_forwarded

_WHEN = datetime(2026, 9, 27, 8, 0, tzinfo=timezone.utc)


def _message(text="Chai này ngon lắm", caption=None, **extra):
    base = dict(text=text, caption=caption, entities=[], caption_entities=[])
    base.update(extra)
    return SimpleNamespace(**base)


def _user_origin():
    return SimpleNamespace(type="user", date=_WHEN, sender_user=SimpleNamespace(full_name="Someone Else"))


def test_typed_message_is_unchanged():
    assert inbound_text(_message()) == "Chai này ngon lắm"
    assert not is_forwarded(_message())


def test_forward_origin_gets_marker_and_no_origin_name():
    out = inbound_text(_message(forward_origin=_user_origin()))
    assert out == f"{FORWARD_MARKER}\n\nChai này ngon lắm"
    assert "Someone Else" not in out


def test_legacy_forward_date_gets_marker():
    assert inbound_text(_message(forward_date=_WHEN, forward_sender_name="X")).startswith(FORWARD_MARKER)


def test_automatic_forward_gets_marker():
    assert inbound_text(_message(is_automatic_forward=True)).startswith(FORWARD_MARKER)


def test_forwarded_caption_gets_marker():
    msg = _message(text=None, caption="ảnh nhãn", forward_origin=_user_origin())
    assert inbound_text(msg) == f"{FORWARD_MARKER}\n\nảnh nhãn"


def test_forward_without_text_is_marker_only():
    assert inbound_text(_message(text=None, forward_origin=_user_origin())) == FORWARD_MARKER


def test_attribute_autocreating_double_is_not_a_forward():
    msg = MagicMock()
    msg.text = "hello"
    msg.entities = []
    assert not is_forwarded(msg)
    assert inbound_text(msg) == "hello"


def _dm_message(**extra):
    chat = SimpleNamespace(id=4242, type="private", title=None, full_name="Owner", is_forum=False)
    user = SimpleNamespace(id=4242, full_name="Owner", is_bot=False)
    base = dict(
        chat=chat, from_user=user, text="Chai này ngon lắm", caption=None, entities=[], caption_entities=[],
        message_thread_id=None, is_topic_message=False, message_id=7, reply_to_message=None, quote=None,
        date=None, forum_topic_created=None)
    base.update(extra)
    return SimpleNamespace(**base)


@pytest.fixture
def adapter():
    pytest.importorskip("telegram")
    from plugins.platforms.telegram.adapter import TelegramAdapter
    return TelegramAdapter(PlatformConfig(enabled=True, token="***", extra={}))


def test_build_message_event_marks_forward(adapter):
    event = adapter._build_message_event(_dm_message(forward_origin=_user_origin()), MessageType.TEXT)
    assert event.text == f"{FORWARD_MARKER}\n\nChai này ngon lắm"


def test_build_message_event_leaves_typed_text(adapter):
    event = adapter._build_message_event(_dm_message(), MessageType.TEXT)
    assert event.text == "Chai này ngon lắm"
