from __future__ import annotations

from qq_llm_bot.onebot_message_filters import (
    strip_forwarded_records,
    strip_quoted_messages,
)
from qq_llm_bot.onebot_message_rendering import render_message_text_and_mentions_with_forwards
from qq_llm_bot.onebot_message_segments import (
    coerce_message_segments,
    format_mention_label,
    image_attachments_from_message,
    image_attachments_from_payload,
    parse_outgoing_mention_parts,
    render_message_text_and_mentions,
    reply_segment_ids,
    segment_data,
    segment_type,
)
from qq_llm_bot.onebot_message_types import (
    FORWARDED_RECORD_END,
    FORWARDED_RECORD_START,
    QUOTED_MESSAGE_END,
    QUOTED_MESSAGE_START,
    ForwardFetcher,
    OutgoingMessagePart,
    ReplyFetcher,
)

__all__ = [
    "FORWARDED_RECORD_END",
    "FORWARDED_RECORD_START",
    "QUOTED_MESSAGE_END",
    "QUOTED_MESSAGE_START",
    "ForwardFetcher",
    "OutgoingMessagePart",
    "ReplyFetcher",
    "coerce_message_segments",
    "format_mention_label",
    "image_attachments_from_message",
    "image_attachments_from_payload",
    "parse_outgoing_mention_parts",
    "render_message_text_and_mentions",
    "render_message_text_and_mentions_with_forwards",
    "reply_segment_ids",
    "segment_data",
    "segment_type",
    "strip_forwarded_records",
    "strip_quoted_messages",
]
