"""Model/tokenizer adapters. Heavy GPU dependencies are imported lazily."""

from .hf import (
    HFChatAdapter,
    build_user_encoding,
    build_user_encoding_tokenwise,
    span_label,
)

__all__ = [
    "HFChatAdapter",
    "build_user_encoding",
    "build_user_encoding_tokenwise",
    "span_label",
]
