# Copyright (C) Miðeind ehf.
# IceBERT POS tagging interface package

"""
IceBERT POS Tagging Interface

This package provides a high-level interface for POS tagging Icelandic text
using the IceBERT model with classical tokenization.
"""

from .interface import (
    Sentence,
    TaggedToken,
    Token,
    batch_sentences,
    pos_tag_text,
    predict_sentences,
    prepare_sentence,
    segment_text_to_sentences,
)

__all__ = [
    "TaggedToken",
    "Token",
    "Sentence",
    "batch_sentences",
    "pos_tag_text",
    "predict_sentences",
    "prepare_sentence",
    "segment_text_to_sentences",
]
