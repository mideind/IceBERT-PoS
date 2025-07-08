# Copyright (C) Miðeind ehf.
# Simple POS tagging interface with classical tokenization

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import tokenizer

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


@dataclass
class Token:
    """A single token with its text and character positions."""

    text: str
    char_start: int  # inclusive start index
    char_end: int  # exclusive end index


@dataclass
class Sentence:
    """A sentence is a collection of tokens."""

    tokens: list[Token]


@dataclass
class TaggedToken(Token):
    """A Token with a PoS tag"""

    category: str
    features: list[str]
    ifd_tag: str


def segment_text_to_sentences(text: str, split_composite_tokens: bool = True) -> list[Sentence]:
    """
    Segment text into sentences using classical tokenization.

    Args:
        text: Input text to segment
        split_composite_tokens: Whether to split certain composite tokens into individual tokens on whitespace.

    Returns:
        List of Sentence objects with tokens and metadata
    """
    logger.debug("Performing classical tokenization")
    # We do not want to replace composite glyphs, as we do not want to alter the original text.
    # We ask for with_annotation=True (the default) which will combine some tokens:
    # - "samkskipta- og kynningarstýra"
    # - "1000 ISK"
    # If we set with_annotation=False, we would get ["samkskipta", "-", "og", "kynningarstýra"]
    # but having the "-" as a separate token in this context is generally not what we want.
    # We rather ask for these composites and if split_composite_tokens=True we split on spaces in the loop below.
    tokenizer_tokens = list(tokenizer.tokenize(text, replace_composite_glyphs=False, with_annotation=True))

    sentences: list[Sentence] = []
    current_tokens: list[Token] = []
    absolute_pos = 0  # Track absolute position using tok.original lengths

    for tokenizer_token in tokenizer_tokens:
        logger.debug(f"Processing token: {tokenizer_token}")
        if (
            tokenizer_token.kind < tokenizer.TOK.META_BEGIN
            and tokenizer_token.txt
            and tokenizer_token.original is not None
            and tokenizer_token.origin_spans
        ):
            # Calculate absolute character positions using tok.original and origin_spans
            # origin_spans are 0-based indices into tok.original
            char_start = absolute_pos + min(tokenizer_token.origin_spans)
            char_end = absolute_pos + max(tokenizer_token.origin_spans) + 1  # +1 because max gives last char index

            # But wait! Let's check if the token contains any whitespace and if we should split it
            if split_composite_tokens and " " in tokenizer_token.txt:
                logger.debug(f"Token '{tokenizer_token.txt}' contains whitespace, splitting into parts")
                # If it does, we split it into multiple tokens
                for part in tokenizer_token.txt.split():
                    part_char_start = char_start + tokenizer_token.txt.index(part)
                    part_char_end = part_char_start + len(part)
                    token = Token(text=part, char_start=part_char_start, char_end=part_char_end)
                    current_tokens.append(token)
            else:
                token = Token(text=tokenizer_token.txt, char_start=char_start, char_end=char_end)
                current_tokens.append(token)

        # Sentence/paragraph boundaries
        elif tokenizer_token.kind > tokenizer.TOK.META_BEGIN:
            if current_tokens:
                sentence = Sentence(tokens=current_tokens)
                sentences.append(sentence)

                current_tokens = []
            else:
                # Nothing to do, just skip
                pass

        else:
            # This should not happen, and we want a bug report if it does
            raise ValueError(
                f"Unexpected token kind: '{tokenizer_token.kind}' for token: '{tokenizer_token.txt}' in text: '{text}'. Please report this issue."
            )

        # Update absolute position using tok.original length (includes whitespace)
        if tokenizer_token.original is not None:
            absolute_pos += len(tokenizer_token.original)

    # Add remaining sentence if any
    if current_tokens:
        sentence = Sentence(tokens=current_tokens)
        sentences.append(sentence)

    if logger.getEffectiveLevel() == logging.DEBUG:
        logger.debug(sentences)

    return sentences


def prepare_sentence(
    sentence: Sentence, model, hf_tokenizer, truncate: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare a single sentence for model prediction.

    Args:
        sentence: Sentence to prepare
        model: Pre-loaded IceBERT model
        hf_tokenizer: Pre-loaded HuggingFace tokenizer

    Returns:
        Tuple of tensors ready for model input
    """
    sentence_words = [word.text for word in sentence.tokens]

    return model.prepare_inputs(
        sentence_words,
        hf_tokenizer,
        truncate=truncate,
    )


def batch_sentences(
    sentence_tensors: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], tokenizer
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batch a list of sentence tensors into a single batch.

    Args:
        sentence_tensors: List of tuples containing input tensors for each sentence

    Returns:
        Batched input tensors
    """
    try:
        from torch.nn.utils.rnn import pad_sequence
    except ModuleNotFoundError as e:
        raise ImportError(
            "The 'torch' library is required for this function. Please install it using "
            "'pip install icebert-pos[torch]'."
        ) from e

    # Unzip the list of tuples into separate lists
    input_ids, attention_mask, word_mask = zip(*sentence_tensors, strict=True)

    batch_input_ids = pad_sequence(list(input_ids), batch_first=True, padding_value=tokenizer.pad_token_id)
    batch_attention_mask = pad_sequence(list(attention_mask), batch_first=True, padding_value=0)
    batch_token_type_ids = pad_sequence(list(word_mask), batch_first=True, padding_value=0)
    return batch_input_ids, batch_attention_mask, batch_token_type_ids


def predict_sentences(
    input_ids,
    attention_mask,
    word_mask,
    model,
) -> list[list[tuple[str, list[str]]]]:
    """
    Predict POS tags for batched sentences.

    Args:
        input_ids: Batched input token IDs
        attention_mask: Batched attention mask
        word_mask: Batched word mask
        model: Pre-loaded IceBERT model

    Returns:
        list of lists of (category, features) tuples for each word in each sentence
    """
    # Get predictions from the model
    predictions = model.predict_labels(input_ids, attention_mask, word_mask)

    return predictions


def pos_tag_text(
    text: str, model, hf_tokenizer, batch_size: int = 1, split_composite_tokens: bool = True, truncate: bool = False
) -> list[list[TaggedToken]]:
    """
    POS tag text using classical tokenization and IceBERT model.

    Args:
        text: Input text to tag
        model: Pre-loaded IceBERT model
        hf_tokenizer: Pre-loaded HuggingFace tokenizer
        batch_size: Number of sentences to process in each batch
        split_composite_tokens: Whether to split certain composite tokens into individual words on whitespace. Defaults to True.
        truncate: Whether to truncate input sequences that exceed the model's maximum length. Defaults to False.

    Returns:
        list of lists of TaggedToken objects, one list per sentence
    """
    # Segment text into sentences
    sentences = segment_text_to_sentences(text, split_composite_tokens=split_composite_tokens)

    if not sentences:
        logger.warning("No sentences found in text")
        return []

    logger.debug(f"Processing {len(sentences)} sentences with batch size {batch_size}")

    all_results = []

    # Process sentences in batches
    for i in range(0, len(sentences), batch_size):
        current_batch_sentences = sentences[i : i + batch_size]
        logger.debug(f"Processing batch {i // batch_size + 1} with {len(current_batch_sentences)} sentences")

        # Prepare all sentences in the batch
        sentence_tensors = []
        for sentence in current_batch_sentences:
            tensors = prepare_sentence(sentence, model, hf_tokenizer, truncate=truncate)
            sentence_tensors.append(tensors)

        # Batch the tensors
        batch_input_ids, batch_attention_mask, batch_word_mask = batch_sentences(sentence_tensors, hf_tokenizer)

        # Get predictions for the batch
        batch_predictions = predict_sentences(batch_input_ids, batch_attention_mask, batch_word_mask, model)

        # Convert predictions to IFD format
        ifd_predictions = model.convert_labels_to_ifd(batch_predictions)

        # Convert predictions to TaggedToken objects
        for sentence, predictions, ifd_tags in zip(
            current_batch_sentences, batch_predictions, ifd_predictions, strict=True
        ):
            sentence_results = []

            # Ensure we have the same number of predictions as tokens
            if len(predictions) != len(sentence.tokens):
                logger.warning(f"Mismatch between tokens ({len(sentence.tokens)}) and predictions ({len(predictions)})")
                # Take the minimum to avoid index errors
                min_len = min(len(predictions), len(sentence.tokens), len(ifd_tags))
                predictions = predictions[:min_len]
                ifd_tags = ifd_tags[:min_len]
                tokens = sentence.tokens[:min_len]
            else:
                tokens = sentence.tokens

            # Create TaggedToken objects
            for token, (category, features), ifd_tag in zip(tokens, predictions, ifd_tags):
                tagged_token = TaggedToken(
                    text=token.text,
                    char_start=token.char_start,
                    char_end=token.char_end,
                    category=category,
                    features=features,
                    ifd_tag=ifd_tag,
                )
                sentence_results.append(tagged_token)

            all_results.append(sentence_results)

    logger.debug(f"Processed {len(all_results)} sentences with {sum(len(s) for s in all_results)} total tokens")
    return all_results
