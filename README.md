# IceBERT PoS Tagging Interface

A high-level Python interface for PoS tagging Icelandic text using the [IceBERT-PoS](https://huggingface.co/mideind/IceBERT-PoS) model with classical tokenization.

## TODOs
- Proper device handling (GPU) for tensors

## Installation

```bash
# This package is currently not available on PyPI, so you need to install it directly from the source repository.

# Without PyTorch (lighter, but model inference won't work)
# Allows you to control the PyTorch version
pip install git+ssh://git@github.com/mideind/IceBERT-PoS.git@main  # Installs directly from main branch

# With PyTorch support (required for model inference) - RECOMMENDED
pip install "git+ssh://git@github.com/mideind/IceBERT-PoS.git[torch]"  # @main implied
```

> **Note**: The `[torch]` extra is required for model inference. Then why package it separately? To avoid pinning to specific versions of PyTorch and allow the user to install the latest version compatible with their system.

## Versioning

Package versioning is done via Git tags. To install a specific version, use the `@` syntax:

```bash
pip install "git+ssh://git@github.com/mideind/IceBERT-PoS.git@<version>"
```

### Version History
- `v0.3.0`: Loosen `transformers` version requirement from `>=4.46.3,<5.0` to `>=4.46.3,<6.0`. In other words, add support for `transformers` version `5`.
- `v0.2.0`: First stable release

## Features

- **Classical Tokenization**: Uses the [Miðeind tokenizer](https://github.com/mideind/tokenizer) for Icelandic tokenziation
- **Character Positions**: Preserves exact character start/end positions in original text
- **Sentence-Aware Processing**: Maintains sentence boundaries and processes them in batches
- **Dual Format Output**: Provides both IFD tags and structured category/features
- **Caller-owned Model**: Load model once, reuse for multiple calls
- **Batch Processing**: Efficient processing of multiple sentences

## Usage

### Command Line Interface

After installation, you can use the `icebert-pos` command:

```bash
# Basic POS tagging with full IFD tags
icebert-pos "Þetta er stutt sýnidæmi."
# Þetta[fahen] er[sfg3en] stutt[lhensf] sýnidæmi[nhen].[pl]

# Get only POS categories (without detailed features)
icebert-pos --only-category "Þetta er stutt sýnidæmi."
# Þetta[fa] er[sf] stutt[l] sýnidæmi[n].[pl]

# Get structured json output
icebert-pos --json "Þetta er stutt sýnidæmi."
# [
#   [
#     {
#       "text": "Þetta",
#       "char_start": 0,
#       "char_end": 5,
#       "category": "fa",
#       "features": [
#         "neut",
#         "sing",
#         "nom"
#       ],
#       "ifd_tag": "fahen"
#     },
#     ...
#     {
#       "text": ".",
#       "char_start": 23,
#       "char_end": 24,
#       "category": "pl",
#       "features": [],
#       "ifd_tag": "pl"
#     }
#   ]
# ]

# Default behavior is to split composite tokens (like "samskipta- og kynningarstýra") into individual tokens
icebert-pos "samskipta- og kynningarstýra"
# 3 tokens:
# samskipta-[kt] og[c] kynningarstýra[nven]
icebert-pos --keep-composite-tokens "samskipta- og kynningarstýra"
# 1 token:
# samskipta- og kynningarstýra[nven]

# Enable debug logging
icebert-pos --debug "Þetta er stutt sýnidæmi."
# lots of output
```
There are some additional command line options available, run `icebert-pos --help` to see them.

### Python API

#### Simple Usage

```python
from icebert_pos import pos_tag_text, TaggedToken
from transformers import AutoModel, AutoTokenizer
import torch

# Load model and tokenizer, you need to have trust_remote_code=True to load the custom model code.
# You can check the model repository for details: https://huggingface.co/mideind/IceBERT-PoS
model = AutoModel.from_pretrained("mideind/IceBERT-PoS", trust_remote_code=True)
# set the model to evaluation mode - otherwise the output will be stochastic
model.eval()
# place the model on the appropriate device (CPU/GPU)
model.to("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("mideind/IceBERT-PoS")

text = "Þetta er stutt sýnidæmi."
# POS tag text - returns List[List[TaggedToken]]
sentence_results = pos_tag_text(text, model, tokenizer)
assert sentence_results == [
    [
        TaggedToken(text="Þetta", char_start=0, char_end=5, category="fa", features=["neut", "sing", "nom"], ifd_tag="fahen"),
        TaggedToken(text="er", char_start=6, char_end=8, category="sf", features=["sing", "act", "3", "pres"], ifd_tag="sfg3en"),
        TaggedToken(text="stutt", char_start=9, char_end=14, category="l", features=["neut", "sing", "nom", "strong", "pos"], ifd_tag="lhensf"),
        TaggedToken(text="sýnidæmi", char_start=15, char_end=23, category="n", features=["neut", "sing", "nom"], ifd_tag="nhen"),
        TaggedToken(text=".", char_start=23, char_end=24, category="pl", features=[], ifd_tag="pl")
    ]
]
```

#### Batch Processing for Efficiency

```python
# For processing multiple sentences efficiently
# The Miðeind tokenizer will split this string into 3 sentences and process them in batches
texts = ["Fyrsti texti.", "Annar texti.", "Þriðji texti."]
# The batching is done automatically by the pos_tag_text function and this will call model.forward twice
sentence_results = pos_tag_text("\n".join(texts), model, tokenizer, batch_size=2)
assert len(sentence_results) == 3  # Should return 3 sentences
```

#### Advanced Usage with Lower-Level Functions

```python
from icebert_pos import (
    segment_text_to_sentences,
    prepare_sentence,
    batch_sentences,
    predict_sentences
)
# Same example as before
text = "Þetta er stutt sýnidæmi."
# Segment text into sentences
sentences = segment_text_to_sentences(text)

# Prepare individual sentences
sentence_tensors = []
for sentence in sentences:
    tensors = prepare_sentence(sentence, model, tokenizer, truncate=True)
    sentence_tensors.append(tensors)

# Batch multiple sentences for efficient processing
batch_input_ids, batch_attention_mask, batch_word_mask = batch_sentences(
    sentence_tensors, tokenizer
)

# Get raw predictions
predictions = predict_sentences(
    batch_input_ids, batch_attention_mask, batch_word_mask, model
)

# predictions is List[List[Tuple[str, List[str]]]]
# - List of sentences
# - Each sentence has List of (category, features) tuples for each word
assert predictions == [
    [
        ("fa", ["neut", "sing", "nom"]),
        ("sf", ["sing", "act", "3", "pres"]),
        ("l", ["neut", "sing", "nom", "strong", "pos"]),
        ("n", ["neut", "sing", "nom"]),
        ("pl", [])
    ]
]
```


## Data Structures

### Token
Basic token with text and position:
- `text`: The token text
- `char_start`: Start position in original text
- `char_end`: End position in original text

### Sentence
Collection of tokens representing a sentence:
- `tokens`: List of Token objects

### TaggedToken
Token with POS tagging information (extends Token):
- `text`: The token text
- `char_start`: Start position in original text
- `char_end`: End position in original text
- `category`: POS category (e.g., "fp", "sfg")
- `features`: List of morphological features (e.g., ["1", "sing", "nom"])
- `ifd_tag`: Full IFD POS tag (e.g., "fp1en", "sfg3en")

## API Reference

### High-Level Functions

- `pos_tag_text(text, model, tokenizer, batch_size=1, split_composite_tokens=True, truncate=False)` - Main function for POS tagging
- `segment_text_to_sentences(text, split_composite_tokens=True)` - Segment text into sentences using classical tokenization

#### Parameters

- `batch_size`: Number of sentences to process in each batch for efficiency (default: 1)
- `split_composite_tokens`: Whether to split composite tokens (like "samskipta- og kynningarstýra") into individual tokens on whitespace (default: True)
- `truncate`: Whether to truncate input sequences that exceed the model's maximum length. If False, long sentences may cause errors (default: False)

### Lower-Level Functions

- `prepare_sentence(sentence, model, tokenizer, truncate=False)` - Prepare tensors for a single sentence
- `batch_sentences(sentence_tensors, tokenizer)` - Batch multiple sentence tensors
- `predict_sentences(input_ids, attention_mask, word_mask, model)` - Get raw predictions from model

When using the lower-level functions you can control more of the processing but will also need to handle device placement and batching manually.

## License
MIT

Copyright (C) Miðeind ehf.
