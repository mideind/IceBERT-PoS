"""
Test cases for IceBERT POS tagging interface.
"""

from dataclasses import asdict
from typing import List

from icebert_pos import TaggedToken, pos_tag_text


def _run_pos_tagging_test(text, expected, model, tokenizer):
    results = pos_tag_text(text, model, tokenizer)
    _compare_results(expected, results)


def _compare_results(expected: List[List[dict]], results: List[List[TaggedToken]]):
    try:
        assert expected == [[asdict(token) for token in sentence] for sentence in results]
    except AssertionError as e:
        # If the structure doesn't match, we need to provide more detailed output
        max_sentences = max(len(expected), len(results))
        for i in range(max_sentences):
            if i >= len(expected):
                raise AssertionError(f"Result contained {results[i]} which we did not expect.") from e
            if i >= len(results):
                raise AssertionError(f"Expected {expected[i]} but got no result for sentence {i}.") from e
            exp = expected[i]
            res = results[i]
            max_tokens = max(len(exp), len(res))
            for j in range(max_tokens):
                if j >= len(exp):
                    raise AssertionError(f"Result contained {res[j]} which we did not expect in sentence {i}.") from e
                if j >= len(res):
                    raise AssertionError(f"Expected {exp[j]} but got no result for token {j} in sentence {i}.") from e
                exp_token = exp[j]
                res_token = asdict(res[j])
                if exp_token != res_token:
                    raise AssertionError(
                        f"Mismatch in sentence {i}, token {j}: expected {exp_token}, got {res_token}"
                    ) from e


def test_one_token(model, tokenizer):
    text = "Halló"
    expected = [[{"text": "Halló", "char_start": 0, "char_end": 5, "category": "au", "features": [], "ifd_tag": "au"}]]
    _run_pos_tagging_test(text, expected, model, tokenizer)


def test_simple_sentence(model, tokenizer):
    text = "Þetta er prófun á POS-tagging."
    expected = [
        [
            {
                "text": "Þetta",
                "char_start": 0,
                "char_end": 5,
                "category": "fa",
                "features": ["neut", "sing", "nom"],
                "ifd_tag": "fahen",
            },
            {
                "text": "er",
                "char_start": 6,
                "char_end": 8,
                "category": "sf",
                "features": ["sing", "act", "3", "pres"],
                "ifd_tag": "sfg3en",
            },
            {
                "text": "prófun",
                "char_start": 9,
                "char_end": 15,
                "category": "n",
                "features": ["fem", "sing", "nom"],
                "ifd_tag": "nven",
            },
            {"text": "á", "char_start": 16, "char_end": 17, "category": "af", "features": [], "ifd_tag": "af"},
            {
                "text": "POS-tagging",
                "char_start": 18,
                "char_end": 29,
                "category": "n",
                "features": ["gender_x", "sing", "dat", "proper"],
                "ifd_tag": "n-eþ-s",
            },
            {"text": ".", "char_start": 29, "char_end": 30, "category": "pl", "features": [], "ifd_tag": "pl"},
        ]
    ]
    _run_pos_tagging_test(text, expected, model, tokenizer)


def test_empty(model, tokenizer):
    text = ""
    expected = []
    _run_pos_tagging_test(text, expected, model, tokenizer)


def test_with_newlines(model, tokenizer):
    """Test complex text with newlines."""
    text = """Ein
setning
."""
    expected = [
        [
            {
                "text": "Ein",
                "char_start": 0,
                "char_end": 3,
                "category": "tf",
                "features": ["fem", "sing", "nom"],
                "ifd_tag": "tfven",
            },
            {
                "text": "setning",
                "char_start": 4,
                "char_end": 11,
                "category": "n",
                "features": ["fem", "sing", "nom"],
                "ifd_tag": "nven",
            },
            {"text": ".", "char_start": 12, "char_end": 13, "category": "pl", "features": [], "ifd_tag": "pl"},
        ]
    ]
    _run_pos_tagging_test(text, expected, model, tokenizer)


def test_with_dates(model, tokenizer):
    text = "Í dag er 29.04.2022"
    expected = [
        [
            {"text": "Í", "char_start": 0, "char_end": 1, "category": "af", "features": [], "ifd_tag": "af"},
            {
                "text": "dag",
                "char_start": 2,
                "char_end": 5,
                "category": "n",
                "features": ["masc", "sing", "acc"],
                "ifd_tag": "nkeo",
            },
            {
                "text": "er",
                "char_start": 6,
                "char_end": 8,
                "category": "sf",
                "features": ["sing", "act", "3", "pres"],
                "ifd_tag": "sfg3en",
            },
            {"text": "29.04.2022", "char_start": 9, "char_end": 19, "category": "ta", "features": [], "ifd_tag": "ta"},
        ]
    ]
    _run_pos_tagging_test(text, expected, model, tokenizer)


def test_batch_processing(model, tokenizer):
    """Test different batch sizes give same results."""
    text = "Þetta er fyrsta setningin. Þetta er setning númer tvö. Og þetta er þriðja."
    results_batch1 = pos_tag_text(text, model, tokenizer, batch_size=1)
    results_batch3 = pos_tag_text(text, model, tokenizer, batch_size=3)
    assert results_batch1 == results_batch3, "Batch processing results differ for batch sizes 1 and 3."


def test_special_characters(model, tokenizer):
    """Test special characters. This test demonstrates the current behavior"""
    text = "Sértákn: €, £, $, 5&! 50%."
    expected = [
        [
            {
                "text": "Sértákn",
                "char_start": 0,
                "char_end": 7,
                "category": "n",
                "features": ["neut", "plur", "nom"],
                "ifd_tag": "nhfn",
            },
            {"text": ":", "char_start": 7, "char_end": 8, "category": "pa", "features": [], "ifd_tag": "pa"},
            {"text": "€", "char_start": 9, "char_end": 10, "category": "pg", "features": [], "ifd_tag": "pg"},
            {"text": ",", "char_start": 10, "char_end": 11, "category": "pk", "features": [], "ifd_tag": "pk"},
            {"text": "£", "char_start": 12, "char_end": 13, "category": "pg", "features": [], "ifd_tag": "pg"},
            {"text": ",", "char_start": 13, "char_end": 14, "category": "pk", "features": [], "ifd_tag": "pk"},
            {"text": "$", "char_start": 15, "char_end": 16, "category": "pg", "features": [], "ifd_tag": "pg"},
            {"text": ",", "char_start": 16, "char_end": 17, "category": "pk", "features": [], "ifd_tag": "pk"},
            {"text": "5", "char_start": 18, "char_end": 19, "category": "ta", "features": [], "ifd_tag": "ta"},
            {"text": "&", "char_start": 19, "char_end": 20, "category": "pa", "features": [], "ifd_tag": "pa"},
            {"text": "!", "char_start": 20, "char_end": 21, "category": "pl", "features": [], "ifd_tag": "pl"},
        ],
        [
            {"text": "50%", "char_start": 22, "char_end": 25, "category": "tp", "features": [], "ifd_tag": "tp"},
            {"text": ".", "char_start": 25, "char_end": 26, "category": "pl", "features": [], "ifd_tag": "pl"},
        ],
    ]
    _run_pos_tagging_test(text, expected, model, tokenizer)


def test_compound_tokens(model, tokenizer):
    """Test tokens which the tokenizer considers compound."""
    # "samskipta- og kynningarstýra" is a compound token
    text = "í gær var samskipta- og kynningarstýra á undan mér upp úr."
    expected = [
        [
            {"text": "í", "char_start": 0, "char_end": 1, "category": "aa", "features": [], "ifd_tag": "aa"},
            {"text": "gær", "char_start": 2, "char_end": 5, "category": "aa", "features": [], "ifd_tag": "aa"},
            {
                "text": "var",
                "char_start": 6,
                "char_end": 9,
                "category": "sf",
                "features": ["sing", "act", "3", "past"],
                "ifd_tag": "sfg3eþ",
            },
            {"text": "samskipta-", "char_start": 10, "char_end": 20, "category": "kt", "features": [], "ifd_tag": "kt"},
            {"text": "og", "char_start": 21, "char_end": 23, "category": "c", "features": [], "ifd_tag": "c"},
            {
                "text": "kynningarstýra",
                "char_start": 24,
                "char_end": 38,
                "category": "n",
                "features": ["fem", "sing", "nom"],
                "ifd_tag": "nven",
            },
            {"text": "á", "char_start": 39, "char_end": 40, "category": "aa", "features": [], "ifd_tag": "aa"},
            {"text": "undan", "char_start": 41, "char_end": 46, "category": "af", "features": [], "ifd_tag": "af"},
            {
                "text": "mér",
                "char_start": 47,
                "char_end": 50,
                "category": "fp",
                "features": ["1", "sing", "dat"],
                "ifd_tag": "fp1eþ",
            },
            {"text": "upp", "char_start": 51, "char_end": 54, "category": "aa", "features": [], "ifd_tag": "aa"},
            {"text": "úr", "char_start": 55, "char_end": 57, "category": "aa", "features": [], "ifd_tag": "aa"},
            {"text": ".", "char_start": 57, "char_end": 58, "category": "pl", "features": [], "ifd_tag": "pl"},
        ]
    ]
    # Test with splitting composite tokens
    results = pos_tag_text(text, model, tokenizer, split_composite_tokens=True)
    _compare_results(expected, results)
    # Now without splitting compound tokens
    expected_no_split = [
        [
            {"text": "í", "char_start": 0, "char_end": 1, "category": "aa", "features": [], "ifd_tag": "aa"},
            {"text": "gær", "char_start": 2, "char_end": 5, "category": "aa", "features": [], "ifd_tag": "aa"},
            {
                "text": "var",
                "char_start": 6,
                "char_end": 9,
                "category": "sf",
                "features": ["sing", "act", "3", "past"],
                "ifd_tag": "sfg3eþ",
            },
            {
                "text": "samskipta- og kynningarstýra",
                "char_start": 10,
                "char_end": 38,
                "category": "n",
                "features": ["fem", "sing", "nom"],
                "ifd_tag": "nven",
            },
            {"text": "á", "char_start": 39, "char_end": 40, "category": "aa", "features": [], "ifd_tag": "aa"},
            {"text": "undan", "char_start": 41, "char_end": 46, "category": "af", "features": [], "ifd_tag": "af"},
            {
                "text": "mér",
                "char_start": 47,
                "char_end": 50,
                "category": "fp",
                "features": ["1", "sing", "dat"],
                "ifd_tag": "fp1eþ",
            },
            {"text": "upp", "char_start": 51, "char_end": 54, "category": "aa", "features": [], "ifd_tag": "aa"},
            {"text": "úr", "char_start": 55, "char_end": 57, "category": "aa", "features": [], "ifd_tag": "aa"},
            {"text": ".", "char_start": 57, "char_end": 58, "category": "pl", "features": [], "ifd_tag": "pl"},
        ]
    ]
    results_no_split = pos_tag_text(text, model, tokenizer, split_composite_tokens=False)
    _compare_results(expected_no_split, results_no_split)


def test_compound_tokens_currency(model, tokenizer):
    text = "Ég hef 1000 ISK í vasanum."
    expected = [
        [
            {
                "text": "Ég",
                "char_start": 0,
                "char_end": 2,
                "category": "fp",
                "features": ["1", "sing", "nom"],
                "ifd_tag": "fp1en",
            },
            {
                "text": "hef",
                "char_start": 3,
                "char_end": 6,
                "category": "sf",
                "features": ["sing", "act", "1", "pres"],
                "ifd_tag": "sfg1en",
            },
            {"text": "1000", "char_start": 7, "char_end": 11, "category": "ta", "features": [], "ifd_tag": "ta"},
            {"text": "ISK", "char_start": 12, "char_end": 15, "category": "ks", "features": [], "ifd_tag": "ks"},
            {"text": "í", "char_start": 16, "char_end": 17, "category": "af", "features": [], "ifd_tag": "af"},
            {
                "text": "vasanum",
                "char_start": 18,
                "char_end": 25,
                "category": "n",
                "features": ["masc", "sing", "dat", "definite"],
                "ifd_tag": "nkeþg",
            },
            {"text": ".", "char_start": 25, "char_end": 26, "category": "pl", "features": [], "ifd_tag": "pl"},
        ]
    ]
    # Test with splitting composite tokens
    results = pos_tag_text(text, model, tokenizer, split_composite_tokens=True)
    _compare_results(expected, results)
    # Now without splitting compound tokens
    expected_no_split = [
        [
            {
                "text": "Ég",
                "char_start": 0,
                "char_end": 2,
                "category": "fp",
                "features": ["1", "sing", "nom"],
                "ifd_tag": "fp1en",
            },
            {
                "text": "hef",
                "char_start": 3,
                "char_end": 6,
                "category": "sf",
                "features": ["sing", "act", "1", "pres"],
                "ifd_tag": "sfg1en",
            },
            {"text": "1000 ISK", "char_start": 7, "char_end": 15, "category": "ta", "features": [], "ifd_tag": "ta"},
            {"text": "í", "char_start": 16, "char_end": 17, "category": "af", "features": [], "ifd_tag": "af"},
            {
                "text": "vasanum",
                "char_start": 18,
                "char_end": 25,
                "category": "n",
                "features": ["masc", "sing", "dat", "definite"],
                "ifd_tag": "nkeþg",
            },
            {"text": ".", "char_start": 25, "char_end": 26, "category": "pl", "features": [], "ifd_tag": "pl"},
        ]
    ]
    results_no_split = pos_tag_text(text, model, tokenizer, split_composite_tokens=False)
    _compare_results(expected_no_split, results_no_split)
