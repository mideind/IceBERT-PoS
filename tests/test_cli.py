"""
Test cases for the CLI interface.
"""

import json
import subprocess
import sys
from unittest.mock import Mock, patch

import pytest
from icebert_pos import TaggedToken
from icebert_pos.cli import (
    CLIError,
    create_parser,
    format_output_as_json,
    format_output_as_text,
    main,
    process_cli_request,
    run_pos_tagging,
)
from rich.text import Text


@pytest.fixture
def mock_model_and_tokenizer():
    """Create mock model and tokenizer for testing."""
    mock_model = Mock()
    mock_tokenizer = Mock()
    return mock_model, mock_tokenizer


@pytest.fixture
def sample_results():
    """Sample POS tagging results for testing."""
    return [
        [
            TaggedToken(text="Halló", char_start=0, char_end=5, category="n", features=["sing"], ifd_tag="nken"),
            TaggedToken(text="heimur", char_start=6, char_end=12, category="n", features=["sing"], ifd_tag="nken"),
        ]
    ]


def test_create_parser():
    """Test argument parser creation."""
    parser = create_parser()

    # Test basic parsing
    args = parser.parse_args(["test text"])
    assert args.text == "test text"
    assert args.model_name == "mideind/IceBERT-PoS"
    assert args.batch_size == 1
    assert args.only_category is False
    assert args.json is False
    assert args.debug is False
    assert args.truncate is False
    assert args.no_split_composite_tokens is False


def test_parser_all_flags():
    """Test parser with all flags."""
    parser = create_parser()
    args = parser.parse_args(
        [
            "test text",
            "--model-name",
            "custom/model",
            # "--only-category", # This flag is mutually exclusive with --json
            "--json",
            "--batch-size",
            "5",
            "--truncate",
            "--no-split-composite-tokens",
            "--debug",
        ]
    )

    assert args.text == "test text"
    assert args.model_name == "custom/model"
    assert args.batch_size == 5
    assert args.json is True
    assert args.debug is True
    assert args.truncate is True
    assert args.no_split_composite_tokens is True


def test_format_output_as_json(sample_results):
    """Test JSON output formatting."""
    json_output = format_output_as_json(sample_results)

    # Should be valid JSON
    parsed = json.loads(json_output)
    assert isinstance(parsed, list)
    assert len(parsed) == 1
    assert len(parsed[0]) == 2

    # Check first token
    token = parsed[0][0]
    assert token["text"] == "Halló"
    assert token["char_start"] == 0
    assert token["char_end"] == 5
    assert token["category"] == "n"
    assert token["features"] == ["sing"]
    assert token["ifd_tag"] == "nken"


def test_format_output_as_text(sample_results):
    """Test text output formatting."""
    text = "Halló heimur"

    # Test with IFD tags
    output = format_output_as_text(text, sample_results, only_category=False)
    assert isinstance(output, Text)

    # Test with categories only
    output_cat = format_output_as_text(text, sample_results, only_category=True)
    assert isinstance(output_cat, Text)


@patch("icebert_pos.cli.pos_tag_text")
def test_run_pos_tagging(mock_pos_tag, mock_model_and_tokenizer, sample_results):
    """Test POS tagging runner."""
    model, tokenizer = mock_model_and_tokenizer
    mock_pos_tag.return_value = sample_results

    result = run_pos_tagging("test", model, tokenizer)
    assert result == sample_results
    mock_pos_tag.assert_called_once_with("test", model, tokenizer, 1, True, False)


@patch("icebert_pos.cli.pos_tag_text")
def test_run_pos_tagging_with_params(mock_pos_tag, mock_model_and_tokenizer, sample_results):
    """Test POS tagging runner with custom parameters."""
    model, tokenizer = mock_model_and_tokenizer
    mock_pos_tag.return_value = sample_results

    result = run_pos_tagging("test", model, tokenizer, batch_size=5, split_composite_tokens=False, truncate=True)
    assert result == sample_results
    mock_pos_tag.assert_called_once_with("test", model, tokenizer, 5, False, True)


@patch("icebert_pos.cli.pos_tag_text")
def test_run_pos_tagging_error(mock_pos_tag, mock_model_and_tokenizer):
    """Test POS tagging error handling."""
    model, tokenizer = mock_model_and_tokenizer
    mock_pos_tag.side_effect = Exception("Model error")

    with pytest.raises(CLIError, match="POS tagging failed"):
        run_pos_tagging("test", model, tokenizer)


@patch("icebert_pos.cli.load_model_and_tokenizer")
@patch("icebert_pos.cli.run_pos_tagging")
def test_process_cli_request_json(mock_run_pos, mock_load, mock_model_and_tokenizer, sample_results):
    """Test CLI request processing with JSON output."""
    model, tokenizer = mock_model_and_tokenizer
    mock_load.return_value = (model, tokenizer)
    mock_run_pos.return_value = sample_results

    parser = create_parser()
    args = parser.parse_args(["test text", "--json"])

    result = process_cli_request(args)

    assert result["output_type"] == "json"
    assert isinstance(result["output"], str)
    assert result["results"] == sample_results

    # Verify JSON is valid
    json.loads(result["output"])


@patch("icebert_pos.cli.load_model_and_tokenizer")
@patch("icebert_pos.cli.run_pos_tagging")
def test_process_cli_request_text(mock_run_pos, mock_load, mock_model_and_tokenizer, sample_results):
    """Test CLI request processing with text output."""
    model, tokenizer = mock_model_and_tokenizer
    mock_load.return_value = (model, tokenizer)
    mock_run_pos.return_value = sample_results

    parser = create_parser()
    args = parser.parse_args(["test text"])

    result = process_cli_request(args)

    assert result["output_type"] == "rich_text"
    assert isinstance(result["output"], Text)
    assert result["results"] == sample_results


def test_process_cli_request_with_preloaded_model(mock_model_and_tokenizer, sample_results):
    """Test CLI request processing with pre-loaded model."""
    model, tokenizer = mock_model_and_tokenizer

    with patch("icebert_pos.cli.run_pos_tagging") as mock_run_pos:
        mock_run_pos.return_value = sample_results

        parser = create_parser()
        args = parser.parse_args(["test text"])

        result = process_cli_request(args, model=model, tokenizer=tokenizer)

        assert result["results"] == sample_results
        # Should not try to load model since it was provided
        mock_run_pos.assert_called_once()


@patch("icebert_pos.cli.process_cli_request")
def test_main_success(mock_process):
    """Test successful main execution."""
    mock_process.return_value = {"output": "test output", "output_type": "rich_text", "results": [], "args": Mock()}

    exit_code = main(["test text"])
    assert exit_code == 0


@patch("icebert_pos.cli.process_cli_request")
def test_main_cli_error(mock_process):
    """Test main with CLI error."""
    mock_process.side_effect = CLIError("Test error")

    exit_code = main(["test text"])
    assert exit_code == 1


@patch("icebert_pos.cli.process_cli_request")
def test_main_keyboard_interrupt(mock_process):
    """Test main with keyboard interrupt."""
    mock_process.side_effect = KeyboardInterrupt()

    exit_code = main(["test text"])
    assert exit_code == 1


@patch("icebert_pos.cli.process_cli_request")
def test_main_unexpected_error(mock_process):
    """Test main with unexpected error."""
    mock_process.side_effect = RuntimeError("Unexpected error")

    exit_code = main(["test text"])
    assert exit_code == 1


def test_actual_cli_command():
    """Test actual CLI command execution."""
    # Test basic command
    result = subprocess.run([sys.executable, "-m", "icebert_pos.cli", "Halló heimur"], capture_output=True, text=True)

    assert result.returncode == 0
    assert "Halló" in result.stdout


def test_cli_json_output():
    """Test CLI JSON output."""
    result = subprocess.run(
        [sys.executable, "-m", "icebert_pos.cli", "--json", "Halló"], capture_output=True, text=True
    )

    assert result.returncode == 0
    # Should be valid JSON
    data = json.loads(result.stdout)
    assert isinstance(data, list)


def test_cli_invalid_args():
    """Test CLI with invalid arguments."""
    # Test missing required argument
    result = subprocess.run([sys.executable, "-m", "icebert_pos.cli"], capture_output=True, text=True)

    assert result.returncode != 0
    assert "the following arguments are required: text" in result.stderr


def test_mutually_exclusive_flags():
    """Test that --json and --only-category are mutually exclusive."""
    parser = create_parser()

    # Should fail when both flags are used together
    with pytest.raises(SystemExit):
        parser.parse_args(["--json", "--only-category", "test text"])

    # Should work with just --json
    args = parser.parse_args(["--json", "test text"])
    assert args.json is True
    assert args.only_category is False

    # Should work with just --only-category
    args = parser.parse_args(["--only-category", "test text"])
    assert args.json is False
    assert args.only_category is True

    # Should work with neither flag
    args = parser.parse_args(["test text"])
    assert args.json is False
    assert args.only_category is False
