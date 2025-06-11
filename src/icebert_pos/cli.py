#!/usr/bin/env python3

import argparse
import json
import logging
import sys
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.text import Text
from transformers import AutoModel, AutoTokenizer

from .interface import TaggedToken, pos_tag_text

logger = logging.getLogger(__name__)


class CLIError(Exception):
    """Custom exception for CLI errors."""

    pass


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(description="POS tag Icelandic text using mideind/IceBERT-PoS")
    parser.add_argument("text", help="Text to POS tag")
    parser.add_argument(
        "--model-name", default="mideind/IceBERT-PoS", help="Model name or path (default: mideind/IceBERT-PoS)"
    )
    
    # Create mutually exclusive group for output format
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--only-category", action="store_true", help="Return only POS categories instead of full IFD tags"
    )
    output_group.add_argument("--json", action="store_true", help="Return json output instead of plain text")
    
    parser.add_argument(
        "--keep-composite-tokens",
        action="store_true",
        help="Keep composite tokens as single tokens instead of splitting on whitespace (default: split them)",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for processing (default: 1)")
    parser.add_argument(
        "--truncate", action="store_true", help="Truncate input sequences that exceed model's maximum length"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    return parser


def load_model_and_tokenizer(model_name: str):
    """Load model and tokenizer."""
    try:
        logger.info(f"Loading model: {model_name}")
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        raise CLIError(f"Failed to load model '{model_name}': {e}")


def format_output_as_text(text: str, results: List[List[TaggedToken]], only_category: bool) -> Text:
    """Create rich-formatted text with POS tags in green brackets."""
    formatted_text = Text()
    current_pos = 0

    # Flatten the nested list structure
    all_tokens = [tok for sentence in results for tok in sentence]

    for tok in all_tokens:
        # Add any text between current position and token start (whitespace, etc.)
        if current_pos < tok.char_start:
            gap_text = text[current_pos : tok.char_start]
            formatted_text.append(gap_text, style="white")

        # Add the token text in white
        formatted_text.append(tok.text, style="white")

        # Add the POS ifd_tag or category in green brackets
        if only_category:
            formatted_text.append(f"[{tok.category}]", style="green")
        else:
            formatted_text.append(f"[{tok.ifd_tag}]", style="green")

        # Update position
        current_pos = tok.char_end

    # Add any remaining text after the last token
    if current_pos < len(text):
        remaining_text = text[current_pos:]
        formatted_text.append(remaining_text, style="white")

    return formatted_text


def format_output_as_json(results: List[List[TaggedToken]]) -> str:
    """Format results as JSON string."""
    results_json = [[asdict(token) for token in sentence] for sentence in results]
    return json.dumps(results_json, ensure_ascii=False, indent=2)


def run_pos_tagging(
    text: str, model, tokenizer, batch_size: int = 1, split_composite_tokens: bool = True, truncate: bool = False
) -> List[List[TaggedToken]]:
    """Run POS tagging with given parameters."""
    try:
        return pos_tag_text(text, model, tokenizer, batch_size, split_composite_tokens, truncate)
    except Exception as e:
        raise CLIError(f"POS tagging failed: {e}")


def process_cli_request(args: argparse.Namespace, model=None, tokenizer=None) -> Dict[str, Any]:
    """
    Process CLI request and return results.

    Args:
        args: Parsed command line arguments
        model: Optional pre-loaded model (for testing)
        tokenizer: Optional pre-loaded tokenizer (for testing)

    Returns:
        Dictionary with results and metadata
    """
    # Load model and tokenizer if not provided (for testing)
    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer(args.model_name)

    # Run POS tagging
    results = run_pos_tagging(
        args.text, model, tokenizer, args.batch_size, not args.keep_composite_tokens, args.truncate
    )

    # Format output
    if args.json:
        output = format_output_as_json(results)
        output_type = "json"
    else:
        output = format_output_as_text(args.text, results, args.only_category)
        output_type = "rich_text"

    return {"output": output, "output_type": output_type, "results": results, "args": args}


def setup_logging(debug: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, stream=sys.stderr)


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main CLI entry point.

    Args:
        argv: Command line arguments (for testing)

    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Set up logging
    setup_logging(args.debug)

    console = Console()

    try:
        # Process the CLI request
        result = process_cli_request(args)

        # Print output
        if result["output_type"] == "json":
            console.print(result["output"])
        else:
            console.print(result["output"])

        return 0

    except CLIError as e:
        console.print(f"Error: {e}", style="red")
        return 1
    except KeyboardInterrupt:
        console.print("\nOperation cancelled by user", style="yellow")
        return 1
    except Exception as e:
        console.print(f"Unexpected error: {e}", style="red")
        if args.debug:
            import traceback

            console.print(traceback.format_exc(), style="red")
        return 1


if __name__ == "__main__":
    sys.exit(main())
