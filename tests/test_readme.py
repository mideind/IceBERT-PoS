"""
Test that all Python code examples and CLI commands in README.md actually work.
"""

import re
import shlex

import pytest
from icebert_pos.cli import create_parser, process_cli_request


def extract_code_blocks(readme_content: str) -> list[str]:
    """Extract Python code blocks from markdown content."""
    pattern = r"```python\n(.*?)\n```"
    matches = re.findall(pattern, readme_content, re.DOTALL)
    return matches


def extract_bash_examples(readme_content: str) -> list[str]:
    """Extract bash command examples from README."""
    pattern = r"```bash\n(.*?)\n```"
    matches = re.findall(pattern, readme_content, re.DOTALL)

    commands = []
    for match in matches:
        lines = match.strip().split("\n")
        for line in lines:
            line = line.strip()
            # Skip empty lines, comments, and TODO lines
            if line and not line.startswith("#") and not line.startswith("TODO"):
                commands.append(line)

    return commands


def test_readme_code_blocks(readme_content):
    """Test that all Python code blocks in README.md execute without errors."""
    code_blocks = extract_code_blocks(readme_content)

    assert len(code_blocks) > 0, "No Python code blocks found in README.md"
    # We treat the code blocks as a single unit
    code = "\n".join(code_blocks)

    try:
        exec(code)
    except Exception as e:
        pytest.fail(f"Code block failed to execute: {e}\n\nCode:\n{code}")


def test_readme_cli_examples(readme_content, model, tokenizer):
    """Test that CLI examples from README work with actual model."""
    commands = extract_bash_examples(readme_content)
    icebert_commands = [cmd for cmd in commands if cmd.startswith("icebert-pos")]

    assert len(icebert_commands) > 0, "No icebert-pos commands found in README"
    parser = create_parser()
    for command in icebert_commands:
        # Remove icebert-pos prefix and split into args
        # Use shlex to properly handle quoted arguments
        argsv = shlex.split(command)[1:]  # Remove 'icebert-pos' and properly split remaining args
        try:
            args = parser.parse_args(argsv)  # Validate arguments
            # Call main function with parsed args
            process_cli_request(args, model=model, tokenizer=tokenizer)
        except Exception as e:
            pytest.fail(f"Command failed with exception: {e}\nCommand: {command}")
