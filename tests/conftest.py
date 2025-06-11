"""
Shared pytest fixtures for icebert_pos tests.
"""

from pathlib import Path

import pytest
from transformers import AutoModel, AutoTokenizer


@pytest.fixture(scope="session")
def model():
    """
    Load model once for all tests.
    
    Using session scope to avoid reloading the model for each test file.
    This makes tests much faster when running the full suite.
    """
    model = AutoModel.from_pretrained("mideind/IceBERT-PoS", trust_remote_code=True)
    model.eval()
    return model


@pytest.fixture(scope="session")
def tokenizer():
    """
    Load tokenizer once for all tests.
    
    Using session scope to avoid reloading the tokenizer for each test file.
    This makes tests much faster when running the full suite.
    """
    return AutoTokenizer.from_pretrained("mideind/IceBERT-PoS")


@pytest.fixture(scope="session")
def readme_content():
    """
    Load README.md content once for all tests.
    
    Using session scope to avoid re-reading the file for each test.
    """
    readme_path = Path(__file__).parent.parent / "README.md"
    return readme_path.read_text(encoding="utf-8")