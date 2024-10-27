import pytest
import os
from typing import Generator
from polyllm import polyllm

# Configure test models
LLAMA_PYTHON_MODEL = os.getenv("TEST_LLAMA_PYTHON_MODEL", "")
LLAMA_PYTHON_SERVER_PORT = os.getenv("TEST_LLAMA_PYTHON_SERVER_PORT", "")
OLLAMA_MODEL = os.getenv("TEST_OLLAMA_MODEL", "")
OPENAI_MODEL = os.getenv("TEST_OPENAI_MODEL", "")
GOOGLE_MODEL = os.getenv("TEST_GOOGLE_MODEL", "")
ANTHROPIC_MODEL = os.getenv("TEST_ANTHROPIC_MODEL", "")

# Test image path
TEST_IMAGE = os.getenv("TEST_IMAGE", "")

@pytest.fixture
def models() -> Generator[list[str], None, None]:
    """Return list of configured test models"""
    models = []
    
    if LLAMA_PYTHON_MODEL:
        from llama_cpp import Llama
        llm = Llama(
            model_path=LLAMA_PYTHON_MODEL,
            n_ctx=1024,
            n_gpu_layers=-1,
            verbose=False,
        )
        models.append(llm)
        
    if LLAMA_PYTHON_SERVER_PORT:
        models.append(f"llamacpp/{LLAMA_PYTHON_SERVER_PORT}")
        
    if OLLAMA_MODEL:
        models.append(f"ollama/{OLLAMA_MODEL}")
        
    if OPENAI_MODEL:
        models.append(OPENAI_MODEL)
        
    if GOOGLE_MODEL:
        models.append(GOOGLE_MODEL)
        
    if ANTHROPIC_MODEL:
        models.append(ANTHROPIC_MODEL)
        
    yield models

@pytest.fixture
def test_image() -> str:
    """Return path to test image if configured"""
    return TEST_IMAGE
