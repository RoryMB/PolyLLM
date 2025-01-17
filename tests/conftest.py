import pytest
import os
from typing import Generator
from llama_cpp import Llama

# Test image path
TEST_IMAGE = os.getenv("TEST_IMAGE", "")

# Configure test models
LLAMA_PYTHON_MODEL = os.getenv("TEST_LLAMA_PYTHON_MODEL", "")
LLAMA_PYTHON_SERVER_PORT = os.getenv("TEST_LLAMA_PYTHON_SERVER_PORT", "")
OLLAMA_MODEL = os.getenv("TEST_OLLAMA_MODEL", "")
OPENAI_MODEL = os.getenv("TEST_OPENAI_MODEL", "")
GOOGLE_MODEL = os.getenv("TEST_GOOGLE_MODEL", "")
ANTHROPIC_MODEL = os.getenv("TEST_ANTHROPIC_MODEL", "")

def get_model_name(model):
    if isinstance(model, str):
        return model
    if isinstance(model, Llama):
        return model.model_path.split('/')[-1]
    return str(model)

def pytest_generate_tests(metafunc):
    if "model" in metafunc.fixturenames:
        models = []

        if LLAMA_PYTHON_MODEL:
            llm = Llama(
                model_path=LLAMA_PYTHON_MODEL,
                n_ctx=1024,
                n_gpu_layers=-1,
                verbose=False,
            )
            models.append(llm)

        if LLAMA_PYTHON_SERVER_PORT:
            models.append(LLAMA_PYTHON_SERVER_PORT)

        if OLLAMA_MODEL:
            models.append(OLLAMA_MODEL)

        if OPENAI_MODEL:
            models.append(OPENAI_MODEL)

        if GOOGLE_MODEL:
            models.append(GOOGLE_MODEL)

        if ANTHROPIC_MODEL:
            models.append(ANTHROPIC_MODEL)

        metafunc.parametrize("model", models, ids=[get_model_name(model) for model in models])

@pytest.fixture
def model(request) -> Generator[str|Llama, None, None]:
    """Return a single configured test model"""
    yield request.param

@pytest.fixture
def test_image() -> str:
    """Return path to test image if configured"""
    return TEST_IMAGE

def pytest_runtest_call(item):
    try:
        item.runtest()
    except NotImplementedError as e:
        pytest.skip(f"Not Implemented: {str(e)}")
