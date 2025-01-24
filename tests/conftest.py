import os
from typing import Generator

import pytest
from llama_cpp import Llama
import mlx.nn
from mlx_lm.tokenizer_utils import TokenizerWrapper

import polyllm

from huggingface_hub.utils import disable_progress_bars
disable_progress_bars()

# Test image path
TEST_IMAGE = os.getenv("TEST_IMAGE", "")

# Configure test models
LLAMA_PYTHON_MODEL = os.getenv("TEST_LLAMA_PYTHON_MODEL", "")
LLAMA_PYTHON_SERVER_PORT = os.getenv("TEST_LLAMA_PYTHON_SERVER_PORT", "")
MLX_MODEL = os.getenv("TEST_MLX_MODEL", "")
OLLAMA_MODEL = os.getenv("TEST_OLLAMA_MODEL", "")
LITELLM_MODEL = os.getenv("TEST_LITELLM_MODEL", "")
OPENAI_MODEL = os.getenv("TEST_OPENAI_MODEL", "")
GOOGLE_MODEL = os.getenv("TEST_GOOGLE_MODEL", "")
ANTHROPIC_MODEL = os.getenv("TEST_ANTHROPIC_MODEL", "")

def get_model_name(model):
    match model:
        case Llama() if polyllm.providers['llamacpppython'].did_import:
            return model.model_path.split('/')[-1]
        case (mlx.nn.Module(), TokenizerWrapper()) if polyllm.providers['mlx'].did_import:
            return 'mlx:' + model[0].model_type
        case str():
            return model
        case _:
            return str(model)

def pytest_generate_tests(metafunc):
    if "model" in metafunc.fixturenames:
        models = []

        if LLAMA_PYTHON_MODEL:
            models.append(polyllm.load_helpers.load_llama(LLAMA_PYTHON_MODEL))

        if LLAMA_PYTHON_SERVER_PORT:
            models.append(LLAMA_PYTHON_SERVER_PORT)

        if MLX_MODEL:
            models.append(polyllm.load_helpers.load_mlx(MLX_MODEL))

        if OLLAMA_MODEL:
            models.append(OLLAMA_MODEL)

        if LITELLM_MODEL:
            models.append(LITELLM_MODEL)

        if OPENAI_MODEL:
            models.append(OPENAI_MODEL)

        if GOOGLE_MODEL:
            models.append(GOOGLE_MODEL)

        if ANTHROPIC_MODEL:
            models.append(ANTHROPIC_MODEL)

        assert models, "You must set at least 1 model to test."

        metafunc.parametrize("model", models, ids=[get_model_name(model) for model in models])

@pytest.fixture
def test_image() -> str:
    """Return path to test image if configured"""
    return TEST_IMAGE

def pytest_runtest_call(item):
    try:
        item.runtest()
    except NotImplementedError as e:
        pytest.skip(f"Not Implemented: {str(e)}")
