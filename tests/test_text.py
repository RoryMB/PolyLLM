import pytest
import polyllm

def test_text_generation(models):
    """Test basic text generation across all models"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello!"},
    ]

    for model in models:
        response = polyllm.generate(model, messages)
        assert isinstance(response, str)
        assert len(response) > 0
        assert "hello" in response.lower()
