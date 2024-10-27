from polyllm import polyllm

def test_text(model):
    """Test basic text generation across all models, and correct handling of system, user, and assistant message roles"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."},
        {"role": "assistant", "content": "Why did the scarecrow win an award?\nBecause he was outstanding in his field!"},
        {"role": "user", "content": "Great! Tell me another joke!"},
    ]

    response = polyllm.generate(model, messages)
    assert isinstance(response, str)
    assert len(response) > 0
