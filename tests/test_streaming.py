from polyllm import polyllm

def test_streaming(model):
    """Test streaming capabilities across all models"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."},
        {"role": "assistant", "content": "Why did the scarecrow win an award?\nBecause he was outstanding in his field!"},
        {"role": "user", "content": "Great! Tell me another joke!"},
    ]

    chunks = []
    for chunk in polyllm.generate(model, messages, stream=True):
    # for chunk in polyllm.generate_stream(model, messages):
        assert isinstance(chunk, str)
        chunks.append(chunk)
    full_response = "".join(chunks)
    assert len(full_response) > 0
