import pytest
import polyllm

def test_streaming(models):
    """Test streaming capabilities across all models"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Count from 1 to 5."},
    ]

    for model in models:
        chunks = []
        for chunk in polyllm.generate_stream(model, messages):
            assert isinstance(chunk, str)
            chunks.append(chunk)

        # Verify we got multiple chunks
        assert len(chunks) > 1

        # Combine and verify the complete response
        full_response = "".join(chunks)
        assert isinstance(full_response, str)
        assert len(full_response) > 0

        # Should contain numbers 1-5
        for i in range(1, 6):
            assert str(i) in full_response
