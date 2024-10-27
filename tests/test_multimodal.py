import pytest
import polyllm

def test_multimodal(models, test_image):
    """Test multimodal capabilities for supported models"""
    if not test_image:
        pytest.skip("No test image configured")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": test_image}},
            ],
        },
    ]

    for model in models:
        # Skip models that don't support multimodal
        if isinstance(model, str) and model.startswith(("llamacpp/", "ollama/")):
            continue

        response = polyllm.generate(model, messages)
        assert isinstance(response, str)
        assert len(response) > 0
