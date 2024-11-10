import pytest
import cv2
import numpy as np
from PIL import Image
from polyllm import polyllm

def test_multimodal_path(model, test_image):
    """Test multimodal capabilities with file path input"""
    if not test_image:
        pytest.skip("No test image configured")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image", "image": test_image},
            ],
        },
    ]

    response = polyllm.generate(model, messages)
    assert isinstance(response, str)
    assert len(response) > 0

def test_multimodal_cv2(model, test_image):
    """Test multimodal capabilities with cv2/numpy array input"""
    if not test_image:
        pytest.skip("No test image configured")

    img_array = cv2.imread(test_image)
    assert img_array is not None, "Failed to load test image with cv2"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image", "image": img_array},
            ],
        },
    ]

    response = polyllm.generate(model, messages)
    assert isinstance(response, str)
    assert len(response) > 0

def test_multimodal_pil(model, test_image):
    """Test multimodal capabilities with PIL Image input"""
    if not test_image:
        pytest.skip("No test image configured")

    pil_image = Image.open(test_image)
    assert pil_image is not None, "Failed to load test image with PIL"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image", "image": pil_image},
            ],
        },
    ]

    response = polyllm.generate(model, messages)
    assert isinstance(response, str)
    assert len(response) > 0
