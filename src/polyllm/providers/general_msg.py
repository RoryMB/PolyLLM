import re
import json

def _extract_last_json(text):
    # Find all potential JSON objects in the text
    pattern = r'{[^{}]*(?:{[^{}]*}[^{}]*)*}'
    matches = re.finditer(pattern, text)
    matches = list(matches)

    if not matches:
        return None

    # Get the last match
    last_json_str = matches[-1].group()

    # Parse the string as JSON to verify it's valid
    try:
        json.loads(last_json_str)
    except json.JSONDecodeError:
        last_json_str = '{}'

    return last_json_str

def _load_image_path(image_path: str) -> bytes:
    with open(image_path, "rb") as image_file:
        return image_file.read()

def _load_image_cv2(image) -> bytes:
    import cv2
    success, buffer = cv2.imencode('.jpg', image)
    if not success:
        raise ValueError("Failed to encode image")
    return buffer.tobytes()

def _load_image_pil(image) -> bytes:
    from io import BytesIO
    buffer = BytesIO()
    image.save(buffer, format='JPEG')
    return buffer.getvalue()

def _load_image(image) -> bytes:
    if isinstance(image, str):
        return _load_image_path(image)

    try:
        import numpy as np
    except ModuleNotFoundError:
        pass
    else:
        if isinstance(image, np.ndarray):
            return _load_image_cv2(image)

    try:
        from PIL import Image
    except ModuleNotFoundError:
        pass
    else:
        if isinstance(image, Image.Image):
            return _load_image_pil(image)

    ... # TODO: Exception
