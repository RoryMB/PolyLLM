import json
import re
from typing import Callable

from pydantic import BaseModel


def extract_last_json(text):
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

def flatten_schema_dict(schema: dict[str, any]) -> dict[str, any]:
    """Transform a Pydantic JSON schema into Anthropic's tool schema format."""
    def _process_property(prop: dict[str, any]) -> dict[str, any]:
        result = {}

        # Copy basic fields
        if "type" in prop:
            result["type"] = prop["type"]
        if "description" in prop:
            result["description"] = prop["description"]

        # Handle array types
        if prop.get("type") == "array" and "items" in prop:
            items = prop["items"]
            if "$ref" in items:
                ref_name = items["$ref"].split("/")[-1]
                if ref_name in schema.get("$defs", {}):
                    result["items"] = _process_property(schema["$defs"][ref_name])
            else:
                result["items"] = _process_property(items)

        # Handle object references
        if "$ref" in prop:
            ref_name = prop["$ref"].split("/")[-1]
            if ref_name in schema.get("$defs", {}):
                return _process_property(schema["$defs"][ref_name])

        # Handle nested objects
        if prop.get("type") == "object":
            result["type"] = "object"
            if "properties" in prop:
                result["properties"] = {
                    k: _process_property(v)
                    for k, v in prop["properties"].items()
                }
            if "required" in prop:
                result["required"] = prop["required"]

        return result

    result = {
        "type": "object",
        "properties": {
            k: _process_property(v)
            for k, v in schema.get("properties", {}).items()
        }
    }

    if "required" in schema:
        result["required"] = schema["required"]

    return result

def structured_output_model_to_schema(structured_output_model: BaseModel, indent: int|str|None = None) -> str:
    return json.dumps(structured_output_model.model_json_schema(), indent=indent)

def structured_output_to_object(structured_output: str, structured_output_model: type[BaseModel]) -> BaseModel:
    try:
        data = json.loads(structured_output)
        response_object = structured_output_model(**data)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON string: {e}")
    except ValueError as e:
        raise RuntimeError(f"Error creating Pydantic model: {e}")

    return response_object

def get_tool_func(tools: list[Callable], tool: str) -> Callable:
    for func in tools:
        if func.__name__ == tool:
            return func

    return None

def load_image_path(image_path: str) -> bytes:
    with open(image_path, "rb") as image_file:
        return image_file.read()

def load_image_cv2(image) -> bytes:
    import cv2
    success, buffer = cv2.imencode('.jpg', image)
    if not success:
        raise RuntimeError("Failed to encode image")
    return buffer.tobytes()

def load_image_pil(image) -> bytes:
    from io import BytesIO
    buffer = BytesIO()
    image.save(buffer, format='JPEG')
    return buffer.getvalue()

def load_image(image) -> bytes:
    if isinstance(image, str):
        return load_image_path(image)

    try:
        import numpy as np
    except ModuleNotFoundError:
        pass
    else:
        if isinstance(image, np.ndarray):
            return load_image_cv2(image)

    try:
        from PIL import Image
    except ModuleNotFoundError:
        pass
    else:
        if isinstance(image, Image.Image):
            return load_image_pil(image)

    ... # TODO: Exception
