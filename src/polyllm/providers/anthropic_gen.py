import json
from typing import Callable, Any

from pydantic import BaseModel

from ..polyllm import _extract_last_json

def _transform_schema_for_anthropic(schema: dict[str, Any]) -> dict[str, Any]:
    """Transform a Pydantic JSON schema into Anthropic's tool schema format."""
    def _process_property(prop: dict[str, Any]) -> dict[str, Any]:
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
from .anthropic_msg import (
    _prepare_anthropic_messages,
    _prepare_anthropic_system_message,
    _prepare_anthropic_tools,
)

try:
    import anthropic
    anthropic_client = anthropic.Anthropic()
    did_import = True
except ImportError:
    did_import = False

_models = []
def get_models():
    lazy_load()
    return _models

lazy_loaded = False
def lazy_load():
    global lazy_loaded, _models

    if lazy_loaded:
        return
    lazy_loaded = True

    if not did_import:
        return

    _models = sorted([
        "claude-1.0",
        "claude-1.1",
        "claude-1.2",
        "claude-1.3-100k",
        "claude-1.3",
        "claude-2.0",
        "claude-2.1",
        "claude-3-5-haiku-20241022",
        "claude-3-5-haiku-latest",
        "claude-3-5-sonnet-20240620",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-latest",
        "claude-3-haiku-20240307",
        "claude-3-opus-20240229",
        "claude-3-opus-latest",
        "claude-3-sonnet-20240229",
        "claude-instant-1.0",
        "claude-instant-1.1-100k",
        "claude-instant-1.1",
        "claude-instant-1.2",
    ])

def _generate(
    model: str,
    messages: list,
    temperature: float,
    json_output: bool,
    structured_output_model: BaseModel|None,
    stream: bool = False,
):
    system_message = _prepare_anthropic_system_message(messages)
    transformed_messages = _prepare_anthropic_messages(messages)

    kwargs = {
        "model": model,
        "messages": transformed_messages,
        "max_tokens": 4000,
        "temperature": temperature,
    }

    if system_message:
        kwargs["system"] = system_message

    if json_output:
        stream = False
        # TODO: Warn
        transformed_messages.append(
            {
                "role": "assistant",
                "content": "Here is the JSON requested:\n{"
            }
        )
    if structured_output_model:
        raw_schema = structured_output_model.model_json_schema()
        schema = _transform_schema_for_anthropic(raw_schema)
        kwargs["tools"] = [{
            "name": "format_response",
            "description": "Format the response using a specific JSON schema",
            "input_schema": schema
        }]
        kwargs["tool_choice"] = {"type": "tool", "name": "format_response"}
        stream = False  # Disable streaming for structured output

    if stream:
        def stream_generator():
            with anthropic_client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    yield text
        return stream_generator()
    else:
        response = anthropic_client.messages.create(**kwargs)

        if structured_output_model and response.stop_reason == "tool_use":
            # Extract structured output from tool response
            text = response.content[1].input
            return json.dumps(text)
        else:
            text = response.content[0].text
            if json_output:
                text = '{' + text[:text.rfind("}") + 1]
                text = _extract_last_json(text)
            return text

def _generate_tools(
    model: str,
    messages: list,
    temperature: float,
    tools: list[Callable],
):
    system_message = _prepare_anthropic_system_message(messages)
    transformed_messages = _prepare_anthropic_messages(messages)
    transformed_tools = _prepare_anthropic_tools(tools) if tools else None

    kwargs = {
        "model": model,
        "messages": transformed_messages,
        "max_tokens": 4000,
        "temperature": temperature,
    }

    if system_message:
        kwargs["system"] = system_message

    if transformed_tools:
        kwargs["tools"] = transformed_tools

    response = anthropic_client.messages.create(**kwargs)

    text = response.content[0].text

    tool = ''
    args = {}
    if response.stop_reason == "tool_use":
        func = response.content[1]
        tool = func.name
        args = func.input

    return text, tool, args
