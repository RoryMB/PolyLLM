import base64
import json
from typing import Callable

from pydantic import BaseModel

from ..utils import flatten_schema_dict, extract_last_json, load_image

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

# https://docs.anthropic.com/en/docs/build-with-claude/tool-use#json-mode

def generate(
    model: str,
    messages: list,
    temperature: float,
    json_output: bool,
    structured_output_model: BaseModel|None,
    stream: bool = False,
):
    system_message = prepare_system_message(messages)
    transformed_messages = prepare_messages(messages)

    kwargs = {
        "model": model,
        "messages": transformed_messages,
        "max_tokens": 8192 if '3-5' in model else 4096,
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
        stream = False
        # TODO: Warn
        raw_schema = structured_output_model.model_json_schema()
        schema = flatten_schema_dict(raw_schema)
        kwargs["tools"] = [{
            "name": "format_response",
            "description": "Format the response using a specific JSON schema",
            "input_schema": schema
        }]
        kwargs["tool_choice"] = {"type": "tool", "name": "format_response"}

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
            for d in response.content:
                if d.type == 'tool_use':
                    text = d.input
            # text = response.content[1].input
            text = json.dumps(text)
        else:
            text = response.content[0].text
            if json_output:
                text = '{' + text[:text.rfind("}") + 1]
                text = extract_last_json(text)

        return text

def generate_tools(
    model: str,
    messages: list,
    temperature: float,
    tools: list[Callable],
):
    system_message = prepare_system_message(messages)
    transformed_messages = prepare_messages(messages)
    transformed_tools = prepare_tools(tools) if tools else None

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

def prepare_messages(messages):
    messages_out = []

    for message in messages:
        assert 'role' in message # TODO: Explanation
        assert 'content' in message # TODO: Explanation

        if message['role'] == 'system':
            continue

        role = message['role']

        if isinstance(message['content'], str):
            content = [{'type': 'text', 'text': message['content']}]
        elif isinstance(message['content'], list):
            content = []
            for item in message['content']:
                assert 'type' in item # TODO: Explanation

                if item['type'] == 'text':
                    content.append({'type': 'text', 'text': item['text']})
                elif item['type'] == 'image':
                    image_data = load_image(item['image'])
                    base64_image = base64.b64encode(image_data).decode('utf-8')
                    content.append({
                        'type': 'image',
                        'source': {
                            'type': 'base64',
                            'media_type': 'image/jpeg',
                            'data': base64_image,
                        },
                    })
                else:
                    ... # TODO: Exception
        else:
            ... # TODO: Exception

        messages_out.append({'role': role, 'content': content})

    return messages_out

def prepare_system_message(messages):
    system_message = None

    for message in messages:
        if message['role'] == 'system':
            system_message = message['content']
            break

    return system_message

def prepare_tools(tools: list[Callable]):
    tools_out = []

    for tool in tools:
        tools_out.append({
            "name": tool.__name__,
            "description": tool.__doc__,
            "input_schema": {
                "type": "object",
                "properties": {
                    param: {"type": "number" if annotation is int else "string"}
                    for param, annotation in tool.__annotations__.items()
                    if param != 'return'
                },
                "required": list(tool.__annotations__.keys())[:-1]
            }
        })

    return tools_out
