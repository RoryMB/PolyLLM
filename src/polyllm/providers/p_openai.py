import base64
import json
from typing import Callable

from pydantic import BaseModel

from ..utils import load_image

try:
    from openai import OpenAI
    openai_client = OpenAI()
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

    _models = sorted(model.id for model in list(openai_client.models.list()))

def generate(
    model: str,
    messages: list,
    temperature: float,
    json_output: bool,
    structured_output_model: BaseModel|None,
    stream: bool = False,
):
    transformed_messages = prepare_messages(messages)

    kwargs = {
        "model": model,
        "messages": transformed_messages,
        "stream": stream,
        "max_tokens": 4096,
        "temperature": temperature,
    }

    if json_output:
        kwargs["response_format"] = {"type": "json_object"}
    if structured_output_model:
        # Structured Output mode doesn't currently support streaming
        stream = False
        kwargs.pop("stream")
        # TODO: Warn
        kwargs["response_format"] = structured_output_model

    client = OpenAI()
    if stream:
        def stream_generator():
            response = client.chat.completions.create(**kwargs)
            for chunk in response:
                text = chunk.choices[0].delta.content
                if text:
                    yield text
        return stream_generator()
    else:
        if structured_output_model:
            response = client.beta.chat.completions.parse(**kwargs)
            if (response.choices[0].message.refusal):
                text = response.choices[0].message.refusal
            else:
                # Auto-generated Pydantic object here:
                #     response.choices[0].message.parsed
                text = response.choices[0].message.content
        else:
            response = client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content

        return text

def generate_tools(
    model: str,
    messages: list,
    temperature: float,
    tools: list[Callable],
):
    transformed_messages = prepare_messages(messages)
    transformed_tools = prepare_tools(tools) if tools else None

    kwargs = {
        "model": model,
        "messages": transformed_messages,
        "stream": False,
        "max_tokens": 4096,
        "temperature": temperature,
    }

    if transformed_tools:
        kwargs["tools"] = transformed_tools
        kwargs["tool_choice"] = "auto"

    client = OpenAI()
    response = client.chat.completions.create(**kwargs)

    text = ''
    if response.choices[0].message.content:
        text = response.choices[0].message.content

    tool = ''
    args = {}
    if response.choices[0].message.tool_calls:
        func = response.choices[0].message.tool_calls[0].function
        tool = func.name
        args = json.loads(func.arguments)

    return text, tool, args

def prepare_messages(messages):
    messages_out = []

    for message in messages:
        assert 'role' in message # TODO: Explanation
        assert 'content' in message # TODO: Explanation

        role = message['role']

        if isinstance(message['content'], str):
            content = message['content']
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
                        'type': 'image_url',
                        'image_url': {
                            'url': f"data:image/jpeg;base64,{base64_image}",
                        },
                    })
                else:
                    ... # TODO: Exception
        else:
            ... # TODO: Exception

        messages_out.append({'role': role, 'content': content})

    return messages_out

def prepare_tools(tools: list[Callable]):
    openai_tools = []

    for tool in tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.__name__,
                "description": tool.__doc__,
                "parameters": {
                    "type": "object",
                    "properties": {
                        param: {"type": "number" if annotation == int else "string"}  # noqa: E721
                        for param, annotation in tool.__annotations__.items()
                        if param != 'return'
                    },
                    "required": list(tool.__annotations__.keys())[:-1]
                }
            }
        })

    return openai_tools
