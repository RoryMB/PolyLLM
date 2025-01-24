import base64
import json
import os
from typing import Callable
import warnings

from pydantic import BaseModel

from ..utils import load_image

try:
    if 'GEMINI_API_KEY' not in os.environ and 'GOOGLE_API_KEY' in os.environ:
        os.environ['GEMINI_API_KEY'] = os.environ['GOOGLE_API_KEY']
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import litellm
    did_import = True
except ImportError:
    did_import = False


_models = []
def get_models():
    return _models

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
        "temperature": temperature,
        "stream": stream,
    }

    if json_output:
        if "response_format" not in litellm.get_supported_openai_params(model):
            raise NotImplementedError
        kwargs["response_format"] = {"type": "json_object"}
    if structured_output_model:
        if "response_format" not in litellm.get_supported_openai_params(model):
            raise NotImplementedError
        if not litellm.supports_response_schema(model):
            raise NotImplementedError
        kwargs["response_format"] = structured_output_model

    response = litellm.completion(**kwargs)

    if stream:
        return (part.choices[0].delta.content or "" for part in response)
    else:
        return response.choices[0].message.content

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
        "temperature": temperature,
    }

    if transformed_tools:
        litellm.supports_function_calling(model="ollama/llama2")
        kwargs["tools"] = transformed_tools
        kwargs["tool_choice"] = "auto"

    response = litellm.completion(**kwargs)

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
    tools_out = []

    for tool in tools:
        tools_out.append({
            "type": "function",
            "function": {
                "name": tool.__name__,
                "description": tool.__doc__,
                "parameters": {
                    "type": "object",
                    "properties": {
                        param: {"type": "number" if annotation is int else "string"}
                        for param, annotation in tool.__annotations__.items()
                        if param != 'return'
                    },
                    "required": list(tool.__annotations__.keys())[:-1]
                }
            }
        })

    return tools_out
