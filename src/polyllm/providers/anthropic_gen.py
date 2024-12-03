from typing import Callable

from pydantic import BaseModel

from ..polyllm import _extract_last_json
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
        # TODO: Exception
        raise NotImplementedError("Anthropic does not support Structured Output")

    if stream:
        def stream_generator():
            with anthropic_client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    yield text
        return stream_generator()
    else:
        response = anthropic_client.messages.create(**kwargs)

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
