import json
import textwrap
from typing import Callable

from pydantic import BaseModel

from .ollama_msg import _prepare_ollama_messages
from .openai_msg import _prepare_openai_tools

try:
    import ollama
    try:
        ollama.list()
    except:  # noqa: E722
        did_import = False
    else:
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

    _models = sorted(model['name'] for model in ollama.list()['models'])

# https://github.com/ollama/ollama/blob/main/docs/api.md#parameters-1

def _generate(
    model: str,
    messages: list,
    temperature: float,
    json_output: bool,
    structured_output_model: BaseModel|None,
    stream: bool = False,
):
    transformed_messages = _prepare_ollama_messages(messages)

    kwargs = {
        "model": model,
        "messages": transformed_messages,
        "stream": stream,
        "options": {
            "temperature": temperature,
            # "num_ctx": 2048,
        }
    }

    if json_output:
        kwargs["format"] = "json"
    if structured_output_model:
        # TODO: Exception
        raise NotImplementedError("Ollama does not support Structured Output")

    response = ollama.chat(**kwargs)

    if stream:
        def stream_generator():
            for chunk in response:
                yield chunk['message']['content']
        return stream_generator()
    else:
        text = response['message']['content']
        return text

def _generate_tools(
    model: str,
    messages: list,
    temperature: float,
    tools: list[Callable],
):
    transformed_messages = _prepare_ollama_messages(messages)
    transformed_tools = _prepare_openai_tools(tools) if tools else None

    system_message = textwrap.dedent(f"""
        You are a helpful assistant.
        You have access to these tools:
            {transformed_tools}

        Always prefer a tool that can produce an answer if such a tool is available.

        Otherwise try to answer it on your own to the best of your ability, i.e. just provide a
        simple answer to the question, without elaborating.

        Always create JSON output.
        If the output requires a tool invocation, format the JSON in this way:
            {{
                "tool_name": "the_tool_name",
                "arguments": {{ "arg1_name": arg1, "arg2_name": arg2, ... }}
            }}
        If the output does NOT require a tool invocation, format the JSON in this way:
            {{
                "tool_name": "",  # empty string for tool name
                "result": response_to_the_query  # place the text response in a string here
            }}
    """).strip()

    transformed_messages.insert(0, {"role": "system", "content": system_message})

    kwargs = {
        "model": model,
        "messages": transformed_messages,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": temperature,
            # "num_ctx": 2048,
        }
    }

    response = ollama.chat(**kwargs)

    j = json.loads(response['message']['content'])

    text = ''
    tool = ''
    args = {}

    if 'tool_name' in j:
        if j['tool_name'] and 'arguments' in j:
            tool = j['tool_name']
            args = j['arguments']
        elif 'result' in j:
            text = j['result']
        else:
            text = 'Did not produce a valid response.'
    else:
        text = 'Did not produce a valid response.'

    return text, tool, args
