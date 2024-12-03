import json
import textwrap
from typing import Callable

from openai import OpenAI
from pydantic import BaseModel

from ..polyllm import structured_output_model_to_schema
from .llamapython_msg import _prepare_llamacpp_messages
from .openai_msg import _prepare_openai_tools

try:
    from llama_cpp.llama_grammar import json_schema_to_gbnf
    did_import = True
except ImportError:
    did_import = False

def get_models():
    return []

def _generate(
    model: str,
    messages: list,
    temperature: float,
    json_output: bool,
    structured_output_model: BaseModel|None,
    stream: bool = False,
):
    transformed_messages = _prepare_llamacpp_messages(messages)

    kwargs = {
        "model": model,
        "messages": transformed_messages,
        "stream": stream,
        "temperature": temperature,
    }

    if json_output:
        kwargs["response_format"] = {"type": "json_object"}
    if structured_output_model:
        schema = structured_output_model_to_schema(structured_output_model)
        gbnf = json_schema_to_gbnf(schema)
        kwargs["extra_body"] = {"grammar": gbnf}

    if ':' in model:
        base_url = f'http://{model}/v1'
    else:
        base_url = f'http://localhost:{model}/v1'

    client = OpenAI(
        base_url=base_url,
        api_key='-',
    )
    response = client.chat.completions.create(**kwargs)

    if stream:
        def stream_generator():
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        return stream_generator()
    else:
        text = response.choices[0].message.content
        return text

def _generate_tools(
    model: str,
    messages: list,
    temperature: float,
    tools: list[Callable],
):
    transformed_messages = _prepare_llamacpp_messages(messages)
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
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }

    if ':' in model:
        base_url = f'http://{model}/v1'
    else:
        base_url = f'http://localhost:{model}/v1'

    client = OpenAI(
        base_url=base_url,
        api_key='-',
    )
    response = client.chat.completions.create(**kwargs)

    j = json.loads(response.choices[0].message.content)

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
