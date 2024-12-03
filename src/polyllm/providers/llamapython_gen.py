import json
import textwrap
from typing import Callable

from pydantic import BaseModel

from ..polyllm import structured_output_model_to_schema
from .llamapython_msg import _prepare_llamacpp_messages
from .openai_msg import _prepare_openai_tools

try:
    from llama_cpp import Llama, LlamaGrammar
    did_import = True
except ImportError:
    did_import = False

def get_models():
    return []

def _generate(
    model: Llama,  # type: ignore
    messages: list,
    temperature: float,
    json_output: bool,
    structured_output_model: BaseModel|None,
    stream: bool = False,
):
    transformed_messages = _prepare_llamacpp_messages(messages)

    kwargs = {
        "messages": transformed_messages,
        "stream": stream,
        "temperature": temperature,
        "max_tokens": -1,
    }

    if json_output:
        kwargs["response_format"] = {"type": "json_object"}
    if structured_output_model:
        schema = structured_output_model_to_schema(structured_output_model)
        grammar = LlamaGrammar.from_json_schema(schema, verbose=False)
        kwargs["grammar"] = grammar

    response = model.create_chat_completion(**kwargs)

    if stream:
        def stream_generator():
            next(response)
            for chunk in response:
                if chunk['choices'][0]['finish_reason'] is not None:
                    break
                token = chunk['choices'][0]['delta']['content']
                # if not token:
                #     break
                yield token
        return stream_generator()
    else:
        text = response['choices'][0]['message']['content']
        return text

def _generate_tools(
    model: Llama,  # type: ignore
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
        "messages": transformed_messages,
        "stream": False,
        "temperature": temperature,
        "max_tokens": -1,
        "response_format": {
            "type": "json_object",
            "schema": {
                "type": "object",
                "properties": {
                    "tool_name": {"type": "string"},
                    "arguments": {"type": "object"},
                    "result": {"type": "string"},
                },
                "required": ["tool_name"],
            },
        },
    }

    response = model.create_chat_completion(**kwargs)

    j = json.loads(response['choices'][0]['message']['content'])

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
