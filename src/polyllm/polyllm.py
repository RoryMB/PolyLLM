import base64
import json
import re
import textwrap
import time
import warnings
from typing import Callable, Generator, Literal, overload

import backoff
from pydantic import BaseModel

try:
    from llama_cpp import Llama, LlamaGrammar
    from llama_cpp.llama_grammar import json_schema_to_gbnf
    llamapython_import = True
except ImportError:
    llamapython_import = False
    Llama = None

try:
    import ollama
    ollama_import = True
except ImportError:
    ollama_import = False

try:
    from openai import OpenAI
    openai_client = OpenAI()
    openai_import = True
except ImportError:
    openai_import = False

try:
    import google.generativeai as genai
    from google.api_core.exceptions import ResourceExhausted as GoogleResourceExhausted
    from google.generativeai.types import HarmBlockThreshold, HarmCategory
    genai.configure()
    google_import = True
except ImportError:
    google_import = False
    GoogleResourceExhausted = None

try:
    import anthropic
    anthropic_client = anthropic.Anthropic()
    anthropic_import = True
except ImportError:
    anthropic_import = False


_ollama_models = []
_openai_models = []
_google_models = []
_anthropic_models = []

lazy_loaded = False
def lazy_load():
    global lazy_loaded, _ollama_models, _openai_models, _google_models, _anthropic_models

    if lazy_loaded:
        return
    lazy_loaded = True

    if ollama_import:
        _ollama_models = sorted(model['name'] for model in ollama.list()['models'])

    if openai_import:
        _openai_models = sorted(model.id for model in list(openai_client.models.list()))

    if google_import:
        _google_models = sorted(model.name.split('/')[1] for model in genai.list_models() if 'generateContent' in model.supported_generation_methods)

    if anthropic_import:
        _anthropic_models = sorted([
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

def ollama_models():
    lazy_load()
    return _ollama_models

def openai_models():
    lazy_load()
    return _openai_models

def google_models():
    lazy_load()
    return _google_models

def anthropic_models():
    lazy_load()
    return _anthropic_models

MODEL_ERR_MSG = "PolyLLM could not find model: {model}. Run `python -m polyllm` to see a list of known models."
if not all((llamapython_import, openai_import, google_import, anthropic_import)):
    missing = []
    if not llamapython_import:
        missing.append('llama-cpp-python')
    if not openai_import:
        missing.append('openai')
    if not google_import:
        missing.append('google-generativeai')
    if not anthropic_import:
        missing.append('anthropic')
    MODEL_ERR_MSG += " Note: Imports failed for: pip install " + " ".join(missing) + " ."


@overload
def generate(
    model: str|Llama, # type: ignore
    messages: list,
    temperature: float = 0.0,
    json_output: bool = False,
    structured_output_model: BaseModel|None = None,
    stream: Literal[False] = False,
) -> str: ...

@overload
def generate(
    model: str|Llama, # type: ignore
    messages: list,
    temperature: float = 0.0,
    json_output: bool = False,
    structured_output_model: BaseModel|None = None,
    stream: Literal[True] = True,
) -> Generator[str, None, None]: ...

def generate(
    model: str|Llama, # type: ignore
    messages: list,
    temperature: float = 0.0,
    json_output: bool = False,
    structured_output_model: BaseModel|None = None,
    stream: bool = False,
) -> str | Generator[str, None, None]:
    if json_output and structured_output_model:
        raise ValueError("generate() cannot simultaneously support JSON mode (json_output) and Structured Output mode (structured_output_model)")

    func = None

    if llamapython_import and isinstance(model, Llama):
        func = _llamapython
    elif model.startswith('llamacpp/'):
        model = model.split('/', maxsplit=1)[1]
        func = _llamacpp
    elif model.startswith('ollama/'):
        model = model.split('/', maxsplit=1)[1]
        func = _ollama
    elif model.startswith('openai/'):
        model = model.split('/', maxsplit=1)[1]
        func = _openai
    elif model.startswith('google/'):
        model = model.split('/', maxsplit=1)[1]
        func = _google
    elif model.startswith('anthropic/'):
        model = model.split('/', maxsplit=1)[1]
        func = _anthropic
    else:
        if model in ollama_models():
            func = _openai
        elif model in openai_models():
            func = _openai
        elif model in google_models():
            func = _google
        elif model in anthropic_models():
            func = _anthropic

    if func:
        return func(model, messages, temperature, json_output, structured_output_model, stream)
    else:
        raise ValueError(MODEL_ERR_MSG.format(model=model))

def generate_stream(
    model: str|Llama, # type: ignore
    messages: list,
    temperature: float = 0.0,
    json_output: bool = False,
    structured_output_model: BaseModel|None = None,
) -> Generator[str, None, None]:
    return generate(model, messages, temperature, json_output, structured_output_model, stream=True)

def generate_tools(
    model: str|Llama, # type: ignore
    messages: list,
    temperature: float = 0.0,
    tools: list[Callable] = None,
) -> tuple[str, str, dict]:
    func = None

    if llamapython_import and isinstance(model, Llama):
        func = _llamapython_tools
    elif model.startswith('llamacpp/'):
        model = model.split('/', maxsplit=1)[1]
        func = _llamacpp_tools
    elif model.startswith('ollama/'):
        model = model.split('/', maxsplit=1)[1]
        func = _ollama_tools
    elif model.startswith('openai/'):
        model = model.split('/', maxsplit=1)[1]
        func = _openai_tools
    elif model.startswith('google/'):
        model = model.split('/', maxsplit=1)[1]
        func = _google_tools
    elif model.startswith('anthropic/'):
        model = model.split('/', maxsplit=1)[1]
        func = _anthropic_tools
    else:
        if model in ollama_models():
            func = _ollama_tools
        elif model in openai_models():
            func = _openai_tools
        elif model in google_models():
            func = _google_tools
        elif model in anthropic_models():
            func = _anthropic_tools

    if func:
        return func(model, messages, temperature, tools)
    else:
        raise ValueError(MODEL_ERR_MSG.format(model=model))

def structured_output_model_to_schema(structured_output_model: BaseModel, indent: int|str|None = None) -> str:
    return json.dumps(structured_output_model.model_json_schema(), indent=indent)

def structured_output_to_object(structured_output: str, structured_output_model: type[BaseModel]) -> BaseModel:
    try:
        data = json.loads(structured_output)
        response_object = structured_output_model(**data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON string: {e}")
    except ValueError as e:
        raise ValueError(f"Error creating Pydantic model: {e}")

    return response_object

def get_tool_func(tools: list[Callable], tool: str) -> Callable:
    for func in tools:
        if func.__name__ == tool:
            return func

    return None

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


def _llamapython(
    model: Llama, # type: ignore
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

def _llamapython_tools(
    model: Llama, # type: ignore
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


def _llamacpp(
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

def _llamacpp_tools(
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


def _ollama_old(
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
        "temperature": temperature,
    }

    if json_output:
        kwargs["response_format"] = {"type": "json_object"}
    if structured_output_model:
        # TODO: Exception
        raise NotImplementedError("Ollama does not support Structured Output")

    client = OpenAI(
        base_url='http://localhost:11434/v1',
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

def _ollama_tools_old(
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
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }

    client = OpenAI(
        base_url='http://localhost:11434/v1',
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

def _ollama(
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

def _ollama_tools(
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

# https://github.com/ollama/ollama/blob/main/docs/api.md#parameters-1


def _openai(
    model: str,
    messages: list,
    temperature: float,
    json_output: bool,
    structured_output_model: BaseModel|None,
    stream: bool = False,
):
    transformed_messages = _prepare_openai_messages(messages)

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

    if stream:
        def stream_generator():
            response = openai_client.chat.completions.create(**kwargs)
            for chunk in response:
                text = chunk.choices[0].delta.content
                if text:
                    yield text
        return stream_generator()
    else:
        if structured_output_model:
            response = openai_client.beta.chat.completions.parse(**kwargs)
            if (response.choices[0].message.refusal):
                text = response.choices[0].message.refusal
            else:
                # Auto-generated Pydantic object here:
                #     response.choices[0].message.parsed
                text = response.choices[0].message.content
        else:
            response = openai_client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content

        return text

def _openai_tools(
    model: str,
    messages: list,
    temperature: float,
    tools: list[Callable],
):
    transformed_messages = _prepare_openai_messages(messages)
    transformed_tools = _prepare_openai_tools(tools) if tools else None

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

    response = openai_client.chat.completions.create(**kwargs)

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


@backoff.on_exception(
    backoff.constant,
    GoogleResourceExhausted,
    interval=60,
    max_tries=4,
)
def _google(
    model: str,
    messages: list,
    temperature: float,
    json_output: bool,
    structured_output_model: BaseModel|None,
    stream: bool = False,
):
    system_message = _prepare_google_system_message(messages)
    transformed_messages = _prepare_google_messages(messages)

    generation_config = {
        "temperature": temperature,
        "max_output_tokens": 8192,
    }

    if json_output or structured_output_model:
        generation_config["response_mime_type"] = "application/json"
    else:
        generation_config["response_mime_type"] = "text/plain"

    if structured_output_model:
        generation_config["response_schema"] = structured_output_model

    gemini_model = genai.GenerativeModel(
        model_name=model,
        system_instruction=system_message,
        generation_config=generation_config,
    )

    response = gemini_model.generate_content(
        transformed_messages,
        stream=stream,
        safety_settings={
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )

    if stream:
        def stream_generator():
            for chunk in response:
                if chunk.parts:
                    for part in chunk.parts:
                        if part.text:
                            yield part.text
                elif chunk.text:
                    yield chunk.text
        return stream_generator()
    else:
        return response.text

def _google_tools(
    model: str,
    messages: list,
    temperature: float,
    tools: list[Callable],
):
    system_message = _prepare_google_system_message(messages)
    transformed_messages = _prepare_google_messages(messages)

    if tools:
        system_message = (system_message or '') + "\nIf you do not have access to a function that can help answer the question, answer it on your own to the best of your ability."

    generation_config = {
        "temperature": temperature,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain"
    }
    gemini_model = genai.GenerativeModel(
        model_name=model,
        system_instruction=system_message,
        generation_config=generation_config,
        tools=tools,
    )
    try:
        response = gemini_model.generate_content(
            transformed_messages,
            stream=False,
            tool_config={"function_calling_config": "AUTO"},
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
    except GoogleResourceExhausted:
        time.sleep(60)
        response = gemini_model.generate_content(
            transformed_messages,
            stream=False,
            tool_config={"function_calling_config": "AUTO"},
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )

    text = ''
    tool = ''
    args = {}

    for part in response.candidates[0].content.parts:
        if not text and part.text:
            text = part.text

        if not tool and part.function_call:
            func = part.function_call
            tool = func.name
            args = dict(func.args)

    return text, tool, args


def _anthropic(
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

def _anthropic_tools(
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

# Message Roles:
# LlamaCPP: Anything goes
# Ollama: ['user', 'assistant', 'system', 'tool']
# OpenAI: ['user', 'assistant', 'system', 'tool']
# Google: ['user', 'model']
# Anthropic: ['user', 'assistant']

# Source:
# https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion
# https://platform.openai.com/docs/api-reference/chat/create
# https://ai.google.dev/api/caching?_gl=1*rgisf*_up*MQ..&gclid=Cj0KCQiArby5BhCDARIsAIJvjIQ-aoQzhR9K-Qanjy99zZ3ajEkoarOm3BkBMCKi4cjpajQ8XYaqvOMaAsW0EALw_wcB&gbraid=0AAAAACn9t64WTefkrGIeU_Xn4Wd9fULrQ#Content
# https://docs.anthropic.com/en/api/messages

def _prepare_llamacpp_messages(messages):
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
                    ... # TODO: Exception
                    warnings.warn("PolyLLM does not yet support multi-modal input with LlamaCPP.")
                else:
                    ... # TODO: Exception
        else:
            ... # TODO: Exception

        messages_out.append({'role': role, 'content': content})

    return messages_out

def _prepare_ollama_messages(messages):
    messages_out = []

    for message in messages:
        assert 'role' in message # TODO: Explanation
        assert 'content' in message # TODO: Explanation

        role = message['role']
        content = []
        images = []

        if isinstance(message['content'], str):
            content = message['content']
        elif isinstance(message['content'], list):
            content = []
            for item in message['content']:
                assert 'type' in item # TODO: Explanation

                if item['type'] == 'text':
                    # content.append({'type': 'text', 'text': item['text']})
                    content.append(item['text'])
                elif item['type'] == 'image':
                    image_data = _load_image(item['image'])
                    images.append(image_data)
                else:
                    ... # TODO: Exception
            content = '\n'.join(content) # TODO: Necessary?
        else:
            ... # TODO: Exception

        if images: # TODO: If-statement necessary?
            messages_out.append({'role': role, 'content': content, 'images': images})
        else:
            messages_out.append({'role': role, 'content': content})

    return messages_out

def _prepare_openai_messages(messages):
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
                    image_data = _load_image(item['image'])
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

def _prepare_openai_tools(tools: list[Callable]):
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

def _prepare_google_messages(messages):
    messages_out = []

    for message in messages:
        assert 'role' in message # TODO: Explanation
        assert 'content' in message # TODO: Explanation

        if message['role'] == 'system':
            continue

        if message['role'] == 'assistant':
            role = 'model'
        else:
            role = message['role']

        if isinstance(message['content'], str):
            content = [message['content']]
        elif isinstance(message['content'], list):
            content = []
            for item in message['content']:
                assert 'type' in item # TODO: Explanation

                if item['type'] == 'text':
                    content.append(item['text'])
                elif item['type'] == 'image':
                    image_data = _load_image(item['image'])
                    content.append({'mime_type': 'image/jpeg', 'data': image_data})
                else:
                    ... # TODO: Exception
        else:
            ... # TODO: Exception

        messages_out.append({'role': role, 'parts': content})

    return messages_out

def _prepare_google_system_message(messages):
    system_message = None

    for message in messages:
        if message['role'] == 'system':
            system_message = message['content']
            break

    return system_message

def _prepare_anthropic_messages(messages):
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
                    image_data = _load_image(item['image'])
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

def _prepare_anthropic_system_message(messages):
    system_message = None

    for message in messages:
        if message['role'] == 'system':
            system_message = message['content']
            break

    return system_message

def _prepare_anthropic_tools(tools: list[Callable]):
    anthropic_tools = []

    for tool in tools:
        anthropic_tools.append({
            "name": tool.__name__,
            "description": tool.__doc__,
            "input_schema": {
                "type": "object",
                "properties": {
                    param: {"type": "number" if annotation == int else "string"}  # noqa: E721
                    for param, annotation in tool.__annotations__.items()
                    if param != 'return'
                },
                "required": list(tool.__annotations__.keys())[:-1]
            }
        })

    return anthropic_tools

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
