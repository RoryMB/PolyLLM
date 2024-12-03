import json
from typing import Callable, Generator, Literal, overload
from functools import deprecated

from pydantic import BaseModel

from .providers import (
    anthropic_gen,
    google_gen,
    llamacpp_gen,
    llamapython_gen,
    # mlx_gen,
    ollama_gen,
    openai_gen,
)

from llama_cpp import Llama

MODEL_ERR_MSG = "PolyLLM could not find model: {model}. Run `python -m polyllm` to see a list of known models."

providers = {
    # 'mlx': mlx_gen,
    'llamapython': llamapython_gen,
    'llamacpp': llamacpp_gen,
    'ollama': ollama_gen,
    'openai': openai_gen,
    'google': google_gen,
    'anthropic': anthropic_gen,
}

# for plugin in get_plugins():
#     providers[plugin.name] = plugin

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

    t_provider, t_model = model.split('/', maxsplit=1)[1]

    if providers['llamapython'].did_import and isinstance(model, Llama):
        func = providers['llamapython']._generate
    elif t_provider in providers:
        func = providers[t_provider]._generate
        model = t_model
    else:
        for provider in providers.values():
            if model in provider.get_models():
                func = provider._generate
                break

    if not func:
        raise ValueError(MODEL_ERR_MSG.format(model=model))

    return func(model, messages, temperature, json_output, structured_output_model, stream)

@deprecated(version='2.0.0', reason='Function `generate_stream()` will be removed in v2.0.0. Use `generate(..., stream=True)` instead')
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

    t_provider, t_model = model.split('/', maxsplit=1)[1]

    if providers['llamapython'].did_import and isinstance(model, Llama):
        func = providers['llamapython']._generate_tools
    elif t_provider in providers:
        func = providers[t_provider]._generate_tools
        model = t_model
    else:
        for provider in providers.values():
            if model in provider.get_models():
                func = provider._generate_tools
                break

    if not func:
        raise ValueError(MODEL_ERR_MSG.format(model=model))

    return func(model, messages, temperature, tools)

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
