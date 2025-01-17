import functools
import warnings
from typing import Callable, Generator, Literal, overload

from llama_cpp import Llama
from pydantic import BaseModel

from .providers import (
    p_anthropic,
    p_google,
    p_llamacpppython,
    p_llamacppserver,
    p_mlx,
    p_ollama,
    p_openai,
)


MODEL_ERR_MSG = "PolyLLM could not find model: {model}. Run `python -m polyllm` to see a list of known models."

providers = {
    'llamacpppython': p_llamacpppython,
    'llamacpp': p_llamacppserver,
    'mlx': p_mlx,
    'ollama': p_ollama,
    'openai': p_openai,
    'google': p_google,
    'anthropic': p_anthropic,
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


    if providers['llamacpppython'].did_import and isinstance(model, Llama):
        func = providers['llamacpppython'].generate
    else:
        t_provider = model.split('/', maxsplit=1)[0]
        if t_provider in providers:
            if not providers[t_provider].did_import:
                raise ImportError(f"PolyLLM failed necessary imports for provider: {t_provider}.")
            func = providers[t_provider].generate
            model = model.split('/', maxsplit=1)[1]
        else:
            for provider in providers.values():
                if model in provider.get_models():
                    func = provider.generate
                    break

    if not func:
        raise ValueError(MODEL_ERR_MSG.format(model=model))

    return func(model, messages, temperature, json_output, structured_output_model, stream)

def deprecated(reason):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(f"{func.__name__} is deprecated: {reason}",
                        category=DeprecationWarning,
                        stacklevel=2)
            return func(*args, **kwargs)
        return wrapper
    return decorator

@deprecated(reason='Function `generate_stream()` will be removed in v2.0.0. Use `generate(..., stream=True)` instead')
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


    if providers['llamacpppython'].did_import and isinstance(model, Llama):
        func = providers['llamacpppython'].generate_tools
    else:
        t_provider = model.split('/', maxsplit=1)[0]
        if t_provider in providers:
            func = providers[t_provider].generate_tools
            model = model.split('/', maxsplit=1)[1]
        else:
            for provider in providers.values():
                if model in provider.get_models():
                    func = provider.generate_tools
                    break

    if not func:
        raise ValueError(MODEL_ERR_MSG.format(model=model))

    return func(model, messages, temperature, tools)

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
