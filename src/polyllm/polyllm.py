import functools
import warnings
from typing import Callable, Generator, Literal, Union, overload

from pydantic import BaseModel

from .providers import (
    p_anthropic,
    p_google,
    p_litellm,
    p_llamacpppython,
    p_llamacppserver,
    p_mlx,
    p_ollama,
    p_openai,
)

try:
    from llama_cpp import Llama
except ImportError:
    pass

try:
    import mlx.nn
    from mlx_lm.tokenizer_utils import TokenizerWrapper
except ImportError:
    pass

providers = {
    'llamacpppython': p_llamacpppython,
    'llamacpp': p_llamacppserver,
    'mlx': p_mlx,
    'ollama': p_ollama,
    'litellm': p_litellm,
    'openai': p_openai,
    'google': p_google,
    'anthropic': p_anthropic,
}

# for plugin in get_plugins():
#     providers[plugin.name] = plugin

@overload
def generate(
    model: Union[str, "Llama", tuple["mlx.nn.Module", "TokenizerWrapper"]],
    messages: list,
    temperature: float = 0.0,
    json_output: bool = False,
    structured_output_model: BaseModel|None = None,
    stream: Literal[False] = False,
) -> str: ...

@overload
def generate(
    model: Union[str, "Llama", tuple["mlx.nn.Module", "TokenizerWrapper"]],
    messages: list,
    temperature: float = 0.0,
    json_output: bool = False,
    structured_output_model: BaseModel|None = None,
    stream: Literal[True] = True,
) -> Generator[str, None, None]: ...

def generate(
    model: Union[str, "Llama", tuple["mlx.nn.Module", "TokenizerWrapper"]],
    messages: list,
    temperature: float = 0.0,
    json_output: bool = False,
    structured_output_model: BaseModel|None = None,
    stream: bool = False,
) -> str | Generator[str, None, None]:
    if json_output and structured_output_model:
        raise RuntimeError("generate() cannot simultaneously support JSON mode (json_output) and Structured Output mode (structured_output_model)")

    match model:
        case Llama() if providers['llamacpppython'].did_import:
            func = providers['llamacpppython'].generate
        case (mlx.nn.Module(), TokenizerWrapper()) if providers['mlx'].did_import:
            func = providers['mlx'].generate
        case str():
            if '/' in model:
                provider, model = model.split('/', maxsplit=1)
                if provider not in providers:
                    raise RuntimeError(f"PolyLLM could not find provider: {provider}.")
                if not providers[provider].did_import:
                    raise ImportError(f"PolyLLM failed necessary imports for provider: {provider}.")
                func = providers[provider].generate
            else:
                for provider in providers.values():
                    if model in provider.get_models():
                        func = provider.generate
                        break
                else:
                    raise RuntimeError(f"PolyLLM could not find model: {model}. Run `python -m polyllm` to see a list of known models.")
        case _:
            raise RuntimeError(f"Unexpected model type: {type(model)}. PolyLLM expects a string, llama_cpp.Llama object, or (mlx.nn.Module, mlx_lm.tokenizer_utils.TokenizerWrapper) tuple.")

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
    model: Union[str, "Llama", tuple["mlx.nn.Module", "TokenizerWrapper"]],
    messages: list,
    temperature: float = 0.0,
    json_output: bool = False,
    structured_output_model: BaseModel|None = None,
) -> Generator[str, None, None]:
    return generate(model, messages, temperature, json_output, structured_output_model, stream=True)

def generate_tools(
    model: Union[str, "Llama", tuple["mlx.nn.Module", "TokenizerWrapper"]],
    messages: list,
    temperature: float = 0.0,
    tools: list[Callable] = None,
) -> tuple[str, str, dict]:
    match model:
        case Llama() if providers['llamacpppython'].did_import:
            func = providers['llamacpppython'].generate_tools
        case (mlx.nn.Module(), TokenizerWrapper()) if providers['mlx'].did_import:
            func = providers['mlx'].generate_tools
        case str():
            if '/' in model:
                provider, model = model.split('/', maxsplit=1)
                if provider not in providers:
                    raise RuntimeError(f"PolyLLM could not find provider: {provider}.")
                if not providers[provider].did_import:
                    raise ImportError(f"PolyLLM failed necessary imports for provider: {provider}.")
                func = providers[provider].generate_tools
            else:
                for provider in providers.values():
                    if model in provider.get_models():
                        func = provider.generate_tools
                        break
                else:
                    raise RuntimeError(f"PolyLLM could not find model: {model}. Run `python -m polyllm` to see a list of known models.")
        case _:
            raise RuntimeError(f"Unexpected model type: {type(model)}. PolyLLM expects a string, llama_cpp.Llama object, or (mlx.nn.Module, mlx_lm.tokenizer_utils.TokenizerWrapper) tuple.")

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
