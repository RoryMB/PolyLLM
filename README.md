# PolyLLM

PolyLLM is a Python package that provides a unified interface for interacting with multiple Large Language Models (LLMs) through a single, consistent API. It simplifies the process of working with different LLM providers by abstracting away their specific implementation details.

Links:
- [Python Package Index (PyPI)](https://pypi.org/project/polyllm/)
- [GitHub Repository](https://github.com/RoryMB/PolyLLM)

## Features

- Unified interface for multiple LLM providers:
  - Local LLMs ([llama.cpp](https://github.com/ggerganov/llama.cpp), [llama-cpp-python](https://github.com/abetlen/llama-cpp-python))
  - [Ollama](https://ollama.com)
  - OpenAI (GPT models)
  - Google (Gemini models)
  - Anthropic (Claude models)
- Support for different interaction modes:
  - Standard chat completion
  - Multimodal through image input
  - Function calling / tools
  - JSON output
  - Structured output (using Pydantic models)
  - Streaming real-time responses

### Feature Support Across Providers

| Provider | Standard Chat | Image Input | JSON | Structured Output | Tool Usage |
|----------|---------------|-------------|------|-------------------|------------|
| llama.cpp | ✅ | 🔶 | ✅ | ✅ | ✅ |
| Ollama    | ✅ | ✅ | ✅ | ❌ | ✅ |
| Openai    | ✅ | ✅ | ✅ | ✅ | ✅ |
| Google    | ✅ | ✅ | ✅ | ✅ | ✅ |
| Anthropic | ✅ | ✅ | ✅ | ❌ | ✅ |

#### Streaming Output Modes

| Provider | Plain Text | JSON | Structured Output | Tool Usage |
|----------|------------|------|-------------------|------------|
| llama.cpp | ✅ | ✅ | ✅ | 🟫 |
| Ollama    | ✅ | ✅ | ❌ | 🟫 |
| Openai    | ✅ | ✅ | ❌ | 🟫 |
| Google    | ✅ | ✅ | ✅ | 🟫 |
| Anthropic | ✅ | 🟫 | ❌ | 🟫 |

✅: Supported

🔶: Support planned

❌: Not yet supported by the LLM provider

🟫: Support not planned


## Installation

```bash
pip install polyllm
```

```bash
pip install polyllm[all] # Gets all optional provider dependencies
```

### Requirements

- Python 3.9+
- `backoff`
- `pydantic`
- Optional dependencies for advanced image input:
  - `numpy`
  - `opencv-python`
  - `pillow`
- Optional dependencies based on which LLM providers you want to use:
  - `llama-cpp-python`
  - `ollama`
  - `openai`
  - `google-generativeai`
  - `anthropic`

## Configuration

Set your API keys as environment variables:

```bash
export OPENAI_API_KEY="your-key-here"
export GOOGLE_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
```

## Demo

```bash
python -m polyllm.demo \
    --image-path /path/to/image.jpg \
    --llama-python-model /path/to/model.gguf \
    --llama-python-server-port 8000 \
    --ollama-model llama3.1 \
    --openai-model gpt-4o \
    --google-model gemini-1.5-flash-latest \
    --anthropic-model claude-3-5-sonnet-latest
```

## Overview

### Model argument

The `model` argument may be provided as one of the following:
- An instance of `llama_cpp.Llama`
- `'llamacpp/MODEL'`, where `MODEL` is either the port or ip:port of a running llama-cpp-python server (`python -m llama_cpp.server --n_gpu_layers -1 --model path/to/model.gguf`)
    - Treated as `f'http://localhost:{MODEL}/v1'` if `MODEL` DOES NOT contain a `:`.
    - Treated as `f'http://{MODEL}/v1'` if `MODEL` DOES contain a `:`.
- `'ollama/MODEL_NAME'`, where `MODEL_NAME` matches the `ollama run MODEL_NAME` command
- `'openai/MODEL_NAME'`
- `'google/MODEL_NAME'`
- `'anthropic/MODEL_NAME'`
- `'MODEL_NAME'`, where `MODEL_NAME` is one of the models printed by `python -m polyllm`
    - Allows you to simply write `gpt-4` instead of `openai/gpt-4`.
    - May be somewhat less reliable, so prefer using the `provider/MODEL_NAME` syntax.

### Avaliable Functions

```python
def generate(
    model: str|Llama,
    messages: list,
    temperature: float = 0.0,
    json_output: bool = False,
    structured_output_model: BaseModel|None = None,
    stream: bool = False,
) -> str | Generator[str, None, None]:
```

Generate a chat message response as either a string or generator of strings depending on the `stream` argument.

```python
def generate_tools(
    model: str|Llama,
    messages: list,
    temperature: float = 0.0,
    tools: list[Callable] = None,
) -> tuple[str, str, dict]:
```

Ask the model to try to use one of the provided tools.

Responds with:
- Text reponse
- Tool name (Use `get_tool_func` to get the tool object)
- Tool arguments dictionary

```python
def get_tool_func(
    tools: list[Callable],
    tool: str,
) -> Callable:
```

Returns the tool corresponding to the name. Intended for use with the output of `generate_tools`.

```python
def structured_output_model_to_schema(
    structured_output_model: BaseModel,
    indent: int|str|None = None,
) -> str:
```

Creates a JSON schema string from a Pydantic model. Include the string in one of the messages in a `generate(..., structured_output_model)` call to help guide the model on how to respond.

```python
def structured_output_to_object(
    structured_output: str,
    structured_output_model: type[BaseModel],
) -> BaseModel:
```

Parse the output of a `generate(..., structured_output_model)` call into an instance of the Pydantic BaseModel.

## Quick Start Examples

```python
import polyllm
```

Run `python -m polyllm` to see the full list of detected Ollama, OpenAI, Google, and Anthropic models.

### Basic Usage
```python
response = polyllm.generate(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    temperature=0.2,
)
print(response)

# Prints:
# Hello! I'm just a computer program, so I don't have feelings, but I'm here to help you. How can I assist you today?
```

### Streaming Response
```python
for chunk in polyllm.generate(
    model="gpt-4",
    messages=[{"role": "user", "content": "Tell me a story"}],
    temperature=0.7,
    stream=True,
):
    print(chunk, end='', flush=True)
print()

# Prints (a word or so at a time):
# Once upon a time, ...
```

### Multimodal (Image Input)
```python
messages = [{
    "role": "user",
    "content": [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image", "image": "/path/to/image"},

        # These also work if you have the image as
        # an np.array / PIL Image instead of on disk:

        # {"type": "image", "image": cv2.imread("/path/to/image")},
        # {"type": "image", "image": Image.open("/path/to/image")},
    ],
}]

response = polyllm.generate(
    model="ollama/llama3.2-vision",
    messages=messages,
)
print(response)

# Prints:
# This image depicts ...
```

### Tools / Function Calling
```python
def multiply_large_numbers(x: int, y: int) -> int:
    """Multiplies two large numbers."""
    return x * y


tools = [multiply_large_numbers]
response, tool, args = polyllm.generate_tools(
    model="gemini-1.5-pro-latest",
    messages=[{"role": "user", "content": "What is 123456 multiplied by 654321?"}],
    tools=tools,
)

tool_func = polyllm.get_tool_func(tools, tool)
if tool_func:
    # print('response:', response)  # Some models (Anthropic) may return both their tool call AND a text response
    tool_result = tool_func(**args)
    print(tool_result)  # 123456 * 654321 = 80779853376
else:
    print(response)

# Prints:
# 80779853376.0
```

### JSON Output
```python
response = polyllm.generate(
    model="claude-3-5-sonnet-latest",
    messages=[{"role": "user", "content": "List three colors in JSON"}],
    json_output=True,
)
print(response)

# Prints:
# {
#   "colors": [
#     "red",
#     "blue",
#     "green"
#   ]
# }

import json
print(json.loads(response))

# Prints:
# {'colors': ['red', 'blue', 'green']}
```

### Structured Output
> [!WARNING]
> Not supported by Ollama or Anthropic.
```python
from pydantic import BaseModel, Field

class Flight(BaseModel):
    departure_time: str = Field(description="The time the flight departs")
    destination: str = Field(description="The destination of the flight")

class FlightList(BaseModel):
    flights: list[Flight] = Field(description="A list of known flight details")

flight_list_schema = polyllm.structured_output_model_to_schema(FlightList, indent=2)


response = polyllm.generate(
    model="gemini-1.5-pro-latest",
    messages=[{
        "role": "user",
        "content": f"Write a list of 2 to 5 random flight details.\nProduce the result in JSON that matches this schema:\n{flight_list_schema}",
    }],
    structured_output_model=FlightList,
)
print(response)

# Prints:
# {"flights": [{"departure_time": "2024-07-20T08:30", "destination": "JFK"}, {"departure_time": "2024-07-21T14:00", "destination": "LAX"}, {"departure_time": "2024-07-22T16:45", "destination": "ORD"}, {"departure_time": "2024-07-23T09:15", "destination": "SFO"}]}

response_object = polyllm.structured_output_to_object(response, FlightList)
print(response_object.flights[0].destination)

# Prints:
# JFK
```
