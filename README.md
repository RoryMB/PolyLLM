# PolyLLM

PolyLLM is a Python package that provides a unified interface for interacting with multiple Large Language Models (LLMs) through a single, consistent API. It simplifies the process of working with different LLM providers by abstracting away their specific implementation details.

Links:
- [Python Package Index (PyPI)](https://pypi.org/project/polyllm/)
- [GitHub Repository](https://github.com/RoryMB/PolyLLM)

## Features

- Unified interface for multiple LLM providers:
  - Local LLMs ([llama.cpp](https://github.com/ggerganov/llama.cpp))
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
| Ollama    | ✅ | 🔶 | ✅ | ❌ | ✅ |
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
pip install polyllm[all] # Gets all llamacpp, ollama, openai, google, anthropic packages
```

### Requirements

- Python 3.9+
- Optional dependencies based on which LLM providers you want to use:
  - `openai`
  - `google-generativeai`
  - `anthropic`
  - `llama-cpp-python`
  - `ollama`

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
    --ollama-model llama3.2 \
    --openai-model gpt-4o \
    --google-model gemini-1.5-flash-latest \
    --anthropic-model claude-3-5-sonnet-latest
```

## Overview

generate()

Options for MODEL

generate_tools()

pydantic_to_schema()

json_to_pydantic()

get_tool_func()

## Quick Start Examples

```python
import polyllm
```

Run `python -m polyllm` to see the full list of known OpenAI, Google, and Anthropic models.

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
        {"type": "image_path", "image_path": "/path/to/image"},

        # These also work if you have the image as
        # an np.array / PIL Image instead of on disk:

        # {"type": "image_cv2", "image_cv2": cv2.imread("/path/to/image")},
        # {"type": "image_pil", "image_pil": Image.open("/path/to/image")},
    ],
}]

response = polyllm.generate(
    model="claude-3-5-sonnet-latest",
    messages=messages,
)
print(response)
```

### Using Tools / Function Calling
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
    json_object=True,
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


response = polyllm.generate(
    model="gemini-1.5-pro-latest",
    messages=[{"role": "user", "content": "Write a list of 2 to 5 random flight details."}],
    json_schema=FlightList,
)
print(response)

# Prints:
# {"flights": [{"departure_time": "2024-07-20T08:30", "destination": "JFK"}, {"departure_time": "2024-07-21T14:00", "destination": "LAX"}, {"departure_time": "2024-07-22T16:45", "destination": "ORD"}, {"departure_time": "2024-07-23T09:15", "destination": "SFO"}]}

pydantic_object = polyllm.json_to_pydantic(response, FlightList)
print(pydantic_object.flights[0].destination)

# Prints:
# JFK
```
