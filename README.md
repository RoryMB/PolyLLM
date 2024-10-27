# PolyLLM

PolyLLM is a Python package that provides a unified interface for interacting with multiple Large Language Models (LLMs) through a single, consistent API. It simplifies the process of working with different LLM providers by abstracting away their specific implementation details.

Links:
- [Python Package Index (PyPI)](https://pypi.org/project/polyllm/)
- [GitHub Repository](https://github.com/RoryMB/PolyLLM)

## Features

- Unified interface for multiple LLM providers:
  - Local LLMs (llama.cpp)
  - Ollama
  - OpenAI (GPT models)
  - Google (Gemini models)
  - Anthropic (Claude models)
- Support for different interaction modes:
  - Standard text completion
  - JSON output
  - Structured output (using Pydantic models)
  - Function calling / tools
- Streaming support for real-time responses

## Installation

```bash
pip install polyllm
```

```bash
pip install polyllm[all] # Get all llamacpp, ollama, openai, google, anthropic packages
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

## Quick Start

```python
import polyllm

# Basic usage
response = polyllm.generate(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    temperature=0.7,
)
print(response)

# Streaming response
for chunk in polyllm.generate_stream(
    model="gpt-4",
    messages=[{"role": "user", "content": "Tell me a story"}],
    temperature=0.7,
):
    print(chunk, end='', flush=True)

# Using tools/functions
def multiply(x: int, y: int) -> int:
    """Multiply two numbers"""
    return x * y

response, tool, args = polyllm.generate_tools(
    model="gemini-pro",
    messages=[{"role": "user", "content": "What is 7 times 6?"}],
    tools=[multiply],
)
print(response)

# JSON output
response = polyllm.generate(
    model="claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "List three colors in JSON"}],
    json_object=True,
)
print(response)

# Controlled JSON output
from pydantic import BaseModel, Field

class Flight(BaseModel):
    departure_time: str = Field(description="The time the flight departs")
    destination: str = Field(description="The destination of the flight")

class FlightList(BaseModel):
    flights: list[Flight] = Field(description="A list of known flight details")

messages = [
    {"role": "user", "content": "Write a list of 2 to 5 random flight details."}
]

response = polyllm.generate(model, messages, json_schema=FlightList)
print(response)
print(polyllm.json_to_pydantic(response, FlightList))
```

## Supported Models

- Local LLMs via llama.cpp
- Ollama models
- OpenAI: GPT-3.5, GPT-4, etc.
- Google: Gemini Pro, etc.
- Anthropic: Claude 3 (Opus, Sonnet, Haiku), Claude 2, etc.

| Model | Plain Text | Multimodal | JSON | Structured Output | Tool Usage | Streaming |
|-------|------------|------------|------|------------------|------------|-----------|
| Llama_cpp | ✅ | 🔶 | ✅ | ✅ | ✅ | ✅ |
| Ollama | ✅ | 🔶 | ✅ | ❌ | ✅ | ✅ |
| Openai | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Google | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Anthropic | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |

✅: Supported

🔶: Support planned

❌: Not yet supported by model provider

## Configuration

Set your API keys as environment variables:

```bash
export OPENAI_API_KEY="your-key-here"
export GOOGLE_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
```

## Requirements

- Python 3.9+
- Optional dependencies based on which LLM providers you want to use:
  - `openai`
  - `google-generativeai`
  - `anthropic`
  - `llama-cpp-python`
  - `ollama`
