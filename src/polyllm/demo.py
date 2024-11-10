import os
import textwrap
import argparse
import sys
from . import polyllm
from pydantic import BaseModel, Field


def red(text):
    return '\033[38;5;197m' + text + '\033[0m'
def green(text):
    return '\033[38;5;41m' + text + '\033[0m'
def purple(text):
    return '\033[38;5;93m' + text + '\033[0m'

parser = argparse.ArgumentParser(description="Run polyllm demo with various models.")
parser.add_argument("--image-path", help="Path to the image file for multi-modal tests")
parser.add_argument("--llama-python-model", help="Path to the llama-cpp-python model file (.gguf). Llama3.1 is suggested for good JSON and function calling/tool usage. Find it at: https://huggingface.co/bullerwins/Meta-Llama-3.1-8B-Instruct-GGUF/tree/main")
parser.add_argument("--llama-python-server-port", help="Port for the llama.cpp server. Make sure llama_cpp is running: `python -m llama_cpp.server --n_gpu_layers -1 --model path/to/model.gguf`")
parser.add_argument("--ollama-model", help="Name of the Ollama model to use. Make sure Ollama is running. Llama3.1 is suggested, pull it with: `ollama pull llama3.1`")
parser.add_argument("--openai-model", help="Name of the OpenAI model to use (e.g., gpt-4o)")
parser.add_argument("--google-model", help="Name of the Google model to use (e.g., gemini-1.5-pro). 'gemini-1.5-pro' is required for structured output use.")
parser.add_argument("--anthropic-model", help="Name of the Anthropic model to use (e.g., claude-3-5-sonnet-latest)")
args = parser.parse_args()

# If no arguments are provided, print usage and exit
if len(sys.argv) == 1:
    print("Usage: python -m polyllm.demo [options]")
    print("Run 'python -m polyllm.demo --help' for more information on available options.")
    sys.exit(1)

IMAGE_PATH = args.image_path or ''
LLAMA_PYTHON_MODEL = args.llama_python_model or ''
LLAMA_PYTHON_SERVER_PORT = args.llama_python_server_port or ''
OLLAMA_MODEL = args.ollama_model or ''
OPENAI_MODEL = args.openai_model or ''
GOOGLE_MODEL = args.google_model or ''
ANTHROPIC_MODEL = args.anthropic_model or ''

image_exists = os.path.isfile(IMAGE_PATH)

if not polyllm.llamapython_import:
    print(red("Import for `llama_cpp_python` failed. Run `pip install -U polyllm[all]` or `pip install llama_cpp_python`"))
    LLAMA_PYTHON_MODEL = ''
    LLAMA_PYTHON_SERVER_PORT = ''
# if not polyllm.ollama_import:
#     print(red("Import for `ollama` failed. Run `pip install -U polyllm[all]` or `pip install ollama`"))
#     OLLAMA_MODEL = '' # ALSO FIX THE ` OLLAMA_MODEL = ''` BELOW!!!!!!
if not polyllm.openai_import:
    print(red("Import for `openai` failed. Run `pip install -U polyllm[all]` or `pip install openai`"))
    OPENAI_MODEL = ''
    OLLAMA_MODEL = ''
if not polyllm.google_import:
    print(red("Import for `google-generativeai` failed. Run `pip install -U polyllm[all]` or `pip install google-generativeai`"))
    GOOGLE_MODEL = ''
if not polyllm.anthropic_import:
    print(red("Import for `anthropic` failed. Run `pip install -U polyllm[all]` or `pip install anthropic`"))
    ANTHROPIC_MODEL = ''

print()

if not image_exists:
    print(red("The image file was not specified or does not exist. Multi-modal tests will be skipped."))
    print(red("Use the --image-path option to point to a valid image file."))
    print()
if not LLAMA_PYTHON_MODEL:
    print(red("No llama-cpp-python model specified. llama-cpp-python tests will be skipped."))
    print(red("Use the --llama-python-model option to point to a .gguf file."))
    print()
if not LLAMA_PYTHON_SERVER_PORT:
    print(red("No llama.cpp port specified. llama.cpp tests will be skipped."))
    print(red("Use the --llama-python-server-port option to point to a running llama_cpp_python server."))
    print()
if not OLLAMA_MODEL:
    print(red("No Ollama model specified. Ollama tests will be skipped."))
    print(red("Use the --ollama-model option to specify a downloaded Ollama model."))
    print()
if not OPENAI_MODEL:
    print(red("No OpenAI model specified. OpenAI tests will be skipped."))
    print(red("Use the --openai-model option to specify an OpenAI model."))
    print()
if not GOOGLE_MODEL:
    print(red("No Google model specified. Google tests will be skipped."))
    print(red("Use the --google-model option to specify a Google model."))
    print()
if not ANTHROPIC_MODEL:
    print(red("No Anthropic model specified. Anthropic tests will be skipped."))
    print(red("Use the --anthropic-model option to specify an Anthropic model."))
    print()


# Example for plain text conversations.
# Tests correct handling of system, user, and assistant message roles.
text_messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me a joke."},
    {"role": "assistant", "content": "Why did the scarecrow win an award?\nBecause he was outstanding in his field!"},
    {"role": "user", "content": "Great! Tell me another joke!"},
]


# Example for multimodal conversations.
# Tests correct handling of images.
image_messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image", "image": IMAGE_PATH},
        ],
    },
]


# Example for json_output output.
# Tests enforcement of JSON responses.
json_messages = [
    {
        "role": "user",
        "content": textwrap.dedent("""
            Find the name of the first president of the USA and get the years that he served.
            Produce the result in JSON that matches this schema:
                {
                    "first_name": "first name",
                    "last_name":  "last name",
                    "years_served": "years served"
                }
            """).strip()
    }
]



class Flight(BaseModel):
    departure_time: str = Field(description="The time the flight departs")
    destination: str = Field(description="The destination of the flight")
class FlightList(BaseModel):
    flights: list[Flight] = Field(description="A list of known flight details")
flight_list_schema = polyllm.structured_output_model_to_schema(FlightList, indent=2)
pydantic_messages = [
    {
        "role": "user",
        "content": textwrap.dedent(f"""
            Write a list of 2 to 5 random flight details.
            Produce the result in JSON that matches this schema:
            {flight_list_schema}
            """).strip()
    }
]



# Example for function calling / tool usage.
# Tests ability to use tools to answer questions, answer questions without tools when no tool is helpful, and avoid using tools or trying to answer when neither is applicable.
def multiply_large_numbers(a: int, b: int) -> int:
    """Multiplies two large numbers."""
    return a * b
# Extracted: function name, argument names, argument types, docstring

tool_message0 = [{"role": "user", "content": "What is 123456 multiplied by 654321?"}]
tool_message1 = [{"role": "user", "content": "How old was George Washington when he became president?"}]
tool_message2 = [{"role": "user", "content": "What is the current temperature in Kalamazoo?"}]



# Examples

if LLAMA_PYTHON_MODEL:
    from llama_cpp import Llama
    llm = Llama(
        model_path=LLAMA_PYTHON_MODEL,
        n_ctx=1024,
        n_gpu_layers=-1,
        verbose=False,
    )
    print(green('======== LlamaCPP Python'))
    print(purple("==== Testing plain text conversation `polyllm.generate(model, messages)`: (Should tell a joke)"))
    print(polyllm.generate(llm, text_messages))
    print(purple("\n==== Testing JSON mode `polyllm.generate(model, messages, json_output=True)`: (Should format: George, Washington, 1789-1797)"))
    print(polyllm.generate(llm, json_messages, json_output=True))
    print(purple("\n==== Testing Structured Output mode `polyllm.generate(model, messages, structured_output_model=PydanticModel)`: (Should list 2-5 random flight times and destinations)"))
    output = polyllm.generate(llm, json_messages, structured_output_model=FlightList)
    print(output)
    print(polyllm.structured_output_to_object(output, FlightList))
    print(purple("\n==== Testing tool usage `polyllm.generate(model, messages, tools=[func])`: (Should choose multiply_large_numbers, a=123456, b=654321)"))
    print(polyllm.generate_tools(llm, tool_message0, tools=[multiply_large_numbers]))
    print(purple("\n==== Testing tool usage with no relevant tool `polyllm.generate(model, messages, tools=[func])`: (Should respond 57)"))
    print(polyllm.generate_tools(llm, tool_message1, tools=[multiply_large_numbers]))
    print(purple("\n==== Testing tool usage with no relevant tool or knowledge `polyllm.generate(model, messages, tools=[func])`: (Should refuse to respond)"))
    print(polyllm.generate_tools(llm, tool_message2, tools=[multiply_large_numbers]))
    print(purple("\n==== Testing streaming `polyllm.generate(model, messages, stream=True)`: (Should tell a joke)"))
    for chunk in polyllm.generate(llm, text_messages, stream=True):
        print(chunk, end='', flush=True)
    print()

if LLAMA_PYTHON_SERVER_PORT:
    print(green(f'======== LlamaCPP Python Server (model="llamacpp/{LLAMA_PYTHON_SERVER_PORT}")'))
    print(purple("==== Testing plain text conversation `polyllm.generate(model, messages)`: (Should tell a joke)"))
    print(polyllm.generate(f"llamacpp/{LLAMA_PYTHON_SERVER_PORT}", text_messages))
    print(purple("\n==== Testing JSON mode `polyllm.generate(model, messages, json_output=True)`: (Should format: George, Washington, 1789-1797)"))
    print(polyllm.generate(f"llamacpp/{LLAMA_PYTHON_SERVER_PORT}", json_messages, json_output=True))
    print(purple("\n==== Testing Structured Output mode `polyllm.generate(model, messages, structured_output_model=PydanticModel)`: (Should list 2-5 random flight times and destinations)"))
    output = polyllm.generate(f"llamacpp/{LLAMA_PYTHON_SERVER_PORT}", json_messages, structured_output_model=FlightList)
    print(output)
    print(polyllm.structured_output_to_object(output, FlightList))
    print(purple("\n==== Testing tool usage `polyllm.generate(model, messages, tools=[func])`: (Should choose multiply_large_numbers, a=123456, b=654321)"))
    print(polyllm.generate_tools(f"llamacpp/{LLAMA_PYTHON_SERVER_PORT}", tool_message0, tools=[multiply_large_numbers]))
    print(purple("\n==== Testing tool usage with no relevant tool `polyllm.generate(model, messages, tools=[func])`: (Should respond 57)"))
    print(polyllm.generate_tools(f"llamacpp/{LLAMA_PYTHON_SERVER_PORT}", tool_message1, tools=[multiply_large_numbers]))
    print(purple("\n==== Testing tool usage with no relevant tool or knowledge `polyllm.generate(model, messages, tools=[func])`: (Should refuse to respond)"))
    print(polyllm.generate_tools(f"llamacpp/{LLAMA_PYTHON_SERVER_PORT}", tool_message2, tools=[multiply_large_numbers]))
    print(purple("\n==== Testing streaming `polyllm.generate(model, messages, stream=True)`: (Should tell a joke)"))
    for chunk in polyllm.generate(f"llamacpp/{LLAMA_PYTHON_SERVER_PORT}", text_messages, stream=True):
        print(chunk, end='', flush=True)
    print()

if OLLAMA_MODEL:
    print(green(f'\n======== Ollama (model="ollama/{OLLAMA_MODEL}")'))
    print(purple("==== Testing plain text conversation `polyllm.generate(model, messages)`: (Should tell a joke)"))
    print(polyllm.generate(f"ollama/{OLLAMA_MODEL}", text_messages))
    if image_exists:
        print(purple("\n==== Testing multi-modal image input: (Should describe your image)"))
        print(polyllm.generate(f"ollama/{OLLAMA_MODEL}", image_messages))
    print(purple("\n==== Testing JSON mode `polyllm.generate(model, messages, json_output=True)`: (Should format: George, Washington, 1789-1797)"))
    print(polyllm.generate(f"ollama/{OLLAMA_MODEL}", json_messages, json_output=True))
    print(purple("\n==== Testing tool usage `polyllm.generate(model, messages, tools=[func])`: (Should choose multiply_large_numbers, a=123456, b=654321)"))
    print(polyllm.generate_tools(f"ollama/{OLLAMA_MODEL}", tool_message0, tools=[multiply_large_numbers]))
    print(purple("\n==== Testing tool usage with no relevant tool `polyllm.generate(model, messages, tools=[func])`: (Should respond 57)"))
    print(polyllm.generate_tools(f"ollama/{OLLAMA_MODEL}", tool_message1, tools=[multiply_large_numbers]))
    print(purple("\n==== Testing tool usage with no relevant tool or knowledge `polyllm.generate(model, messages, tools=[func])`: (Should refuse to respond)"))
    print(polyllm.generate_tools(f"ollama/{OLLAMA_MODEL}", tool_message2, tools=[multiply_large_numbers]))
    print(purple("\n==== Testing streaming `polyllm.generate(model, messages, stream=True)`: (Should tell a joke)"))
    for chunk in polyllm.generate(f"ollama/{OLLAMA_MODEL}", text_messages, stream=True):
        print(chunk, end='', flush=True)
    print()

if OPENAI_MODEL:
    print(green(f'\n======== OpenAI (model="{OPENAI_MODEL}")'))
    print(purple("==== Testing plain text conversation `polyllm.generate(model, messages)`: (Should tell a joke)"))
    print(polyllm.generate(OPENAI_MODEL, text_messages))
    if image_exists:
        print(purple("\n==== Testing multi-modal image input: (Should describe your image)"))
        print(polyllm.generate(OPENAI_MODEL, image_messages))
    print(purple("\n==== Testing JSON mode `polyllm.generate(model, messages, json_output=True)`: (Should format: George, Washington, 1789-1797)"))
    print(polyllm.generate(OPENAI_MODEL, json_messages, json_output=True))
    print(purple("\n==== Testing Structured Output mode `polyllm.generate(model, messages, structured_output_model=PydanticModel)`: (Should list 2-5 random flight times and destinations)"))
    output = polyllm.generate(OPENAI_MODEL, json_messages, structured_output_model=FlightList)
    print(output)
    print(polyllm.structured_output_to_object(output, FlightList))
    print(purple("\n==== Testing tool usage `polyllm.generate(model, messages, tools=[func])`: (Should choose multiply_large_numbers, a=123456, b=654321)"))
    print(polyllm.generate_tools(OPENAI_MODEL, tool_message0, tools=[multiply_large_numbers]))
    print(purple("\n==== Testing tool usage with no relevant tool `polyllm.generate(model, messages, tools=[func])`: (Should respond 57)"))
    print(polyllm.generate_tools(OPENAI_MODEL, tool_message1, tools=[multiply_large_numbers]))
    print(purple("\n==== Testing tool usage with no relevant tool or knowledge `polyllm.generate(model, messages, tools=[func])`: (Should refuse to respond)"))
    print(polyllm.generate_tools(OPENAI_MODEL, tool_message2, tools=[multiply_large_numbers]))
    print(purple("\n==== Testing streaming `polyllm.generate(model, messages, stream=True)`: (Should tell a joke)"))
    for chunk in polyllm.generate(OPENAI_MODEL, text_messages, stream=True):
        print(chunk, end='', flush=True)
    print()

if GOOGLE_MODEL:
    print(green(f'\n======== Google (model="{GOOGLE_MODEL}")'))
    print(purple("==== Testing plain text conversation `polyllm.generate(model, messages)`: (Should tell a joke)"))
    print(polyllm.generate(GOOGLE_MODEL, text_messages))
    if image_exists:
        print(purple("\n==== Testing multi-modal image input: (Should describe your image)"))
        print(polyllm.generate(GOOGLE_MODEL, image_messages))
    print(purple("\n==== Testing JSON mode `polyllm.generate(model, messages, json_output=True)`: (Should format: George, Washington, 1789-1797)"))
    print(polyllm.generate(GOOGLE_MODEL, json_messages, json_output=True))
    print(purple("\n==== Testing Structured Output mode `polyllm.generate(model, messages, structured_output_model=PydanticModel)`: (Should list 2-5 random flight times and destinations)"))
    print(purple('\n(model="gemini-1.5-pro")'))
    output = polyllm.generate("gemini-1.5-pro", json_messages, structured_output_model=FlightList)
    print(output)
    print(polyllm.structured_output_to_object(output, FlightList))
    print(purple("\n==== Testing tool usage `polyllm.generate(model, messages, tools=[func])`: (Should choose multiply_large_numbers, a=123456, b=654321)"))
    print(polyllm.generate_tools(GOOGLE_MODEL, tool_message0, tools=[multiply_large_numbers]))
    print(purple("\n==== Testing tool usage with no relevant tool `polyllm.generate(model, messages, tools=[func])`: (Should respond 57)"))
    print(polyllm.generate_tools(GOOGLE_MODEL, tool_message1, tools=[multiply_large_numbers]))
    print(purple("\n==== Testing tool usage with no relevant tool or knowledge `polyllm.generate(model, messages, tools=[func])`: (Should refuse to respond)"))
    print(polyllm.generate_tools(GOOGLE_MODEL, tool_message2, tools=[multiply_large_numbers]))
    print(purple("\n==== Testing streaming `polyllm.generate(model, messages, stream=True)`: (Should tell a joke)"))
    for chunk in polyllm.generate(GOOGLE_MODEL, text_messages, stream=True):
        print(chunk, end='', flush=True)
    print()

if ANTHROPIC_MODEL:
    print(green(f'\n======== Anthropic (model="{ANTHROPIC_MODEL}")'))
    print(purple("==== Testing plain text conversation `polyllm.generate(model, messages)`: (Should tell a joke)"))
    print(polyllm.generate(ANTHROPIC_MODEL, text_messages))
    if image_exists:
        print(purple("\n==== Testing multi-modal image input: (Should describe your image)"))
        print(polyllm.generate(ANTHROPIC_MODEL, image_messages))
    print(purple("\n==== Testing JSON mode `polyllm.generate(model, messages, json_output=True)`: (Should format: George, Washington, 1789-1797)"))
    print(polyllm.generate(ANTHROPIC_MODEL, json_messages, json_output=True))
    print(purple("\n==== Testing tool usage `polyllm.generate(model, messages, tools=[func])`: (Should choose multiply_large_numbers, a=123456, b=654321)"))
    print(polyllm.generate_tools(ANTHROPIC_MODEL, tool_message0, tools=[multiply_large_numbers]))
    print(purple("\n==== Testing tool usage with no relevant tool `polyllm.generate(model, messages, tools=[func])`: (Should respond 57)"))
    print(polyllm.generate_tools(ANTHROPIC_MODEL, tool_message1, tools=[multiply_large_numbers]))
    print(purple("\n==== Testing tool usage with no relevant tool or knowledge `polyllm.generate(model, messages, tools=[func])`: (Should refuse to respond)"))
    print(polyllm.generate_tools(ANTHROPIC_MODEL, tool_message2, tools=[multiply_large_numbers]))
    print(purple("\n==== Testing streaming `polyllm.generate(model, messages, stream=True)`: (Should tell a joke)"))
    for chunk in polyllm.generate(ANTHROPIC_MODEL, text_messages, stream=True):
        print(chunk, end='', flush=True)
    print()
