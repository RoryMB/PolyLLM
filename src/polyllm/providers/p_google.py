import time
from typing import Callable

import backoff
from pydantic import BaseModel

from ..utils import load_image

try:
    import google.generativeai as genai
    from google.api_core.exceptions import ResourceExhausted as GoogleResourceExhausted
    from google.generativeai.types import HarmBlockThreshold, HarmCategory
    genai.configure()
    did_import = True
except ImportError:
    did_import = False
    GoogleResourceExhausted = None

_models = []
def get_models():
    lazy_load()
    return _models

lazy_loaded = False
def lazy_load():
    global lazy_loaded, _models

    if lazy_loaded:
        return
    lazy_loaded = True

    if not did_import:
        return

    _models = sorted(model.name.split('/')[1] for model in genai.list_models() if 'generateContent' in model.supported_generation_methods)

genai.configure()

@backoff.on_exception(
    backoff.constant,
    GoogleResourceExhausted,
    interval=60,
    max_tries=4,
)
def generate(
    model: str,
    messages: list,
    temperature: float,
    json_output: bool,
    structured_output_model: BaseModel|None,
    stream: bool = False,
):
    system_message = prepare_system_message(messages)
    transformed_messages = prepare_messages(messages)

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

def generate_tools(
    model: str,
    messages: list,
    temperature: float,
    tools: list[Callable],
):
    system_message = prepare_system_message(messages)
    transformed_messages = prepare_messages(messages)

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
        # TODO: Change to `backoff`
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

def prepare_messages(messages):
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
                    image_data = load_image(item['image'])
                    content.append({'mime_type': 'image/jpeg', 'data': image_data})
                else:
                    ... # TODO: Exception
        else:
            ... # TODO: Exception

        messages_out.append({'role': role, 'parts': content})

    return messages_out

def prepare_system_message(messages):
    system_message = None

    for message in messages:
        if message['role'] == 'system':
            system_message = message['content']
            break

    return system_message
