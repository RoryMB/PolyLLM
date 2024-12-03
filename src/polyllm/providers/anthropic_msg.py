import base64
from typing import Callable

from .general_msg import _load_image


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
