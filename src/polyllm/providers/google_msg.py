from .general_msg import _load_image


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
