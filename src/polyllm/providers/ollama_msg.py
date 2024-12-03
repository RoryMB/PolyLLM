from .general_msg import _load_image


def _prepare_ollama_messages(messages):
    messages_out = []

    for message in messages:
        assert 'role' in message # TODO: Explanation
        assert 'content' in message # TODO: Explanation

        role = message['role']
        content = []
        images = []

        if isinstance(message['content'], str):
            content = message['content']
        elif isinstance(message['content'], list):
            content = []
            for item in message['content']:
                assert 'type' in item # TODO: Explanation

                if item['type'] == 'text':
                    # content.append({'type': 'text', 'text': item['text']})
                    content.append(item['text'])
                elif item['type'] == 'image':
                    image_data = _load_image(item['image'])
                    images.append(image_data)
                else:
                    ... # TODO: Exception
            content = '\n'.join(content) # TODO: Necessary?
        else:
            ... # TODO: Exception

        if images: # TODO: If-statement necessary?
            messages_out.append({'role': role, 'content': content, 'images': images})
        else:
            messages_out.append({'role': role, 'content': content})

    return messages_out
