from typing import Callable

from pydantic import BaseModel

try:
    import mlx_lm
    from mlx_lm.sample_utils import make_sampler
    did_import = True
except ImportError:
    did_import = False


_models = []
def get_models():
    return _models

def generate(
    model: str,
    messages: list,
    temperature: float,
    json_output: bool,
    structured_output_model: BaseModel|None,
    stream: bool = False,
):
    transformed_messages = prepare_messages(messages)

    model, tokenizer = mlx_lm.load(model)
    prompt = tokenizer.apply_chat_template(transformed_messages, tokenize=False, add_generation_prompt=True)
    sampler = make_sampler(temp=temperature)

    kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "prompt": prompt,
        "max_tokens": 8192,
        "sampler": sampler,
    }

    if json_output:
        raise NotImplementedError("PolyLLM does not yet support JSON output with MLX.")
    if structured_output_model is not None:
        raise NotImplementedError("PolyLLM does not yet support Structured Output with MLX.")

    if stream:
        stream_generator = mlx_lm.stream_generate(**kwargs)
        return (t.text for t in stream_generator)
    else:
        text = mlx_lm.generate(**kwargs)
        return text

def generate_tools(
    model: str,
    messages: list,
    temperature: float,
    tools: list[Callable],
):
    raise NotImplementedError("PolyLLM does not yet support Tool Use with MLX.")

def prepare_messages(messages):
    messages_out = []

    for message in messages:
        assert 'role' in message # TODO: Explanation
        assert 'content' in message # TODO: Explanation

        role = message['role']
        content = []

        if isinstance(message['content'], str):
            content = message['content']
        elif isinstance(message['content'], list):
            content = []
            for item in message['content']:
                assert 'type' in item # TODO: Explanation

                if item['type'] == 'text':
                    content.append({'type': 'text', 'text': item['text']})
                elif item['type'] == 'image':
                    # image_data = load_image(item['image'])
                    # content.append({'type': 'image', 'image': image_data})
                    raise NotImplementedError("PolyLLM does not yet support images with MLX.")
                else:
                    ... # TODO: Exception
        else:
            ... # TODO: Exception

        messages_out.append({'role': role, 'content': content})

    return messages_out
