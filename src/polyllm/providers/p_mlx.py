from pathlib import Path
from typing import Callable
import warnings

from pydantic import BaseModel

from ..utils import structured_output_model_to_schema
from . import gbnf

try:
    import mlx.core as mx
    import mlx.nn
    import mlx_lm
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.tokenizer_utils import TokenizerWrapper
    from llama_cpp.llama_grammar import json_schema_to_gbnf
    did_import = True
except ImportError:
    did_import = False


_models = []
def get_models():
    return _models

def generate(
    model: tuple[mlx.nn.Module, TokenizerWrapper],
    messages: list,
    temperature: float,
    json_output: bool,
    structured_output_model: BaseModel|None,
    stream: bool = False,
):
    transformed_messages = prepare_messages(messages)

    model_instance, tokenizer = model
    prompt = tokenizer.apply_chat_template(transformed_messages, tokenize=False, add_generation_prompt=True)
    sampler = make_sampler(temp=temperature)

    kwargs = {
        "model": model_instance,
        "tokenizer": tokenizer,
        "prompt": prompt,
        "max_tokens": 512 if json_output or structured_output_model else -1,
        "sampler": sampler,
    }

    if json_output:
        grammar_file = Path(__file__).parent / "json.gbnf"
        grammar = grammar_file.read_text()
        kwargs['sampler'] = GrammarSampler(grammar, tokenizer, kwargs['sampler'])
        warnings.warn("PolyLLM: JSON output with MLX is experimental. Expect bugs and lag. Prefer using Structured Output mode.", category=Warning, stacklevel=2)
    if structured_output_model:
        schema = structured_output_model_to_schema(structured_output_model)
        grammar = json_schema_to_gbnf(schema)
        kwargs['sampler'] = GrammarSampler(grammar, tokenizer, kwargs['sampler'])
        warnings.warn("PolyLLM: Structured Output with MLX is experimental. Expect bugs and lag.", category=Warning, stacklevel=2)

    if stream:
        stream_generator = mlx_lm.stream_generate(**kwargs)
        return (t.text for t in stream_generator)
    else:
        return mlx_lm.generate(**kwargs)

def generate_tools(
    model: tuple[mlx.nn.Module, TokenizerWrapper],
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
            for item in message['content']:
                if item.get('type') == 'image':
                    # image_data = load_image(item['image'])
                    # content.append({'type': 'image', 'image': image_data})
                    raise NotImplementedError("PolyLLM does not yet support images with MLX.")
        else:
            ... # TODO: Exception?

        messages_out.append({'role': role, 'content': content})

    return messages_out

class GrammarSampler:
    def __init__(self, grammar: str, tokenizer, sub_sampler=None):
        try:
            self.parse_state = gbnf.GBNF(grammar)
        except gbnf.GrammarParseError as e:
            raise ValueError(f"Error parsing grammar: {e}")

        self.tokenizer = tokenizer
        self.sub_sampler = sub_sampler or (lambda x: mx.argmax(x, axis=-1))

    def __call__(self, logprobs: 'mx.array') -> 'mx.array':
        # If the grammar is at a valid end state, send EOS token
        if self.is_grammar_complete():
            return mx.array([list(self.tokenizer.eos_token_ids)[0]], dtype=mx.uint32)

        logprobs_copy = logprobs[:]

        for _ in range(len(logprobs_copy[0])):
            best_token_id = self.sub_sampler(logprobs_copy)

            text = self.tokenizer._tokenizer._decode(best_token_id.item())

            # Attempt to add the token to the grammar state
            try:
                new_state = self.parse_state(text)
            except gbnf.InputParseError:
                logprobs_copy[0, best_token_id] = -mx.inf
                continue

            break
        else:
            raise RuntimeError("Bug in GrammarSampler: No valid continuations possible.")

        # Update the grammar state with the selected token text
        self.parse_state = new_state

        return best_token_id

    def is_grammar_complete(self) -> bool:
        """Checks if the current grammar state represents a valid completion."""
        # A simple heuristic: if all possible next rules are RuleEnd, the grammar is complete.
        for rule in self.parse_state.rules():
            if not isinstance(rule, gbnf.RuleEnd):
                return False
        return True
