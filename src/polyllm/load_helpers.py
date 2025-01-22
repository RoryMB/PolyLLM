from pathlib import Path


def load_llama(model_path: str|Path, n_ctx: int = 8192):
    from llama_cpp import Llama

    model = Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_gpu_layers=-1,
        verbose=False,
    )

    return model

def load_mlx(model_path: Path, auto_download: bool = False):
    import mlx_lm
    from huggingface_hub import snapshot_download

    if not Path(model_path).exists():
        try:
            model_path = snapshot_download(
                model_path,
                local_files_only=True,
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.py",
                    "tokenizer.model",
                    "*.tiktoken",
                    "*.txt",
                ],
            )
        except Exception:
            if not auto_download:
                raise RuntimeError(f'PolyLLM could not find {model_path} locally. Try downloading it or setting `load_mlx(..., auto_download=True)`')

            model_path = snapshot_download(
                model_path,
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.py",
                    "tokenizer.model",
                    "*.tiktoken",
                    "*.txt",
                ],
            )

    model, tokenizer = mlx_lm.load(model_path)

    return (model, tokenizer)
