from typing import Any, Dict, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from pydantic import Field

from .polyllm import generate


class LCPolyLLM(LLM):
    """LangChain interface to PolyLLM.

    Wraps PolyLLM's generate() function as a LangChain LLM.

    Args:
        model: The model name or instance to use
        temperature: Sampling temperature between 0 and 1

    Example:
        .. code-block:: python

            from polyllm import LCPolyLLM

            llm = LCPolyLLM(model="openai/gpt-3.5-turbo", temperature=0.7)
            result = llm.invoke("Tell me a joke")
    """

    model: str = Field(description="The model name")
    temperature: float = Field(default=0.0, description="Sampling temperature")

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Generate text from the model.

        Args:
            prompt: The prompt to generate from
            stop: A list of strings to stop generation when encountered
            run_manager: Optional callback manager
            **kwargs: Additional arguments to pass to the model

        Returns:
            The generated text
        """
        if stop is not None:
            raise RuntimeError("Stop sequences are not supported with PolyLLM")

        messages = [{"role": "user", "content": prompt}]
        return generate(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature
        }

    @property
    def _llm_type(self) -> str:
        return "polyllm"
