import logging
import os

from langchain_anthropic import ChatAnthropic
from pydantic import SecretStr

from src.rag.models.claude_model import ClaudeModel

LOG = logging.getLogger(__name__)


class LLMLoader:
    """Initialises and returns a configured ChatAnthropic LLM instance."""

    def __init__(
        self,
        model: str = ClaudeModel.SONNET.value,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout: int = 30,
        streaming: bool = True,
        max_retries: int = 2,
    ):
        """Configures and validates the LLM loader parameters at construction time.

        Args:
            model: Anthropic model ID — must be one of the supported ClaudeModel values.
            temperature: Sampling temperature — keep low (0.1) for factual RAG answers.
            max_tokens: Maximum tokens in the generated response.
            timeout: Request timeout in seconds.
            streaming: Whether to stream tokens as they are generated.
            max_retries: Number of retries on transient API failures.
        """
        if model not in ClaudeModel.values():
            raise ValueError(f"model must be one of {ClaudeModel.values()}, got '{model}'")
        if not 0.0 <= temperature <= 1.0:
            raise ValueError(f"temperature must be between 0.0 and 1.0, got {temperature}")
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be a positive integer, got {max_tokens}")
        if timeout <= 0:
            raise ValueError(f"timeout must be a positive integer, got {timeout}")
        if max_retries < 0:
            raise ValueError(f"max_retries must be non-negative, got {max_retries}")

        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._streaming = streaming
        self._max_retries = max_retries
        self._llm: ChatAnthropic | None = None

        LOG.debug(
            "LLMLoader configured: model=%s, temperature=%s, max_tokens=%d, timeout=%d, streaming=%s, max_retries=%d",
            model, temperature, max_tokens, timeout, streaming, max_retries,
        )

    def load(self) -> ChatAnthropic:
        """Validates the API key and returns a ready-to-use ChatAnthropic instance.

        The instance is cached after the first call — subsequent calls return the
        same object without reinitialising.
        """
        if self._llm is not None:
            LOG.debug("Returning cached LLM instance: model=%s", self._model)
            return self._llm

        LOG.debug("Loading LLM: model=%s", self._model)

        # Read the API key at load time so it reflects the current environment
        api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            LOG.error("ANTHROPIC_API_KEY is not set — cannot initialise the LLM")
            raise ValueError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")

        llm = ChatAnthropic(
            model_name=self._model,
            temperature=self._temperature,
            max_tokens_to_sample=self._max_tokens,
            timeout=self._timeout,
            streaming=self._streaming,
            max_retries=self._max_retries,
            api_key=SecretStr(api_key),
            stop=None,
        )

        LOG.info("LLM loaded successfully: model=%s", self._model)
        self._llm = llm
        return llm