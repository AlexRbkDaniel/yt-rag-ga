import pytest
from unittest.mock import patch, MagicMock

from src.rag.loaders.llm_loader import LLMLoader
from src.rag.models.claude_model import ClaudeModel


class TestLLMLoaderInit:

    def test_valid_default_config(self):
        loader = LLMLoader()
        assert loader._model == ClaudeModel.SONNET.value
        assert loader._temperature == 0.1
        assert loader._max_tokens == 1024
        assert loader._timeout == 30
        assert loader._streaming is True
        assert loader._max_retries == 2

    def test_custom_config_haiku(self):
        loader = LLMLoader(model=ClaudeModel.HAIKU.value, temperature=0.5, max_tokens=512, timeout=60, streaming=False, max_retries=3)
        assert loader._model == ClaudeModel.HAIKU.value
        assert loader._temperature == 0.5
        assert loader._max_tokens == 512
        assert loader._timeout == 60
        assert loader._streaming is False
        assert loader._max_retries == 3

    def test_unsupported_model_raises(self):
        with pytest.raises(ValueError, match="model must be one of"):
            LLMLoader(model="claude-gpt-fake")

    def test_empty_model_raises(self):
        with pytest.raises(ValueError, match="model must be one of"):
            LLMLoader(model="")

    def test_whitespace_model_raises(self):
        with pytest.raises(ValueError, match="model must be one of"):
            LLMLoader(model="   ")

    def test_sonnet_model_accepted(self):
        loader = LLMLoader(model=ClaudeModel.SONNET.value)
        assert loader._model == ClaudeModel.SONNET.value

    def test_haiku_model_accepted(self):
        loader = LLMLoader(model=ClaudeModel.HAIKU.value)
        assert loader._model == ClaudeModel.HAIKU.value

    def test_temperature_below_zero_raises(self):
        with pytest.raises(ValueError, match="temperature must be between"):
            LLMLoader(temperature=-0.1)

    def test_temperature_above_one_raises(self):
        with pytest.raises(ValueError, match="temperature must be between"):
            LLMLoader(temperature=1.1)

    def test_temperature_boundary_zero_valid(self):
        loader = LLMLoader(temperature=0.0)
        assert loader._temperature == 0.0

    def test_temperature_boundary_one_valid(self):
        loader = LLMLoader(temperature=1.0)
        assert loader._temperature == 1.0

    def test_max_tokens_zero_raises(self):
        with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
            LLMLoader(max_tokens=0)

    def test_max_tokens_negative_raises(self):
        with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
            LLMLoader(max_tokens=-1)

    def test_timeout_zero_raises(self):
        with pytest.raises(ValueError, match="timeout must be a positive integer"):
            LLMLoader(timeout=0)

    def test_timeout_negative_raises(self):
        with pytest.raises(ValueError, match="timeout must be a positive integer"):
            LLMLoader(timeout=-5)

    def test_max_retries_negative_raises(self):
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            LLMLoader(max_retries=-1)

    def test_max_retries_zero_is_valid(self):
        loader = LLMLoader(max_retries=0)
        assert loader._max_retries == 0

    def test_llm_not_cached_at_init(self):
        loader = LLMLoader()
        assert loader._llm is None


class TestLLMLoaderLoad:

    def test_missing_api_key_raises(self):
        loader = LLMLoader()
        with patch("src.rag.loaders.llm_loader.os.getenv", return_value=None):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
                loader.load()

    def test_empty_api_key_raises(self):
        loader = LLMLoader()
        with patch("src.rag.loaders.llm_loader.os.getenv", return_value=""):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
                loader.load()

    def test_returns_chat_anthropic_instance(self):
        loader = LLMLoader()
        mock_llm = MagicMock()
        with patch("src.rag.loaders.llm_loader.os.getenv", return_value="test-key"), \
             patch("src.rag.loaders.llm_loader.ChatAnthropic", return_value=mock_llm):
            result = loader.load()
        assert result is mock_llm

    def test_instance_is_cached_after_first_load(self):
        loader = LLMLoader()
        mock_llm = MagicMock()
        with patch("src.rag.loaders.llm_loader.os.getenv", return_value="test-key"), \
             patch("src.rag.loaders.llm_loader.ChatAnthropic", return_value=mock_llm) as mock_cls:
            first = loader.load()
            second = loader.load()

        assert first is second
        mock_cls.assert_called_once()  # ChatAnthropic instantiated only once

    def test_cached_instance_set_after_load(self):
        loader = LLMLoader()
        mock_llm = MagicMock()
        with patch("src.rag.loaders.llm_loader.os.getenv", return_value="test-key"), \
             patch("src.rag.loaders.llm_loader.ChatAnthropic", return_value=mock_llm):
            loader.load()

        assert loader._llm is mock_llm