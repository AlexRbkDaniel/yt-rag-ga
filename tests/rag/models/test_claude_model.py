import pytest

from src.rag.models.claude_model import ClaudeModel


class TestClaudeModel:
    def test_sonnet_value(self):
        assert ClaudeModel.SONNET.value == "claude-sonnet-4-6"

    def test_haiku_value(self):
        assert ClaudeModel.HAIKU.value == "claude-haiku-4-5-20251001"

    def test_values_returns_list_of_strings(self):
        vals = ClaudeModel.values()
        assert isinstance(vals, list)
        assert all(isinstance(v, str) for v in vals)

    def test_values_contains_sonnet(self):
        assert "claude-sonnet-4-6" in ClaudeModel.values()

    def test_values_contains_haiku(self):
        assert "claude-haiku-4-5-20251001" in ClaudeModel.values()

    def test_values_length_matches_members(self):
        assert len(ClaudeModel.values()) == len(list(ClaudeModel))

    def test_is_str_subclass(self):
        assert isinstance(ClaudeModel.SONNET, str)

    def test_string_comparison(self):
        assert ClaudeModel.SONNET == "claude-sonnet-4-6"

    def test_lookup_by_value(self):
        model = ClaudeModel("claude-sonnet-4-6")
        assert model is ClaudeModel.SONNET

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            ClaudeModel("claude-unknown-model")
