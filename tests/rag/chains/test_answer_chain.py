import pytest
from unittest.mock import MagicMock

from src.rag.chains.answer_chain import generate_answer


def _make_retriever(chunks: list[str]) -> MagicMock:
    retriever = MagicMock()
    retriever.retrieve.return_value = chunks
    return retriever


def _make_qa_chain(answer: str) -> MagicMock:
    chain = MagicMock()
    chain.invoke.return_value = answer
    return chain


class TestGenerateAnswer:
    def test_empty_question_raises(self):
        with pytest.raises(ValueError, match="question cannot be empty"):
            generate_answer("", _make_retriever([]), _make_qa_chain(""))

    def test_whitespace_question_raises(self):
        with pytest.raises(ValueError, match="question cannot be empty"):
            generate_answer("   ", _make_retriever([]), _make_qa_chain(""))

    def test_returns_fallback_when_no_chunks(self):
        result = generate_answer(
            "What is this?", _make_retriever([]), _make_qa_chain("")
        )
        assert "could not find" in result.lower()

    def test_passes_question_to_retriever(self):
        retriever = _make_retriever(["some chunk"])
        qa_chain = _make_qa_chain("answer")
        generate_answer("What is Python?", retriever, qa_chain)
        retriever.retrieve.assert_called_once_with("What is Python?")

    def test_chunks_joined_as_context(self):
        retriever = _make_retriever(["chunk one", "chunk two"])
        qa_chain = _make_qa_chain("answer")
        generate_answer("question?", retriever, qa_chain)
        _, kwargs = qa_chain.invoke.call_args
        payload = qa_chain.invoke.call_args[0][0]
        assert payload["context"] == "chunk one\n\nchunk two"

    def test_question_passed_to_qa_chain(self):
        retriever = _make_retriever(["some chunk"])
        qa_chain = _make_qa_chain("answer")
        generate_answer("What is Python?", retriever, qa_chain)
        payload = qa_chain.invoke.call_args[0][0]
        assert payload["question"] == "What is Python?"

    def test_returns_answer_from_chain(self):
        retriever = _make_retriever(["relevant chunk"])
        qa_chain = _make_qa_chain("Python is a programming language.")
        result = generate_answer("What is Python?", retriever, qa_chain)
        assert result == "Python is a programming language."

    def test_single_chunk_no_separator(self):
        retriever = _make_retriever(["only chunk"])
        qa_chain = _make_qa_chain("answer")
        generate_answer("question?", retriever, qa_chain)
        payload = qa_chain.invoke.call_args[0][0]
        assert payload["context"] == "only chunk"

    def test_qa_chain_not_called_when_no_chunks(self):
        retriever = _make_retriever([])
        qa_chain = _make_qa_chain("answer")
        generate_answer("question?", retriever, qa_chain)
        qa_chain.invoke.assert_not_called()

    def test_qa_chain_failure_returns_friendly_message(self):
        retriever = _make_retriever(["some context"])
        qa_chain = MagicMock()
        qa_chain.invoke.side_effect = RuntimeError("API failure")
        result = generate_answer("What is Python?", retriever, qa_chain)
        assert "error" in result.lower()
        assert "try again" in result.lower()
