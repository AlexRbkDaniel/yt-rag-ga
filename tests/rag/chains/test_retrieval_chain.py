from unittest.mock import MagicMock

from langchain_core.runnables import Runnable

from src.rag.chains.retrieval_chain import create_retrieval_chain


class TestCreateRetrievalChain:
    def test_returns_runnable(self):
        retriever = MagicMock()
        chain = create_retrieval_chain(retriever)
        assert isinstance(chain, Runnable)

    def test_invoke_calls_retriever_retrieve(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = ["chunk one", "chunk two"]
        chain = create_retrieval_chain(retriever)

        result = chain.invoke("What is the video about?")

        retriever.retrieve.assert_called_once_with("What is the video about?")
        assert result == ["chunk one", "chunk two"]

    def test_returns_list_of_strings(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = ["chunk one", "chunk two", "chunk three"]
        chain = create_retrieval_chain(retriever)

        result = chain.invoke("some query")

        assert isinstance(result, list)
        assert all(isinstance(c, str) for c in result)

    def test_empty_results_propagated(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = []
        chain = create_retrieval_chain(retriever)

        result = chain.invoke("some query")

        assert result == []
