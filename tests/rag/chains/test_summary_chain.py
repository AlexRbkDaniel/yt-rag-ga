from unittest.mock import MagicMock

from langchain_core.runnables import Runnable

from src.rag.chains.summary_chain import create_summary_chain


class TestCreateSummaryChain:
    def test_returns_runnable(self):
        llm = MagicMock()
        prompt = MagicMock()
        prompt.__or__ = lambda self, other: MagicMock(spec=Runnable)
        chain = create_summary_chain(llm, prompt)
        assert chain is not None

    def test_chain_is_prompt_piped_to_llm(self):
        llm = MagicMock()
        prompt = MagicMock()
        piped = MagicMock()
        prompt.__or__ = MagicMock(return_value=piped)
        piped.__or__ = MagicMock(return_value=MagicMock())

        create_summary_chain(llm, prompt)

        prompt.__or__.assert_called_once_with(llm)

    def test_chain_pipes_str_output_parser(self):
        from langchain_core.output_parsers import StrOutputParser

        llm = MagicMock()
        prompt = MagicMock()
        piped = MagicMock()
        prompt.__or__ = MagicMock(return_value=piped)
        piped.__or__ = MagicMock(return_value=MagicMock())

        create_summary_chain(llm, prompt)

        args, _ = piped.__or__.call_args
        assert isinstance(args[0], StrOutputParser)

    def test_invoke_returns_string(self):
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import AIMessage

        prompt = ChatPromptTemplate.from_messages(
            [("system", "You are a helpful assistant."), ("human", "{transcript}")]
        )
        mock_llm = MagicMock(spec=ChatAnthropic)
        mock_llm.invoke = MagicMock(
            return_value=AIMessage(content="This is a summary.")
        )

        chain = create_summary_chain(mock_llm, prompt)
        result = chain.invoke({"transcript": "Some transcript text."})

        assert isinstance(result, str)
        assert result == "This is a summary."
