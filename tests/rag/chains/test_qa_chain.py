from unittest.mock import MagicMock

from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.rag.chains.qa_chain import create_qa_chain


class TestCreateQaChain:
    def test_returns_runnable(self):
        chain = create_qa_chain(MagicMock(), MagicMock())
        assert chain is not None

    def test_chain_pipes_prompt_to_llm(self):
        llm = MagicMock()
        prompt = MagicMock()
        piped = MagicMock()
        prompt.__or__ = MagicMock(return_value=piped)
        piped.__or__ = MagicMock(return_value=MagicMock())

        create_qa_chain(llm, prompt)

        prompt.__or__.assert_called_once_with(llm)

    def test_chain_pipes_str_output_parser(self):
        llm = MagicMock()
        prompt = MagicMock()
        piped = MagicMock()
        prompt.__or__ = MagicMock(return_value=piped)
        piped.__or__ = MagicMock(return_value=MagicMock())

        create_qa_chain(llm, prompt)

        args, _ = piped.__or__.call_args
        assert isinstance(args[0], StrOutputParser)

    def test_invoke_returns_string(self):
        from langchain_anthropic import ChatAnthropic

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant."),
                ("human", "{context}\n{question}"),
            ]
        )
        mock_llm = MagicMock(spec=ChatAnthropic)
        mock_llm.invoke = MagicMock(return_value=AIMessage(content="42 is the answer."))

        chain = create_qa_chain(mock_llm, prompt)
        result = chain.invoke(
            {"context": "Some context.", "question": "What is the answer?"}
        )

        assert isinstance(result, str)
        assert result == "42 is the answer."
