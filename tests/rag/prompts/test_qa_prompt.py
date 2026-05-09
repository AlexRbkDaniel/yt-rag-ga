from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from src.rag.prompts.qa_prompt import create_qa_prompt


class TestCreateQaPrompt:
    def test_returns_chat_prompt_template(self):
        prompt = create_qa_prompt()
        assert isinstance(prompt, ChatPromptTemplate)

    def test_has_context_and_question_input_variables(self):
        prompt = create_qa_prompt()
        assert "context" in prompt.input_variables
        assert "question" in prompt.input_variables

    def test_no_extra_input_variables(self):
        prompt = create_qa_prompt()
        assert set(prompt.input_variables) == {"context", "question"}

    def test_has_two_messages(self):
        prompt = create_qa_prompt()
        assert len(prompt.messages) == 2

    def test_first_message_is_system(self):
        prompt = create_qa_prompt()
        assert isinstance(prompt.messages[0], SystemMessagePromptTemplate)

    def test_second_message_is_human(self):
        prompt = create_qa_prompt()
        assert isinstance(prompt.messages[1], HumanMessagePromptTemplate)

    def test_context_placeholder_in_human_message(self):
        prompt = create_qa_prompt()
        human_template = prompt.messages[1].prompt.template
        assert "{context}" in human_template

    def test_question_placeholder_in_human_message(self):
        prompt = create_qa_prompt()
        human_template = prompt.messages[1].prompt.template
        assert "{question}" in human_template

    def test_formats_with_context_and_question(self):
        prompt = create_qa_prompt()
        messages = prompt.format_messages(
            context="The video is about Python.",
            question="What is the video about?",
        )
        human_content = messages[1].content
        assert "The video is about Python." in human_content
        assert "What is the video about?" in human_content

    def test_system_message_has_no_placeholders(self):
        prompt = create_qa_prompt()
        system_template = prompt.messages[0].prompt.template
        assert "{context}" not in system_template
        assert "{question}" not in system_template
