from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from src.rag.prompts.summary_prompt import create_summary_prompt


class TestCreateSummaryPrompt:
    def test_returns_chat_prompt_template(self):
        prompt = create_summary_prompt()
        assert isinstance(prompt, ChatPromptTemplate)

    def test_has_transcript_input_variable(self):
        prompt = create_summary_prompt()
        assert "transcript" in prompt.input_variables

    def test_no_extra_input_variables(self):
        prompt = create_summary_prompt()
        assert prompt.input_variables == ["transcript"]

    def test_has_two_messages(self):
        prompt = create_summary_prompt()
        assert len(prompt.messages) == 2

    def test_first_message_is_system(self):
        prompt = create_summary_prompt()
        assert isinstance(prompt.messages[0], SystemMessagePromptTemplate)

    def test_second_message_is_human(self):
        prompt = create_summary_prompt()
        assert isinstance(prompt.messages[1], HumanMessagePromptTemplate)

    def test_transcript_placeholder_in_human_message(self):
        prompt = create_summary_prompt()
        human_template = prompt.messages[1].prompt.template
        assert "{transcript}" in human_template

    def test_formats_with_transcript(self):
        prompt = create_summary_prompt()
        messages = prompt.format_messages(transcript="This is a test transcript.")
        human_content = messages[1].content
        assert "This is a test transcript." in human_content

    def test_system_message_does_not_contain_transcript_placeholder(self):
        prompt = create_summary_prompt()
        system_template = prompt.messages[0].prompt.template
        assert "{transcript}" not in system_template
