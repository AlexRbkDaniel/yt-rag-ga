from langchain_core.prompts import ChatPromptTemplate

_SYSTEM = """You are an AI assistant that summarizes YouTube video transcripts.

Instructions:
1. Summarize the transcript in a single concise paragraph.
2. Focus on the key points and main ideas of the spoken content.
3. Do not add information that is not present in the transcript."""

_HUMAN = """Please summarize the following YouTube video transcript:

{transcript}"""


def create_summary_prompt() -> ChatPromptTemplate:
    """Return a ChatPromptTemplate for summarizing a YouTube transcript."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM),
            ("human", _HUMAN),
        ]
    )
