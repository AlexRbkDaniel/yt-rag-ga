from langchain_core.prompts import ChatPromptTemplate

_SYSTEM = """You are an expert assistant that answers questions strictly based on the provided video transcript context.

Instructions:
1. Answer only using information present in the context.
2. If the context does not contain enough information to answer, say so clearly.
3. Be concise and accurate."""

_HUMAN = """Relevant video context:
{context}

Question: {question}"""


def create_qa_prompt() -> ChatPromptTemplate:
    """Return a ChatPromptTemplate for answering questions based on video context."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM),
            ("human", _HUMAN),
        ]
    )
