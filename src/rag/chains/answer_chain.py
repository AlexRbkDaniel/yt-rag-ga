from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable
    from src.rag.retrieval.retriever import Retriever

LOG = logging.getLogger(__name__)

_CONTEXT_SEPARATOR = "\n\n"


def generate_answer(question: str, retriever: Retriever, qa_chain: Runnable) -> str:
    """Retrieve relevant context and generate an answer to the user's question.

    Args:
        question: The user's question.
        retriever: Retriever wrapping the FAISS store — replaces faiss_index from the lab.
        qa_chain: LCEL Runnable produced by create_qa_chain — replaces LLMChain.predict.

    Returns:
        The generated answer as a plain string.
    """
    if not question.strip():
        raise ValueError("question cannot be empty")

    LOG.debug("Retrieving context for question: %s", question)
    chunks = retriever.retrieve(question)

    if not chunks:
        LOG.warning("No relevant context found for question: %s", question)
        return "I could not find relevant information in the video transcript to answer your question."

    context = _CONTEXT_SEPARATOR.join(chunks)
    LOG.debug(
        "Retrieved %d chunks, context length: %d chars", len(chunks), len(context)
    )

    try:
        answer = qa_chain.invoke({"context": context, "question": question})
    except Exception as e:
        LOG.error("QA chain failed for question '%s': %s", question, e)
        return "An error occurred while generating the answer. Please try again."

    LOG.info("Answer generated for question: %s", question)
    return answer
