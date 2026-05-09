from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.prompts import ChatPromptTemplate

LOG = logging.getLogger(__name__)


def create_qa_chain(llm: BaseChatModel, prompt: ChatPromptTemplate) -> Runnable:
    """Return an LCEL chain that answers a question from retrieved context.

    Expects input keys: {"context": str, "question": str}.

    Usage:
        chain = create_qa_chain(llm, prompt)
        answer = chain.invoke({"context": "...", "question": "..."})
    """
    LOG.debug("Creating QA chain")
    return prompt | llm | StrOutputParser()
