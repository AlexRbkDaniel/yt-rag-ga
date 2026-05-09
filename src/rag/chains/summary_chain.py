from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.prompts import ChatPromptTemplate

LOG = logging.getLogger(__name__)


def create_summary_chain(llm: BaseChatModel, prompt: ChatPromptTemplate) -> Runnable:
    """Return an LCEL chain that summarizes a transcript.

    The chain pipes the prompt into the LLM and parses the response to a
    plain string: prompt | llm | StrOutputParser.

    Usage:
        chain = create_summary_chain(llm, prompt)
        summary = chain.invoke({"transcript": "..."})
    """
    LOG.debug("Creating summary chain")
    return prompt | llm | StrOutputParser()
