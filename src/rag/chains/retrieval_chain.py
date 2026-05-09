from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.runnables import Runnable, RunnableLambda

if TYPE_CHECKING:
    from src.rag.retrieval.retriever import Retriever

LOG = logging.getLogger(__name__)


def create_retrieval_chain(retriever: Retriever) -> Runnable:
    """Return an LCEL Runnable that retrieves relevant chunks for a query.

    The lab equivalent is faiss_index.similarity_search(query, k=k).
    Here the FAISS store is encapsulated inside Retriever, which also applies
    score-threshold filtering and returns plain strings instead of Document objects.

    Usage:
        chain = create_retrieval_chain(retriever)
        chunks = chain.invoke("What is the video about?")
    """
    LOG.debug("Creating retrieval chain")
    return RunnableLambda(retriever.retrieve)
