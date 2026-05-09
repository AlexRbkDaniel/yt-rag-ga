from __future__ import annotations

import logging

from src.rag.chains.answer_chain import generate_answer
from src.rag.chains.qa_chain import create_qa_chain
from src.rag.chains.summary_chain import create_summary_chain
from src.rag.loaders.llm_loader import LLMLoader
from src.rag.loaders.youtube_loader import YtLoader
from src.rag.processing.chunker import TranscriptChunker
from src.rag.processing.embedder import Embedder
from src.rag.prompts.qa_prompt import create_qa_prompt
from src.rag.prompts.summary_prompt import create_summary_prompt
from src.rag.retrieval.retriever import Retriever

LOG = logging.getLogger(__name__)

_default_llm_loader = LLMLoader()


def summarize_video(video_url: str, llm_loader: LLMLoader | None = None) -> str:
    """Load a YouTube transcript and return a Claude-generated summary.

    Args:
        video_url: The YouTube video URL to summarize.
        llm_loader: Optional pre-configured LLMLoader. A default instance is
                    created when not provided — useful for injecting a custom
                    model or temperature in tests and the UI.

    Returns:
        A plain-string summary of the video transcript.
    """
    if not video_url.strip():
        return "Please provide a valid YouTube URL."

    LOG.info("Summarizing video: %s", video_url)

    try:
        video_data = YtLoader.load_video_data(video_url)
    except ValueError as e:
        LOG.warning("Could not load video transcript: %s", e)
        return f"Could not load video: {e}"
    except Exception as e:
        LOG.error("Unexpected error loading video: %s", e)
        return "An unexpected error occurred while loading the video. Please try again."

    if not video_data.transcript:
        return "No transcript available for this video."

    transcript_text = " ".join(segment.text for segment in video_data.transcript)
    LOG.debug("Transcript length: %d chars", len(transcript_text))

    loader = llm_loader or _default_llm_loader
    llm = loader.load()

    summary_chain = create_summary_chain(llm, create_summary_prompt())
    summary = summary_chain.invoke({"transcript": transcript_text})

    LOG.info("Summary generated for video: %s", video_url)
    return summary


def answer_question(
    video_url: str,
    user_question: str,
    llm_loader: LLMLoader | None = None,
) -> str:
    """Retrieve relevant context from the transcript and answer the user's question.

    Args:
        video_url: The YouTube video URL to load the transcript from.
        user_question: The question posed by the user.
        llm_loader: Optional pre-configured LLMLoader. A default instance is
                    created when not provided.

    Returns:
        A plain-string answer grounded in the video transcript.
    """
    if not video_url.strip():
        return "Please provide a valid YouTube URL."

    if not user_question.strip():
        return "Please provide a valid question."

    LOG.info("Answering question for video: %s", video_url)

    try:
        video_data = YtLoader.load_video_data(video_url)
    except ValueError as e:
        LOG.warning("Could not load video transcript: %s", e)
        return f"Could not load video: {e}"
    except Exception as e:
        LOG.error("Unexpected error loading video: %s", e)
        return "An unexpected error occurred while loading the video. Please try again."

    if not video_data.transcript:
        return "No transcript available for this video."

    # Step 1: Chunk the transcript for context retrieval
    chunks = TranscriptChunker().chunk(video_data.transcript)

    # Step 2: Embed chunks and build the FAISS store
    store = Embedder().embed(chunks)

    # Step 3: Wrap the store in a Retriever
    retriever = Retriever(store)

    # Step 4: Load the LLM and set up the Q&A chain
    loader = llm_loader or _default_llm_loader
    llm = loader.load()
    qa_chain = create_qa_chain(llm, create_qa_prompt())

    # Step 5: Retrieve context and generate the answer
    answer = generate_answer(user_question, retriever, qa_chain)

    LOG.info("Answer generated for question: %s", user_question)
    return answer
