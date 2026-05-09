import logging

import gradio as gr
from dotenv import load_dotenv

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

load_dotenv()

LOG = logging.getLogger(__name__)

_llm_loader = LLMLoader()


def _format_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"[{minutes:02d}:{secs:02d}]"


def _format_transcript(segments) -> str:
    return "\n".join(f"{_format_timestamp(s.start)} {s.text}" for s in segments)


def _format_metadata(meta) -> str:
    lines = []

    if meta.title:
        lines.append(f"**Title:** {meta.title}")
    if meta.author:
        author_str = (
            f"[{meta.author}]({meta.author_url})" if meta.author_url else meta.author
        )
        lines.append(f"**Author:** {author_str}")

    lines.append("")
    lines.append("*The following fields require the YouTube Data API v3:*")
    lines.append(f"**Views:** {meta.views if meta.views else '—'}")
    lines.append(f"**Likes:** {meta.likes if meta.likes else '—'}")
    lines.append(f"**Published:** {meta.published_at or '—'}")
    lines.append(f"**Duration:** {meta.duration or '—'}")

    if meta.description:
        lines.append(f"\n**Description:** {meta.description}")

    if meta.tags:
        lines.append(f"**Tags:** {', '.join(meta.tags)}")

    return "\n\n".join(lines)


def on_load(url: str):
    empty = (
        gr.update(visible=False),  # video_details_row
        "",  # thumbnail_img
        "",  # metadata_md
        gr.update(value="", visible=False),  # transcript_box
        gr.update(value="", visible=False),  # summary_output
        gr.update(visible=False),  # summarize_btn
        gr.update(value=[], visible=False),  # chatbot
        gr.update(visible=False),  # qa_input_row
        None,  # video_state
        None,  # retriever_state
        [],  # chat_history_state
        gr.update(value="", visible=False),  # error_md
    )

    if not url.strip():
        return empty

    try:
        video_data = YtLoader.load_video_data(url)
    except ValueError as e:
        LOG.warning("Failed to load video: %s", e)
        return (*empty[:-1], gr.update(value=f"**Error:** {e}", visible=True))
    except Exception as e:
        LOG.error("Unexpected error loading video: %s", e)
        return (
            *empty[:-1],
            gr.update(
                value="**Error:** An unexpected error occurred. Please try again.",
                visible=True,
            ),
        )

    transcript_text = _format_transcript(video_data.transcript)
    thumbnail_url = video_data.metadata.thumbnail_url or ""
    metadata_text = _format_metadata(video_data.metadata)

    try:
        chunks = TranscriptChunker().chunk(video_data.transcript)
        store = Embedder().embed(chunks)
        retriever = Retriever(store)
        LOG.info("FAISS index built: %d chunks", len(chunks))
    except Exception as e:
        LOG.warning("Could not build FAISS index: %s", e)
        retriever = None

    return (
        gr.update(visible=True),  # video_details_row
        thumbnail_url,  # thumbnail_img
        metadata_text,  # metadata_md
        gr.update(value=transcript_text, visible=True),  # transcript_box
        gr.update(value="", visible=False),  # summary_output (hidden until clicked)
        gr.update(visible=True),  # summarize_btn
        gr.update(value=[], visible=True),  # chatbot
        gr.update(visible=True),  # qa_input_row
        video_data,  # video_state
        retriever,  # retriever_state
        [],  # chat_history_state
        gr.update(value="", visible=False),  # error_md (clear on success)
    )


def on_summarize(video_data):
    if video_data is None or not video_data.transcript:
        return gr.update(
            value="No transcript loaded. Please load a video first.", visible=True
        )
    transcript_text = " ".join(s.text for s in video_data.transcript)
    try:
        chain = create_summary_chain(_llm_loader.load(), create_summary_prompt())
        summary = chain.invoke({"transcript": transcript_text})
    except Exception as e:
        LOG.error("Summarization failed: %s", e)
        return gr.update(
            value="An error occurred while generating the summary. Please try again.",
            visible=True,
        )
    return gr.update(value=summary, visible=True)


def on_ask(question: str, retriever, history: list):
    if not question.strip():
        return history, history, ""

    if retriever is None:
        updated = history + [
            {"role": "user", "content": question},
            {
                "role": "assistant",
                "content": "The video index is not available. Please reload the video.",
            },
        ]
        return updated, updated, ""

    try:
        qa_chain = create_qa_chain(_llm_loader.load(), create_qa_prompt())
        answer = generate_answer(question, retriever, qa_chain)
    except Exception as e:
        LOG.error("Q&A failed: %s", e)
        answer = "An error occurred while generating the answer. Please try again."
    updated = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    return updated, updated, ""


with gr.Blocks(title="YouTube RAG Q&A") as app:
    video_state = gr.State(None)
    retriever_state = gr.State(None)
    chat_history_state = gr.State([])

    gr.Markdown("# YouTube RAG Q&A")

    # ── Outer layout: main content (left) + Q&A sidebar (right) ──────────────
    with gr.Row(equal_height=False):
        # ── LEFT: all main content ────────────────────────────────────────────
        with gr.Column(scale=7):
            # Row 1 — URL input
            with gr.Row():
                url_input = gr.Textbox(
                    label="YouTube URL",
                    placeholder="https://www.youtube.com/watch?v=...",
                    scale=6,
                    show_label=False,
                )
                load_btn = gr.Button("Continue", variant="primary", scale=1)

            error_md = gr.Markdown("", visible=False)

            # Row 2 — Video details (hidden until loaded)
            with gr.Row(visible=False) as video_details_row:
                with gr.Column(scale=2, min_width=200):
                    thumbnail_img = gr.Image(
                        label="",
                        height=180,
                        show_label=False,
                    )
                with gr.Column(scale=5):
                    metadata_md = gr.Markdown("")

            # Row 3 — Transcript
            transcript_box = gr.Textbox(
                label="Transcript",
                lines=14,
                max_lines=20,
                interactive=False,
                visible=False,
            )

            # Row 4 — Summary (above button) + Summarize button
            summary_output = gr.Textbox(
                label="Summary",
                lines=6,
                max_lines=12,
                interactive=False,
                visible=False,
            )
            summarize_btn = gr.Button("Summarize", variant="secondary", visible=False)

        # ── RIGHT: Q&A sidebar ────────────────────────────────────────────────
        with gr.Column(scale=3, min_width=320):
            gr.Markdown("### Ask the video")
            chatbot = gr.Chatbot(
                label="",
                visible=False,
                height=600,
                show_label=False,
            )
            with gr.Row(visible=False) as qa_input_row:
                question_input = gr.Textbox(
                    label="",
                    placeholder="Ask a question...",
                    scale=5,
                    show_label=False,
                )
                ask_btn = gr.Button("Ask", variant="primary", scale=1)

    # ── Events ────────────────────────────────────────────────────────────────

    _load_outputs = [
        video_details_row,
        thumbnail_img,
        metadata_md,
        transcript_box,
        summary_output,
        summarize_btn,
        chatbot,
        qa_input_row,
        video_state,
        retriever_state,
        chat_history_state,
        error_md,
    ]

    load_btn.click(fn=on_load, inputs=[url_input], outputs=_load_outputs)
    url_input.submit(fn=on_load, inputs=[url_input], outputs=_load_outputs)

    summarize_btn.click(
        fn=on_summarize,
        inputs=[video_state],
        outputs=[summary_output],
    )

    ask_btn.click(
        fn=on_ask,
        inputs=[question_input, retriever_state, chat_history_state],
        outputs=[chatbot, chat_history_state, question_input],
    )

    question_input.submit(
        fn=on_ask,
        inputs=[question_input, retriever_state, chat_history_state],
        outputs=[chatbot, chat_history_state, question_input],
    )


if __name__ == "__main__":
    app.launch()
