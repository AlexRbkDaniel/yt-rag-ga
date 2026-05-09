# yt-rag-ga

AI-powered YouTube Q&A — paste a video URL, ask questions, get answers grounded in the transcript.

Built with Anthropic Claude, Voyage AI embeddings, FAISS, and a Gradio UI.

## How It Works

1. Paste a YouTube URL — the transcript and metadata are fetched automatically
2. The transcript is split into overlapping chunks and embedded via Voyage AI (`voyage-3-lite`)
3. Chunks are stored in a FAISS in-memory vector index
4. Ask a question — the retriever finds the most relevant chunks by semantic similarity
5. Claude (`claude-sonnet-4-6`) generates a grounded answer from the retrieved context
6. A full video summary is also available on demand

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Anthropic Claude (Sonnet / Haiku) |
| Embeddings | Voyage AI `voyage-3-lite` |
| Vector store | FAISS (in-memory) |
| Orchestration | LangChain 1.x (LCEL) |
| Transcript | youtube-transcript-api |
| UI | Gradio |
| Python | 3.12 |

## Project Structure

```
yt-rag-ga/
├── src/
│   ├── rag/
│   │   ├── chains/
│   │   │   ├── answer_chain.py       # Retrieves context and generates an answer
│   │   │   ├── qa_chain.py           # LCEL chain: prompt | LLM | parser
│   │   │   ├── retrieval_chain.py    # Wraps Retriever as a composable Runnable
│   │   │   └── summary_chain.py      # LCEL chain for transcript summarisation
│   │   ├── loaders/
│   │   │   ├── llm_loader.py         # Initialises and caches the Claude LLM
│   │   │   └── youtube_loader.py     # Fetches transcript + metadata from YouTube
│   │   ├── models/
│   │   │   ├── claude_model.py       # Enum of supported Claude model IDs
│   │   │   ├── transcript.py         # TranscriptSegment dataclass
│   │   │   └── video.py              # VideoMetadata and VideoData dataclasses
│   │   ├── processing/
│   │   │   ├── chunker.py            # Splits transcript into overlapping chunks
│   │   │   └── embedder.py           # Embeds chunks and builds the FAISS index
│   │   ├── prompts/
│   │   │   ├── qa_prompt.py          # ChatPromptTemplate for Q&A
│   │   │   └── summary_prompt.py     # ChatPromptTemplate for summarisation
│   │   ├── retrieval/
│   │   │   └── retriever.py          # Similarity search against the FAISS store
│   │   └── pipeline.py               # High-level API: summarize_video, answer_question
│   └── ui/
│       └── app.py                    # Gradio interface
├── tests/                            # Mirrors src/ structure, pytest
│   ├── rag/
│   │   ├── chains/                   # Tests for all chain modules
│   │   ├── loaders/                  # Tests for YouTube and LLM loaders
│   │   ├── models/                   # Tests for dataclasses and enums
│   │   ├── processing/               # Tests for chunker and embedder
│   │   ├── prompts/                  # Tests for prompt templates
│   │   ├── retrieval/                # Tests for the retriever
│   │   └── test_pipeline.py          # Integration-style pipeline tests
│   └── ui/
│       └── test_app.py               # Tests for Gradio event handlers
├── docs/
│   └── youtube-data-api-setup.md    # Guide for YouTube Data API v3 integration
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                    # Install, lint, and run tests on PR
│   │   ├── format.yml                # Ruff format check on PR
│   │   └── audit.yml                 # pip-audit on PR + weekly schedule
│   ├── dependabot.yml                # Weekly dependency update PRs
│   └── CODEOWNERS                    # PR review assignments
├── .env.example                      # Required environment variables
├── requirements.txt                  # Pinned runtime dependencies
├── requirements-dev.txt              # Pinned dev dependencies (lint, test, audit)
└── pytest.ini
```

## Installation

### Prerequisites

- Python 3.12
- An [Anthropic API key](https://console.anthropic.com/)
- A [Voyage AI API key](https://www.voyageai.com/)

### Setup

**1. Clone the repository**

```bash
git clone https://github.com/your-username/yt-rag-ga.git
cd yt-rag-ga
```

**2. Create and activate a virtual environment**

```bash
python3.12 -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

For development (linting, tests, security audit):

```bash
pip install -r requirements-dev.txt
```

**4. Configure environment variables**

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```
ANTHROPIC_API_KEY=your-anthropic-api-key-here
VOYAGE_API_KEY=your-voyage-api-key-here
```

## Running the App

```bash
python -m src.ui.app
```

Then open the URL printed in the terminal (default: `http://127.0.0.1:7860`).

Paste a YouTube video URL and press **Continue** to load the transcript and metadata. Once loaded you can:

- Read the full transcript with timestamps
- Click **Summarize** to generate a video summary
- Type questions in the sidebar and press **Ask**

## Running Tests

```bash
pytest
```

With coverage:

```bash
pytest --cov=src --cov-report=term-missing
```

## Linting and Formatting

```bash
# Lint
ruff check src tests

# Format
ruff format src tests

# Check format without applying changes
ruff format --check src tests
```

## Security Audit

```bash
pip-audit -r requirements.txt -r requirements-dev.txt
```

## CI / CD

Every pull request to `develop` or `main` runs three checks:

| Workflow | What it does |
|---|---|
| **CI** | Installs deps, runs ruff lint, then runs the full pytest suite |
| **Format** | Checks code style with `ruff format --check` |
| **Security Audit** | Scans all pinned dependencies for known CVEs (also runs weekly) |

Dependabot opens weekly PRs for dependency updates, grouped by ecosystem (LangChain, Anthropic, Voyage AI).

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
