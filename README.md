# yt-rag-qa

AI-powered YouTube Q&A — paste a video URL, ask questions, get answers grounded in the transcript.

Built with Anthropic Claude, Voyage AI embeddings, FAISS, and a Gradio UI.

## How It Works

1. Paste a YouTube URL — the transcript is extracted automatically
2. Transcript is split into overlapping chunks and embedded via Voyage AI (`voyage-3-lite`)
3. Chunks are stored in a FAISS in-memory vector index
4. Ask a question — the retriever finds the most relevant chunks by semantic similarity
5. Claude (`claude-sonnet-4-6`) generates a grounded answer from the retrieved context

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Anthropic Claude (Sonnet / Haiku) |
| Embeddings | Voyage AI `voyage-3-lite` |
| Vector store | FAISS (in-memory) |
| Orchestration | LangChain 1.x |
| Transcript | youtube-transcript-api |
| UI | Gradio |
| Python | 3.12 |

## Project Structure

```
yt-rag-qa/
├── src/
│   ├── rag/
│   │   ├── loaders/
│   │   │   ├── youtube_loader.py     # Extracts transcript + metadata from YouTube
│   │   │   └── llm_loader.py         # Initialises and caches the Claude LLM
│   │   ├── models/
│   │   │   ├── claude_model.py       # Enum of supported Claude model IDs
│   │   │   ├── transcript.py         # TranscriptSegment dataclass
│   │   │   └── video.py              # VideoMetadata and VideoData dataclasses
│   │   ├── processing/
│   │   │   ├── chunker.py            # Splits transcript into overlapping chunks
│   │   │   └── embedder.py           # Embeds chunks and builds the FAISS index
│   │   └── retrieval/
│   │       └── retriever.py          # Similarity search against the FAISS store
│   └── ui/                           # Gradio interface (in progress)
├── tests/                            # Mirrors src/ structure, pytest
├── docs/
│   └── youtube-data-api-setup.md    # Guide for YouTube Data API v3 integration
├── .github/
│   ├── workflows/
│   │   ├── build.yml                 # Install deps + lint on PR
│   │   ├── format.yml                # Ruff format check on PR
│   │   ├── test.yml                  # Pytest — triggered after build passes
│   │   └── audit.yml                 # pip-audit on PR + weekly schedule
│   └── dependabot.yml               # Weekly dependency update PRs
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
git clone https://github.com/your-username/yt-rag-qa.git
cd yt-rag-qa
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

### Running Tests

```bash
pytest
```

With coverage:

```bash
pytest --cov=src --cov-report=term-missing
```

### Linting and Formatting

```bash
# Lint
ruff check src tests

# Format
ruff format src tests

# Check format without applying changes
ruff format --check src tests
```

### Security Audit

```bash
pip-audit -r requirements.txt -r requirements-dev.txt
```

## CI / CD

Every pull request to `develop` runs four checks:

| Workflow | What it does |
|---|---|
| **Build** | Installs deps, runs ruff lint, verifies all imports |
| **Format** | Checks code style with `ruff format --check` |
| **Test** | Runs the full pytest suite (triggered only if Build passes) |
| **Security Audit** | Scans all pinned dependencies for known CVEs |

Dependabot opens weekly PRs for dependency updates, grouped by ecosystem (LangChain, Anthropic, Voyage AI).

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.