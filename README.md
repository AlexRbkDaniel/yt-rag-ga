# yt-rag-qa

AI-powered YouTube video summarizer and QA tool built with LangChain, FAISS, and Streamlit — answers questions from video transcripts using RAG.

## Overview

This project is a Retrieval-Augmented Generation (RAG) application that extracts a transcript from a YouTube video, processes the text into searchable chunks, stores embeddings in a FAISS vector database, and lets users ask questions through a Streamlit interface.

## Features

- Extracts transcript data from YouTube videos
- Summarizes video content in a concise, readable form
- Answers user questions based on the video transcript
- Uses LangChain to orchestrate the RAG pipeline
- Uses FAISS for efficient vector similarity search
- Provides a simple Streamlit web interface

## Tech Stack

- **Python**
- **LangChain**
- **FAISS**
- **Streamlit**
- Embedding model for transcript vectorization
- LLM for summarization and question answering

## How It Works

1. User provides a YouTube video URL
2. The app loads the video transcript
3. Transcript is split into smaller text chunks
4. Each chunk is converted into embeddings
5. Embeddings are stored in a FAISS index
6. User asks a question → retriever finds relevant chunks
7. LLM generates an answer using the retrieved context

## Installation

```bash
git clone https://github.com/your-username/yt-rag-qa.git
cd yt-rag-qa
python -m venv venv
source venv/bin/activate
pip install -r requirements
