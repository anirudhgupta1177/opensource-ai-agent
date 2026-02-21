# Open Source Web Scraper Agent

A workflow where third-party apps call an API with a prompt; the agent uses **free** open-source GPT-OSS (Ollama + Hugging Face fallback), free search (duckduckgo-search), and trafilatura-based crawling to research and return accurate, cited results.

**Only free services:** No SerpApi, no paid Brave tier, no OpenAI API, no paid Hugging Face tiers.

## Features

- **API**: `POST /research` with a prompt; returns cited content and sources.
- **AI**: GPT-OSS via Ollama (primary) and Hugging Face Inference Providers (fallback).
- **Search**: DuckDuckGo text search (no API key).
- **Crawl**: requests + trafilatura for main-text extraction.
- **Robustness**: Context-window limits, retries, timeouts, citation-only answers to reduce hallucination.

## Setup

### 1. Python environment

Requires Python 3.10+.

```bash
cd /path/to/Opensource\ AI\ Agent
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. LLM (choose one or both)

**Option A – Ollama (recommended, free, local)**

1. Install [Ollama](https://ollama.com).
2. Run and pull the model:

   ```bash
   ollama serve
   ollama pull gpt-oss:20b
   ```

**Option B – Hugging Face fallback (free tier)**

If Ollama is not running, the app will use Hugging Face Inference Providers (free tier).

1. Create a token at [Hugging Face Settings](https://huggingface.co/settings/tokens).
2. Copy `.env.example` to `.env` and set:

   ```bash
   HF_TOKEN=your_token_here
   ```

### 3. Run the API

From the project root (so `src` and `app` are on the path):

```bash
export PYTHONPATH=.   # or run from project root
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Or:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Docs: http://localhost:8000/docs

## API

### POST /research

**Request body:**

```json
{
  "prompt": "What are the main benefits of renewable energy? Answer in markdown.",
  "options": {
    "format": "markdown",
    "max_sources": 8
  }
}
```

- `prompt` (required): Research question or instruction (max 2000 chars).
- `options.format`: `"markdown"` | `"bullet_list"` | `"raw"`.
- `options.max_sources`: Number of URLs to fetch (1–15, default 8).

**Response:**

```json
{
  "content": "...",
  "sources": [
    { "url": "...", "title": "...", "snippet": "..." }
  ],
  "error": null
}
```

### GET /health

Returns `{"status": "ok"}`.

## Project layout

```
├── app/main.py           # FastAPI app, POST /research
├── src/
│   ├── llm/client.py     # Ollama + HF fallback
│   ├── search/duckduckgo_search.py
│   ├── crawl/fetcher.py  # requests + trafilatura
│   ├── context/budget.py # token budget, [Source: url]
│   └── agent/orchestrator.py
├── requirements.txt
├── .env.example
└── README.md
```

## Rate limits

- In-app: 10 requests per minute per client (slowapi).
- Hugging Face free tier has its own rate limits; Ollama has none.

## License

Use and modify as needed. Dependencies have their own licenses (MIT, Apache-2.0, etc.).
