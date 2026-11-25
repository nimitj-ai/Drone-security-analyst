# Drone Security Analyst Agent 🚁

> **FlytBase AI Engineer Assignment** — Prototype for autonomous drone-based property patrol and intelligent threat detection.

## Overview

An end-to-end AI pipeline that:
1. **Ingests** patrol video frames + synchronized drone telemetry
2. **Understands** each frame using a Vision-Language Model (VLM)
3. **Indexes** frames with CLIP multimodal embeddings in a vector database (cross-domain retrieval)
4. **Analyses** for security threats using a LangChain agent + rules engine
5. **Alerts** operators in real-time with structured incident reports
6. **Summarises** the full patrol mission and answers conversational Q&A

---

## Architecture

```
Video Frames + Telemetry
        │
        ▼
  Frame-Telemetry Synchronizer
        │
   ┌────┴────┐
   │         │
VLM Captioner   CLIP Embedder
   │         │
   └────┬────┘
        │
  LanceDB / ChromaDB (Hybrid Vector + Metadata Store)
        │
  LangChain Security Analyst Agent
        ├── Context Manager (loitering, revisit tracking)
        ├── Rule Engine (curfew / breach / abandoned object)
        ├── Tool: search_frame_archive
        ├── Tool: get_loiter_history
        ├── Tool: get_vehicle_visits
        └── Tool: generate_alert
        │
  ┌─────┴──────┐
  │            │
Alerts     Patrol Summarizer + Q&A Chat
  │
Streamlit Dashboard
```

---

## Quickstart

### 1. Install dependencies

```bash
cd drone-security-analyst
pip install -r requirements.txt
```

### 2. Run with mock backends (no GPU / API key required)

```bash
# Generate synthetic frames + telemetry, run full pipeline
python -m src.pipeline \
  --frames  data/sample_frames \
  --telem   data/telemetry_stream.json \
  --out     data/session_output.json
```

### 3. Launch the Streamlit dashboard

```bash
streamlit run src/app.py
```

Then open http://localhost:8501 in your browser.

---

## Configuration

### Environment variables

| Variable | Options | Default | Description |
|---|---|---|---|
| `VLM_BACKEND` | `mock`, `blip2`, `florence2` | `mock` | Vision-language model backend |
| `EMBED_BACKEND` | `mock`, `clip` | `mock` | Embedding backend |
| `VECTOR_BACKEND` | `mock`, `lancedb` | `mock` | Vector store backend |
| `LANGCHAIN_LLM` | `mock`, `openai` | `mock` | LLM for agent reasoning |
| `OPENAI_API_KEY` | — | — | Required if `LANGCHAIN_LLM=openai` |

### Use real backends (GPU recommended for VLM)

```bash
export VLM_BACKEND=florence2
export EMBED_BACKEND=clip
export VECTOR_BACKEND=lancedb
export LANGCHAIN_LLM=openai
export OPENAI_API_KEY=sk-...
python -m src.pipeline --frames data/sample_frames
```

---

## Project Structure

```
drone-security-analyst/
├── config/
│   └── rules_config.yaml          # Declarative alert rules
├── data/
│   ├── sample_frames/             # Generated patrol frames
│   └── telemetry_stream.json      # Drone telemetry
├── src/
│   ├── ingestion/
│   │   ├── video_loader.py        # Frame extractor from video/dir
│   │   ├── telemetry_sync.py      # Telemetry generator & sync
│   │   └── synthetic_frames.py    # Synthetic test dataset generator
│   ├── perception/
│   │   ├── vlm_captioner.py       # VLM backends (BLIP-2, Florence-2, mock)
│   │   └── embedder.py            # CLIP multimodal embedder
│   ├── indexing/
│   │   └── vector_store.py        # LanceDB / mock vector store
│   ├── agent/
│   │   ├── context_manager.py     # Loitering & reconnaissance tracker
│   │   ├── security_rules.py      # Deterministic rule engine
│   │   ├── langchain_agent.py     # LangChain agent + 4 security tools
│   │   └── summarizer.py          # Patrol summarizer & Q&A
│   ├── pipeline.py                # End-to-end orchestration script
│   └── app.py                     # Streamlit dashboard
├── tests/
│   ├── test_indexing.py
│   ├── test_rules.py
│   └── test_agent_qa.py
└── requirements.txt
```

---

## Running Tests

```bash
pytest tests/ -v
```

Expected output: all tests pass in mock mode without any GPU or API key.

---

## AI-Assisted Development Log

| Tool | Usage |
|---|---|
| **Antigravity (Gemini / Claude)** | Architecture design, full codebase generation, test suite scaffolding |
| **GitHub Copilot** | Inline completions for LangChain tool definitions |

All AI-generated code was reviewed, understood, and validated by the developer.

---

## Security Rules Reference

| Rule | Trigger Condition | Alert Level |
|---|---|---|
| **Curfew Loitering** | Person in restricted zone, 22:00–06:00, duration > 60s | CRITICAL |
| **Reconnaissance** | Same vehicle seen ≥ 2 times in one patrol | HIGH |
| **Perimeter Breach** | Climbing/jumping keywords + drone altitude < 30m | CRITICAL |
| **Abandoned Object** | Bag/backpack detected with no person in frame | HIGH |

Rules are fully configurable via `config/rules_config.yaml`.

---

## Extending the System

- **Add a new rule:** Add a section to `rules_config.yaml` and a matching evaluator method in `security_rules.py`.
- **Add a new VLM:** Implement a subclass of `BaseCaptioner` in `vlm_captioner.py` and register it in the factory.
- **Swap LLM:** Set `LANGCHAIN_LLM=openai` or extend `langchain_agent.py` to support Ollama / local LLMs.
- **Scale up:** Replace `MockVectorStore` with `LanceDBStore` and back the `PatrolContextManager` with Redis for multi-drone sessions.
