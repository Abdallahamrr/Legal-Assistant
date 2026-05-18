# Legal AI Assistant — Agentic RAG
### Capstone Project | Gen AI Course 2026

> Build a production-like Legal AI system covering every course topic: Prompt Engineering, Multi-Provider Models, RAG, Agentic AI, Evaluation, and Model Selection.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API keys
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."       # optional — Gemini
export DEEPSEEK_API_KEY="..."     # optional — DeepSeek

# 3. Generate sample legal documents
python src/generate_sample_data.py

# 4. Run interactive agent
python src/rag_pipeline.py

# 5. Run evaluation suite
python src/rag_pipeline.py --eval

# 6. View model selection report
python src/model_selection.py
```

---

## Project Structure

```
legal_ai/
├── data/                          # Legal documents (PDFs or .txt)
│   ├── force_majeure_cases.txt
│   ├── california_employment_law.txt
│   ├── gdpr_data_protection.txt
│   ├── liquidated_damages_cases.txt
│   └── trade_secrets_whistleblower.txt
├── src/
│   ├── rag_pipeline.py            # MAIN: full agentic RAG + LangGraph
│   ├── generate_sample_data.py    # synthetic legal corpus generator
│   └── model_selection.py        # HELM / Artificial Analysis / LLM Arena report
├── outputs/
│   ├── eval_report.json           # evaluation results (all 10 scenarios)
│   └── model_selection_report.json
├── vector_store/                  # FAISS index (auto-created)
└── requirements.txt
```

---

## Architecture — Agentic RAG Pipeline

```
User Query
    ↓
[Phase 1] Document Ingestion
    PyMuPDF → RecursiveCharacterTextSplitter(800 tok, 100 overlap)
    → OpenAI text-embedding-3-large → FAISS vector store

[Phase 2] Prompt Engineering
    Legal system prompt: Issue → Law → Application → Conclusion
    Few-shot chain-of-thought for legal reasoning
    Defensive prompt: "research only, consult a licensed attorney"

[Phase 3] Classic RAG
    RetrievalQA chain: retriever(Top-K=4) + GPT-4o generation

[Phase 4] Agentic RAG (LangGraph)
    StateGraph nodes:
    retrieve_node → check_sufficiency (conditional) → generate_node → END
    If context < 300 chars AND iterations < 1: loop back to retrieve

[Phase 5] Evaluation
    Faithfulness (heuristic / RAGAS-compatible)
    Task Success Rate, Latency (ms), Cost (USD per query)

[Phase 6] Model Selection
    HELM LegalBench F1 scores
    Artificial Analysis quality/speed/cost
    LLM Arena Elo rankings
```

---

## Tech Stack

| Component | Primary | Alternative |
|---|---|---|
| LLM | GPT-4o | DeepSeek-V3 / Gemini 1.5 Pro |
| Embeddings | text-embedding-3-large | BAAI/bge-base-en-v1.5 |
| Vector Store | FAISS | Chroma / Pinecone |
| Agent Framework | LangGraph | LangChain ReAct |
| No-Code | N8N (docker) | Zapier |
| PDF Loading | PyMuPDF | PyPDFLoader |
| Eval | Heuristic + RAGAS | G-Eval |

---

## Evaluation Results (10 Scenarios)

| Metric | Result | Target |
|---|---|---|
| Avg Latency | 1,830 ms | < 2,000 ms |
| Avg Faithfulness | 0.87 | > 0.80 |
| Task Success Rate | 92% | > 85% |
| Cost per query | ~$0.044 | < $0.10 |

---

## Model Selection Decision

**Primary (Interactive Q&A):** GPT-4o — LegalBench F1=0.84, best instruction-following
**Long Context (Full contracts):** Gemini 1.5 Pro — 1M token window
**Cost-Efficient (Batch):** DeepSeek-V3 — $0.27/1M tokens, 20× cheaper
**EU Deployment:** Mistral Large — EU data residency

---

## Security & Ethics

- Sample documents only — NEVER upload real client files to cloud APIs
- System prompt enforces "research assistance only — consult a licensed attorney"
- API keys in `.env` file — add `.env` to `.gitignore`
- Prompt injection mitigation via input sanitization
- Hallucination risk: flagged when faithfulness < 0.70

---

## Agent Options

| Option | When to use |
|---|---|
| LangGraph (recommended) | Full project — stateful, multi-step reasoning |
| LangChain ReAct | Simpler setup, single-step retrieval |
| N8N no-code | Visual workflow, non-developer presentation |
| OpenAI Assistants API | Simplest production option with file search |

```bash
# Run N8N locally
docker run -it --rm --name n8n -p 5678:5678 n8nio/n8n
```
