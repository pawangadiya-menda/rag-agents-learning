# 🤖 RAG & AI Agents Learning — PM Lens Series

> **7-day intensive learning sprint** covering RAG pipelines, AI agent architecture, and LLM foundations — built by a Senior PM with 15 years of enterprise IT experience, pairing every technical project with product thinking, business decisions, and PM frameworks.

---

## 👤 About

**Pawan Kumar Gadiya** | Senior Product Manager @ Accenture  
MBA — ISB 2023 | AI/ML Program — IIIT Hyderabad  
15 years in enterprise IT: ERP platforms, B2B SaaS, AI workflow automation

This repo documents my hands-on AI/ML learning journey — not just as an engineer learning to code, but as a **Technical PM** asking:
- Why this architecture over alternatives?
- What are the cost, latency, and accuracy tradeoffs?
- How does this translate into product decisions?

---

## 📁 Repo Structure

```
rag-agents-learning/
│
├── rag-pipeline/
│   ├── RAG_Pipeline_Complete.ipynb     ← End-to-end RAG: load → chunk → embed → store → retrieve → generate
│   └── pm-lens.md                      ← Product brief: decisions, tradeoffs, business framing
│
├── ai-agents/                          ← Coming soon
│   └── ...
│
├── llm-foundations/                    ← Coming soon
│   └── ...
│
└── README.md                           ← This file
```

---

## 🔗 Project 1 — RAG Pipeline (End to End)

### What it does
Builds a complete **Retrieval Augmented Generation (RAG)** pipeline that answers questions from private documents — without hallucinating.

A user asks a question → the system finds the most relevant chunks from your documents → feeds only those chunks to Claude → Claude answers based strictly on your data.

**Real-world use case:** Internal policy Q&A bot, contract search assistant, enterprise knowledge base, SOW document retrieval.

---

### 🛠️ Tech Stack

| Component | Tool | Why This Choice |
|---|---|---|
| **Document Loading** | LangChain `DirectoryLoader` | Loads real files from disk — scales to any folder |
| **Text Splitting** | `RecursiveCharacterTextSplitter` | Splits on paragraphs first — preserves meaning |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` | Free, local, 384-dim — data never leaves machine |
| **Vector Store** | ChromaDB | Local, persisted to disk — no infra needed for POC |
| **Orchestration** | LangChain LCEL | Clean pipe syntax — easy to swap components |
| **LLM** | Anthropic Claude `claude-3-haiku` | Fast, accurate, cost-effective for factual Q&A |

---

### 📐 Architecture

```
docs/*.txt files
      ↓
DirectoryLoader + TextLoader
      ↓
RecursiveCharacterTextSplitter
chunk_size=500, overlap=50
      ↓
HuggingFace all-MiniLM-L6-v2
Text → 384-dimensional vector
      ↓
ChromaDB (./chroma_db/)
Stores: vector + text + metadata
      ↓
Retriever (cosine similarity, k=3)
User question → embed → find top-3 chunks
      ↓
Prompt Template
"Answer ONLY from context: {context}. Question: {question}"
      ↓
Claude claude-3-haiku (temperature=0)
      ↓
Grounded Answer — based only on your documents
```

---

### 🧠 PM Lens — Key Decisions

| Decision | Config | Reasoning |
|---|---|---|
| `chunk_size=500` | 500 chars per chunk | Larger = noisy retrieval. Smaller = loses context. 500 is sweet spot for policy docs. |
| `chunk_overlap=50` | 50 char overlap | Prevents context loss at chunk boundaries |
| HuggingFace embeddings | `all-MiniLM-L6-v2` | Free + local vs OpenAI: saves cost + data privacy for enterprise use |
| ChromaDB local | Persisted to disk | POC-ready without cloud infra. Migration path to Pinecone for production. |
| `k=3` retrieval | Top 3 chunks | Classic precision vs recall tradeoff. Tune up for broader queries. |
| `temperature=0` | Deterministic output | Policy Q&A needs consistent factual answers — not creative variation |
| Prompt guard | "ONLY from context" | Prevents hallucination — non-negotiable for enterprise deployment |

---

### ▶️ How to Run

**On Google Colab (recommended):**

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `RAG_Pipeline_Complete.ipynb`
3. Add your Anthropic API key:
   - Click 🔑 key icon in left sidebar
   - Add secret: `ANTHROPIC_API_KEY` = your key from [console.anthropic.com](https://console.anthropic.com)
4. Click **Runtime → Run all**

**Dependencies installed automatically by the notebook:**
```
langchain, langchain-community, langchain-anthropic,
chromadb, sentence-transformers, anthropic
```

---

### 💡 What I Learned (PM Takeaways)

**RAG is a product architecture decision, not just a technical one.**

- **Build vs Buy:** ChromaDB (free, local) vs Pinecone (managed, scalable) — same tradeoff as build vs buy in SaaS
- **Chunking strategy is a product feature:** Wrong chunk size = wrong answers = user trust erosion
- **Hallucination guard = product safety:** The prompt instruction "answer ONLY from context" is as important as any feature flag — it's what makes the product trustworthy in enterprise
- **k value = UX tradeoff:** Higher k = more thorough but slower and more expensive. This is a user experience decision, not just a technical parameter
- **Embedding model choice = data privacy policy:** Using HuggingFace means data stays local. Using OpenAI embeddings means data leaves your system. In regulated industries (BFSI, Healthcare), this is a compliance decision

---

## 🗺️ Learning Roadmap

| Day | Topic | Status |
|---|---|---|
| Day 1 | LLM Foundations — tokens, embeddings, attention | ✅ Done |
| Day 2 | RAG Pipeline — end to end | ✅ Done |
| Day 3 | Advanced RAG — reranking, hybrid search | 🔄 In Progress |
| Day 4 | AI Agent Architecture — tools, memory, planning | 📅 Upcoming |
| Day 5 | Multi-Agent Systems | 📅 Upcoming |
| Day 6 | Capstone Project | 📅 Upcoming |
| Day 7 | Documentation + Portfolio polish | 📅 Upcoming |

---

## 🎯 Target Roles

This portfolio is built to demonstrate **Technical PM** capabilities for:

- **Salesforce** — Einstein AI, Agentforce platform
- **SAP** — Business AI, Joule copilot
- **ServiceNow** — Now Assist, AI platform
- **Adobe** — Firefly, Sensei GenAI
- **Sprinklr** — AI-powered CXM platform
- **Wolters Kluwer** — AI-powered professional information

---

## 📬 Connect

- **LinkedIn:** [linkedin.com/in/pawangadiya](https://linkedin.com/in/pawangadiya)
- **GitHub:** [github.com/pawangadiya-menda](https://github.com/pawangadiya-menda)

---

*Built with curiosity, PM discipline, and a lot of Colab runtime restarts.*
