# 🧠 PM Lens — RAG Pipeline Product Brief

> This document pairs with `RAG_Pipeline_Complete.ipynb`.  
> It captures the **product thinking, business decisions, and PM frameworks** behind every technical choice in the RAG pipeline.  
> Written from the perspective of a Senior PM evaluating this as a product architecture.

---

## 1. Problem Statement

### The Business Problem
Enterprise teams sit on massive amounts of private knowledge — SOW documents, SLA contracts, policy files, Confluence pages, support tickets. When employees or customers ask questions, two things happen:

- **Without RAG:** The LLM answers from general training data → hallucinated, generic, legally risky answers
- **With RAG:** The LLM answers from YOUR specific documents → accurate, grounded, auditable answers

### The PM Framing
This is not a technical problem. It is a **trust problem.**

Enterprises will not deploy an AI assistant that makes things up. RAG is the architecture that makes LLM output trustworthy enough for enterprise use.

**Analogy:** RAG is to an LLM what a knowledge base is to a support agent. Without it, the agent guesses. With it, the agent looks it up.

---

## 2. User Stories

| Persona | Pain Point | RAG Solution |
|---|---|---|
| Delivery Manager | "Which SLA metrics did we commit to for this client?" | Ask the bot → retrieves from contract → exact answer |
| Sales Engineer | "What does our Enterprise plan include vs Growth?" | Ask the bot → retrieves from pricing doc → accurate comparison |
| Support Agent | "Customer says we promised 99.9% uptime — is that right?" | Ask the bot → retrieves SLA → confirms or corrects |
| Legal / Compliance | "Are we GDPR compliant? Where is data stored?" | Ask the bot → retrieves compliance doc → grounded answer |
| New Joiner | "What is the refund policy for annual plans?" | Ask the bot → retrieves policy → no need to hunt through Confluence |

---

## 3. Architecture Decisions — PM Rationale

### Decision 1: DirectoryLoader over hardcoded strings

**What:** Load documents from a real `docs/` folder on disk using `DirectoryLoader`

**Why it matters as a PM:**
- Hardcoded strings = the knowledge base is locked inside the code. Every update requires a developer.
- DirectoryLoader = the knowledge base is a folder. A non-technical team can update it by adding/removing files.
- **Product principle:** Separate content from code. This is what makes it maintainable at scale.

**Production path:** Replace `docs/` folder with a Confluence API connector, SharePoint integration, or S3 bucket sync.

---

### Decision 2: chunk_size = 500, overlap = 50

**What:** Split documents into 500-character chunks with 50-character overlap

**Why it matters as a PM:**

| chunk_size | Problem |
|---|---|
| Too large (2000+) | Retrieves the whole document → LLM reads irrelevant content → noisy answers |
| Too small (100) | Splits sentences mid-thought → loses context → incomplete answers |
| 500 (chosen) | One coherent idea per chunk → precise retrieval → focused answers |

**The overlap of 50** prevents this scenario:
```
Chunk 1 ends:   "...refund is issued within"
Chunk 2 starts: "15 business days of billing"
```
Without overlap, retrieval misses the complete sentence. With overlap, both chunks contain the full context.

**PM takeaway:** Chunking strategy directly impacts answer quality. This is a product decision disguised as a technical parameter. Wrong chunking = user trust erosion.

---

### Decision 3: HuggingFace embeddings over OpenAI embeddings

**What:** Use `sentence-transformers/all-MiniLM-L6-v2` (free, local) instead of OpenAI `text-embedding-ada-002`

**Tradeoff analysis:**

| | HuggingFace MiniLM | OpenAI Ada-002 |
|---|---|---|
| Cost | Free | ~$0.0001 per 1K tokens |
| Data privacy | Runs locally, data never leaves | Data sent to OpenAI servers |
| Quality | Good (384-dim) | Better (1536-dim) |
| Setup | Download once (~90MB) | API key required |
| Best for | POC, privacy-sensitive data | Production, highest accuracy |

**PM decision:** For a portfolio POC with policy documents, HuggingFace wins on cost and privacy. For a production enterprise deployment handling sensitive contracts, OpenAI or a private embedding model is better.

**This is the same build vs buy decision PMs make every day** — free and private vs paid and better.

---

### Decision 4: ChromaDB over Pinecone

**What:** Use ChromaDB (local, free) instead of Pinecone (managed, paid)

**Tradeoff analysis:**

| | ChromaDB | Pinecone |
|---|---|---|
| Cost | Free | Starts at $70/month |
| Setup | Zero infra | Account + API key |
| Scale | Up to ~1M vectors locally | Billions of vectors |
| Persistence | Local disk | Cloud managed |
| Best for | POC, local development | Production at scale |

**PM decision:** ChromaDB for POC validation. If this RAG system proves value and needs to scale to 100K+ documents across a team, migrate to Pinecone or Weaviate.

**Product principle:** Don't pay for infra before you've validated the use case. ChromaDB lets you validate in hours, not days.

---

### Decision 5: k=3 (Top-K retrieval)

**What:** Return the 3 most semantically similar chunks for every query

**This is a precision vs recall tradeoff:**

| k value | Effect | When to use |
|---|---|---|
| k=1 | Very precise, might miss context | Simple factual queries |
| k=3 (chosen) | Balanced — good for most policy Q&A | General purpose |
| k=5 | Higher recall, more tokens sent to LLM | Complex multi-part questions |
| k=10 | Very thorough, expensive, slow | Research-style queries |

**PM insight:** k directly affects cost and latency. Every extra chunk = more tokens = higher API cost. At scale, this is a pricing lever. k=3 is a reasonable default. Expose it as a configurable parameter for different query types.

---

### Decision 6: temperature=0 for Claude

**What:** Set `temperature=0` making Claude's output deterministic

**Why:** For policy Q&A, you need consistent, reproducible answers. If two users ask the same question, they should get the same answer. Temperature=0 ensures this.

**When to increase temperature:**
- Creative writing tasks → 0.7–1.0
- Brainstorming → 0.8
- Summarization → 0.3
- Factual Q&A → 0 (this project)

**PM framing:** Temperature is a UX dial. Turn it up for creativity, down for reliability. Enterprise use cases almost always want reliability.

---

### Decision 7: Hallucination Guard in Prompt

**What:** The prompt explicitly says:
```
Answer ONLY based on the context provided.
If not found in context, say "I don't have enough information..."
```

**Why this is the most important product decision:**

Without this instruction, Claude answers from its general training data when it can't find the answer in context. This means:
- Wrong answers that sound confident
- Answers not grounded in your documents
- Legal and compliance risk in enterprise deployments

**This prompt instruction is a product safety feature.** It is the equivalent of a guardrail in a physical product. The hallucination guard is what makes this system enterprise-deployable.

**Test it yourself:** Query 6 in the notebook asks for the Android app download link — which is not in any document. Claude correctly says it doesn't know, instead of making up a URL.

---

## 4. Failure Modes — What Can Go Wrong

| Failure Mode | Cause | Fix |
|---|---|---|
| Wrong answer retrieved | chunk_size too large, k too low | Tune chunk size, increase k |
| Answer not found despite existing in docs | Chunk boundary splits the answer | Increase chunk_overlap |
| Slow responses | k too high, large chunks | Reduce k, reduce chunk_size |
| Claude ignores context | Prompt instruction missing or weak | Strengthen prompt guard |
| Stale answers | ChromaDB not re-indexed after doc update | Add re-indexing trigger on file change |
| High cost at scale | k too high, expensive LLM | Reduce k, use haiku over sonnet |

---

## 5. Production Readiness Checklist

| Feature | POC (this notebook) | Production |
|---|---|---|
| Document loading | Local .txt files | Confluence API, SharePoint, S3 |
| Embedding model | HuggingFace (local) | OpenAI or private model |
| Vector store | ChromaDB (local) | Pinecone, Weaviate, pgvector |
| LLM | Claude haiku | Claude sonnet or opus based on complexity |
| Re-indexing | Manual (re-run notebook) | Triggered on document change |
| Auth & access control | None | Per-user document permissions |
| Observability | Print statements | LangSmith, Arize, or custom logging |
| Latency | ~3–5 seconds | <2 seconds with caching |
| Cost tracking | None | Token usage monitoring per query |

---

## 6. Business Impact Framing

If this RAG system were deployed internally at a 500-person enterprise:

| Metric | Estimate |
|---|---|
| Queries handled per day | 200 (policy/contract questions) |
| Time saved per query | 10 minutes (vs manual search) |
| Daily time saved | 2,000 minutes = 33 hours |
| Monthly time saved | ~660 hours |
| Cost at ₹2,000/hour fully-loaded | ₹13.2 lakh/month saved |
| Infrastructure cost (ChromaDB POC) | ₹0 |
| Claude API cost at 200 queries/day | ~$5–10/day |

**ROI is immediate and significant.** This is the business case for deploying RAG in an enterprise knowledge management context.

---

## 7. Connections to PM Frameworks

### Reforge — Build What Matters
This RAG system solves a **retention problem**, not an acquisition problem. Users who can't find answers leave or stop trusting the product. RAG keeps them engaged with reliable answers.

### Jobs To Be Done
**Job:** "When I need to find a specific policy or contract detail, help me get the exact answer in under 30 seconds without hunting through Confluence."

**RAG hires itself for this job** better than keyword search (too literal) or asking a colleague (too slow).

### Feature vs Infrastructure
RAG is infrastructure, not a feature. It is the foundation on which features (policy bot, contract assistant, onboarding Q&A) are built. This distinction matters for roadmap prioritization — infrastructure investments have compounding returns.

---

## 8. What I Would Build Next

1. **Reranker layer** — After retrieval, re-rank chunks by relevance before sending to LLM. Improves accuracy by ~20%.
2. **Hybrid search** — Combine vector search (semantic) with BM25 (keyword). Better for exact term matches like product names, codes.
3. **Source citation in answer** — Claude shows which document and section the answer came from. Builds user trust.
4. **Conversation memory** — Allow follow-up questions ("what about for monthly plans?") by maintaining chat history.
5. **Feedback loop** — Thumbs up/down on answers → retrain retrieval weights → system gets better over time.

---

*PM Lens Series — RAG Pipeline Project*  
*Author: Pawan Kumar Gadiya | Senior PM, Accenture | ISB MBA 2023 | IIIT Hyderabad AI/ML*
