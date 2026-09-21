# Product Brief: RAG (Retrieval-Augmented Generation)

**Date:** [Today's date]
**Author:** Pawan Menda | Senior PM | AI/ML Learning Journey — Day 1

---

## Problem
Enterprise knowledge workers (support agents, sales reps, analysts) 
spend 30–40% of their time searching for information across 
disconnected internal systems — wikis, PDFs, CRMs, email threads. 
When they can't find it, they either make decisions on incomplete 
information or escalate, slowing down operations.

LLMs promise to answer these questions instantly — but without 
grounding in actual company documents, they hallucinate confidently 
wrong answers. This is worse than no answer in an enterprise context 
where decisions have downstream consequences (compliance, finance, ops).

---

## Primary User
**Role:** Enterprise knowledge worker (Support Agent / Sales Rep / Analyst)  
**Workflow today:** Search → Ctrl+F across docs → Ask a colleague → 
                    Wait → Maybe find answer → Decide  
**Core frustration:** "I know the answer is somewhere in our docs. 
                       I just can't find it fast enough."

---

## Solution: RAG Changes the Workflow
**Before RAG:**
User searches manually → reads 5 docs → pieces together an answer → 
3–10 minutes per query → often incomplete

**After RAG:**
User asks in natural language → system retrieves the 2-3 most 
relevant document chunks → LLM synthesizes a grounded answer 
with source citations → 10–30 seconds → verifiable

---

## Success Metrics
| Metric | Definition | Target |
|---|---|---|
| Answer Accuracy | % of answers factually correct vs. source doc | > 90% |
| Hallucination Rate | % of answers containing unsupported claims | < 5% |
| Time-to-Answer | Avg seconds from query to response | < 30s |
| Task Completion Rate | % of queries that didn't require human follow-up | > 75% |
| Support Ticket Deflection | % of tickets resolved by RAG before human agent | > 40% |

---

## Key Risks & Failure Modes
- **Wrong chunk retrieved:** Semantic search returns a plausible but 
  wrong document section → model answers confidently from wrong context
- **Outdated documents:** KB not updated → RAG returns stale information
- **Access control gaps:** User retrieves documents they shouldn't 
  see → compliance/security risk (critical in B2B enterprise)
- **Query-document mismatch:** User asks in casual language, 
  document uses technical jargon → poor retrieval

---

## PM Insight
Understanding RAG changes how I evaluate AI products as a PM. 
When a vendor says "our AI assistant knows your company's data" — 
the right question is: *how is retrieval happening? What's the 
chunking strategy? How do you handle access control on retrieval?* 
Most enterprise RAG failures aren't model failures — they're 
retrieval architecture failures. As a PM in B2B SaaS, 
this is the layer I need to spec, test, and own.
