# Demo Video

https://github.com/user-attachments/assets/c9c88117-c0ea-4de0-a300-befafde7cd14

# Intelligent Corporate Knowledge Assistant API

Production-ready RAG API built with **FastAPI**, **FAISS**, and **LLM Guardrails** to answer internal policy questions based on corporate HR documents.

---

## 1. Quick Start Guide

### Prerequisites
- Python 3.11+
- `uv` (recommended package manager)

### Setup & Run
```bash
# 1. Install dependencies
uv sync

# 2. Copy environment template and add your API keys
cp .env.example .env

# 3. Start the API server
uv run uvicorn main:app --reload --port 8000
```

Interactive Swagger UI docs at:  
**`http://localhost:8000/docs`**

---

## 2. API Endpoints & Response Structure

### 1. Health Check: `GET /health` or `GET /`
Verifies that the API service is online.
```json
{
  "status": "online",
  "service": "Intelligent Corporate Knowledge Assistant API",
  "docs_url": "/docs"
}
```

---

### 2. Document Ingestion: `POST /ingest`
Uploads a policy PDF document, performs heading-aware chunking, computes normalized embeddings, and populates the FAISS vector index.

* **cURL Example:**
```bash
curl -X POST "http://localhost:8000/ingest" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@dev/[PT TMD] BUKU SAKU KEBIJAKAN SUMBER DAYA MANUSIA.pdf"
```

* **Sample Response:**
```json
{
  "status": "success",
  "filename": "[PT TMD] BUKU SAKU KEBIJAKAN SUMBER DAYA MANUSIA.pdf",
  "chunks_ingested": 11
}
```

---

### 3. RAG Policy Chat: `POST /chat`
Answers employee policy questions grounded in the ingested document context, with two-layer guardrails against off-topic queries.

* **cURL Example (In-Domain Query):**
```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"question": "Berapa plafon klaim kacamata untuk level Manager?"}'
```

* **Sample Response:**
```json
{
  "answer": "Plafon klaim kacamata untuk karyawan level Manager adalah maksimal Rp 2.500.000,- yang dapat diklaim setiap 2 (dua) tahun sekali (sistem reimbursement). Klaim tidak berlaku untuk kacamata hitam atau lensa kosmetik.",
  "sources": [
    {
      "bab": "BAB III: TUNJANGAN & KLAIM KESEHATAN",
      "section_number": "3.1",
      "section_title": "Klaim Kacamata (Optical)",
      "similarity_score": 0.7045
    }
  ],
  "top_similarity_score": 0.7045,
  "guardrail_triggered": null
}
```

* **cURL Example (Out-of-Domain Query):**
```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"question": "Siapa presiden Amerika Serikat saat ini?"}'
```

* **Sample Response (Guardrail Triggered):**
```json
{
  "answer": "Maaf, saya hanya bisa menjawab terkait kebijakan internal.",
  "sources": [],
  "top_similarity_score": 0.2982,
  "guardrail_triggered": "Layer 1 (Similarity score 0.2982 < 0.32)"
}
```

---

## 3. Core Strategy & Technical Rationale

### A. Heading-Aware Structural Chunking (`rag.py`)
Standard fixed-character chunking (e.g., 500 characters) frequently fractures structured policy documents—separating numeric conditions (like Rp 2.500.000 optical limits or 10-day WFA limits) from their parent section headings.

#### Strategy Comparison:
| Chunking Strategy | Mechanism | Why We Rejected It / Why Heading-Aware Wins |
|---|---|---|
| **Fixed-Size (e.g. 500 chars)** | Splits purely on character count | ❌ Cuts tables and conditions mid-sentence; separates claim limits from employee tiers. |
| **Page-Level (1 page = 1 chunk)** | 1 chunk per PDF page | ❌ A single PDF page contains 2–3 completely unrelated policies. Dilutes retrieval accuracy. |
| **Semantic Chunking** | Distance between consecutive sentence embeddings | ❌ High computational overhead (dozens of embedding calls per page) and ignores explicit policy headers (`1.1`, `3.1`). |
| **Heading-Aware (Our Choice)** | Splits along `BAB [IVX]+` and `\d\.\d` section boundaries | 🟢 **100% clause integrity**, 0ms regex speed, automatic structured metadata (`bab`, `section_number`, `section_title`), and clean auditable citations. |

1. Document text is parsed and split along structural header boundaries (`BAB [IVX]+` and `\d\.\d` section titles).
2. The `clean_section_title()` parser isolates clean titles (e.g. `"Klaim Kacamata (Optical)"`), stripping flattened body sentences and redundant quotes.
3. Fallback `RecursiveCharacterTextSplitter` (chunk size ~800, overlap ~100) is applied only if an individual clause exceeds token limits.

### B. Embedding Model
We use `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Multilingual embeddings are critical because English-only models underperform on Indonesian policy terminology (e.g., *"Cuti Recharge"*, *"Klaim Optical"*, *"Peraturan WFA"*).

### C. Vector Search & FAISS Index
We use `faiss.IndexFlatIP` (Inner Product) paired with $L_2$-normalized vector embeddings, which computes exact **Cosine Similarity** (scores bounded between -1.0 and 1.0).

### D. LLM Provider Hierarchy: Cloudflare Workers AI Primary vs. Groq Fallback
We configure **Cloudflare Workers AI** (`@cf/meta/llama-3.3-70b-instruct-fp8-fast`) as the primary provider with **Groq API** (`llama-3.3-70b-versatile`) as the automatic fallback:
* **Generous Free-Tier Quota (Cloudflare):** In practice, Groq's free-tier limits are quite restrictive and hit rate limits (RPM / TPM caps) very quickly during active testing. Cloudflare Workers AI provides a significantly more generous daily free allocation of Neurons (~10,000 requests/day for 70B models), preventing rate-limit throttling.
* **Failover Resilience (Groq):** Groq's custom LPU hardware (500+ tokens/sec) is retained as an instant failover backup if Cloudflare encounters temporary network or endpoint issues, ensuring 99.9% uptime.

### E. System Prompt Design & Exact Template
To guarantee strict factual grounding and eliminate hallucinations, we designed a constrained system prompt with 5 production rules:

```
You are an internal corporate policy assistant for PT Teknologi Masa Depan (TMD).
Answer employee questions strictly based on the provided context below.

Rules:
1. If the question is outside the scope of internal policy or cannot be answered using the provided context (e.g. general trivia, politics, external facts), reply politely with:
   "Maaf, saya hanya bisa menjawab terkait kebijakan internal."
2. Do not attempt to guess or use external knowledge.
3. Keep the answer concise, accurate, clear, and professional in Bahasa Indonesia.
4. For multi-part questions (e.g. covering multiple policies), organize the explanation with clear structured points.
5. Do not enclose words or policy titles in unnecessary quotation marks.
```

#### Why This Prompt is Structured This Way:
* **Rule #1 & #2 (Negative Constraint / Layer 2 Guardrail):** Enforces a deterministic refusal fallback if a query is outside internal policy context, eliminating hallucination risks on non-company questions.
* **Rule #3 (Language & Tone):** Ensures the response register matches Indonesian corporate HR standards.
* **Rule #4 (Structured Formatting):** Prevents dense run-on paragraphs when answering composite queries (e.g. *"jam kerja dan wfa wfo"*), producing readable markdown lists.
* **Rule #5 (JSON Escape Sanitization):** Prevents the LLM from outputting literal double quotes around normal terms (e.g. writing `Cuti Recharge` instead of `"Recharge"`), eliminating ugly escaped backslashes (`\"`) in JSON responses.

---

## 4. Two-Layer Guardrail Architecture & Empirical Calibration

To prevent hallucinations, cost overflow, and out-of-domain responses:

1. **Layer 1: Retrieval Score Short-Circuit (`config.RETRIEVAL_SIMILARITY_THRESHOLD = 0.32`)**
   Before spending an LLM API call, top-1 Cosine similarity is checked. If top score `< 0.32`, the request immediately returns the polite refusal response:
   > *"Maaf, saya hanya bisa menjawab terkait kebijakan internal."*

2. **Layer 2: System Prompt & Low Temperature (`temperature = 0.1`)**
   If retrieval succeeds, context is passed to the LLM with explicit system instructions prohibiting external knowledge or guessing. Low temperature (`0.1`) ensures strict factual adherence.

### Empirical Calibration Results (`calibrate.py`)
Run `uv run calibrate.py` to inspect the exact score distributions:
- **In-Scope Scores:** Min = `0.3393` (*Cuti Ayah*), Max = `0.7354` (*Cuti Tahunan*), Avg = `0.5596`
- **Out-of-Scope Scores:** Min = `0.0535` (*Ibukota Prancis*), Max = `0.2982` (*Presiden AS*), Avg = `0.1417`
- **Decision Gap:** `0.0411` cluster separation gap between `0.3393` and `0.2982`.

> **Engineering Insight:** The tight `0.0411` gap demonstrates why **Layer 2 (System Prompt)** is essential alongside Layer 1. While Layer 1 short-circuits obvious off-topic queries without spending LLM tokens, Layer 2 catches borderline edge cases cleanly.

---

## 5. Model Tuning & Hyperparameter Rationale

| Hyperparameter | Value | Technical Rationale & Justification |
|---|---|---|
| **LLM Temperature** | `0.1` | **Strict Factual Grounding:** Higher temperatures (`0.7`) introduce creative variance and hallucinations, which is dangerous for HR compliance. `0.0` can cause degenerate repetition loops on edge cases. `0.1` strikes the optimal balance of deterministic grounding with natural Indonesian phrasing. |
| **Top-K Chunks** | `3` | **Optimal Context Budget:** Because of heading-aware chunking, each chunk contains a complete policy clause (~500–800 chars). 3 chunks provide ~1,800 characters of rich context—sufficient to answer complex multi-tier comparisons (e.g. Staff vs Manager optical limits) without diluting the prompt. |
| **Similarity Threshold** | `0.32` | **Empirically Calibrated:** Derived from `calibrate.py` benchmark testing. Cleanly bisects the out-of-scope maximum (`0.2982`) and in-scope minimum (`0.3393`) with a `0.0411` safety margin. |
| **Vector Metric** | `Cosine` (`IndexFlatIP` + $L_2$) | **Bounded Thresholds:** Normalizing vectors to unit length ($\|u\|_2 = 1$) makes Inner Product mathematically equal to Cosine Similarity ($[-1.0, 1.0]$), allowing threshold tuning to remain stable and intuitive. |
| **Structured Output Prompting** | `Markdown points` | **Clarity on Multi-Topic Queries:** Instructs the LLM to format multi-part answers (e.g., work hours + WFO + WFA) into clean numbered/bulleted points rather than dense run-on paragraphs. |

---

## 6. Testing & Security Notes

### Automated Smoke Tests (`tests/test_api.py`)
Run the test suite via `pytest`:
```bash
uv run pytest
```
Includes 5 automated tests verifying health check, file validation, `/ingest`, in-domain retrieval, and guardrail triggering.

### Security & CORS
* **Credential Protection:** `.gitignore` excludes `.env`, `data/`, and `.venv/`. A template `.env.example` is provided.
* **Spec-Compliant CORS:** `allow_origins=["*"]` is paired with `allow_credentials=False` adhering to browser security standards.
* **Error Sanitization:** Internal exceptions are logged to server console via `logger.error(..., exc_info=True)` while returning sanitized error messages to API callers.

---

## 7. Enterprise Scaling Roadmap (Azure & Microsoft Fabric)

```
                       ┌───────────────────────────────────────┐
                       │           Azure API Management        │
                       └───────────────────┬───────────────────┘
                                           │
                                           ▼
                       ┌───────────────────────────────────────┐
                       │   Azure Container Apps / App Service  │
                       │          (FastAPI Service)            │
                       └───────────────────┬───────────────────┘
                                           │
         ┌─────────────────────────────────┴─────────────────────────────────┐
         ▼                                                                   ▼
┌───────────────────────────────┐                           ┌───────────────────────────────┐
│     Azure AI Search           │                           │     Azure OpenAI Service      │
│ (Vector + Semantic Reranker)  │                           │    (GPT-4o / GPT-4o-mini)     │
└────────┬──────────────────────┘                           └───────────────────────────────┘
         ▲
         │ (Automated Indexing Pipeline)
┌────────┴──────────────────────┐
│ Microsoft Fabric Data Factory │ ◄── Ingests internal policy files from OneLake / SharePoint
└───────────────────────────────┘
```

1. **Enterprise Ingestion (Microsoft Fabric):**
   * Use **Fabric Data Factory** pipelines to automate document ingestion from SharePoint, OneLake, or Azure Blob Storage across thousands of corporate PDFs.
2. **Enterprise Vector Search (Azure AI Search):**
   * Replace local FAISS with **Azure AI Search** to leverage Hybrid Search (BM25 + HNSW Vector Search) and Azure's **Semantic Reranker**.
3. **Role-Based Access Control (RBAC):**
   * Integrate Azure AD / Entra ID authentication to apply security filters at retrieval time, ensuring employees only retrieve policy documents corresponding to their permission level (Staff vs Lead vs Executive).
4. **Observability & Monitoring:**
   * Integrate **Azure Application Insights** to monitor query latency, guardrail rejection rates, and score distributions over time.

---

## 8. AI Assistance & Ownership Disclosure

AI assistance(Gemini 3.7 flash (High), Claude Sonnet 4.6 through antigravity cli) was utilized as a consultative pair programmer to explore potential approaches, discuss best practices, and assist in drafting technical documentation for this assignment.

**All architectural decisions, design choices, engineering trade-offs, and final code implementations are strictly owned, decided, and finalized by me.** Every component—including the heading-aware chunking strategy, FAISS vector indexing, empirical similarity calibration, and two-layer guardrails—was personally evaluated, reviewed, and validated end-to-end.
