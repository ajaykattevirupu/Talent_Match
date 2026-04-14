# Mini Talent-Matching Pipeline

A production-style RAG pipeline that matches job candidates to job descriptions using:

- **GPT-4o-mini** — structured extraction via function calling + LLM reranking with explanations  
- **sentence-transformers** — dense embeddings (`all-MiniLM-L6-v2`, 384-dim)  
- **pgvector** — vector similarity search inside PostgreSQL  

Covers: structured output, embeddings, RAG, vector search, LLM reranking.

## Architecture

```
Resumes ──► GPT-4o-mini (extract) ──► CandidateProfile ──► Embed ──► pgvector
                                                                           │
JD ──────► GPT-4o-mini (extract) ──► JobProfile ──► Embed ──► cos search ─┘
                                                                   │
                                              top-10 candidates    │
                                                                   ▼
                                              GPT-4o-mini (rerank + explain)
                                                                   │
                                              top-3 MatchResults ──► stdout
```

## Quick Start

### 1. Install PostgreSQL with pgvector

Download PostgreSQL 16 from https://www.postgresql.org/download/windows/ and install the pgvector extension.

Then create the database:
```powershell
psql -U postgres -h 127.0.0.1 -c "CREATE DATABASE talent_match;"
psql -U postgres -h 127.0.0.1 -d talent_match -c "CREATE EXTENSION vector;"
```

### 2. Install dependencies

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

### 3. Set environment variables

```powershell
$env:OPENAI_API_KEY = "sk-..."
$env:DATABASE_URL = "postgresql://postgres:password@localhost:5432/talent_match"
```

### 4. Run the pipeline

```powershell
py main.py
```

## File Layout

| File | Purpose |
|------|---------|
| `data.py` | 5 sample resumes + 5 job descriptions |
| `models.py` | Pydantic models: `CandidateProfile`, `JobProfile`, `MatchResult` |
| `extractor.py` | GPT-4o-mini function calling extraction for resumes and JDs |
| `embeddings.py` | sentence-transformers embedding generation |
| `db.py` | pgvector upsert + cosine similarity search |
| `matcher.py` | Two-stage matching: vector recall → LLM rerank |
| `main.py` | Pipeline orchestrator |

## Key Design Decisions

**Why embed the extracted profile, not the raw resume?**  
The LLM normalizes terminology (e.g., "Hugging Face" and "transformers" become consistent), removes noise (addresses, formatting), and condenses signal. Embedding the structured text produces tighter, more comparable vectors.

**Why two-stage retrieval?**  
Vector search is fast but context-unaware. LLM reranking can weigh factors like seniority fit, rare skill combinations, and domain relevance that pure cosine similarity misses. This is the same pattern used in production search systems.

**Why function calling for extraction?**  
Forcing the model to call a typed function schema guarantees structured, validated output — no regex parsing or prompt hacking needed. The schema acts as a contract between the LLM and the rest of the pipeline.
