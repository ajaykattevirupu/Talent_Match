# Mini Talent-Matching Pipeline

A production-style RAG pipeline that matches job candidates to job descriptions using:

- **Claude (Haiku)** — structured extraction via tool use + LLM reranking with explanations  
- **sentence-transformers** — dense embeddings (`all-MiniLM-L6-v2`, 384-dim)  
- **pgvector** — vector similarity search inside PostgreSQL  

Covers: structured output, embeddings, RAG, vector search, LLM reranking.

## Architecture

```
Resumes ──► Claude (extract) ──► CandidateProfile ──► Embed ──► pgvector
                                                                      │
JD ──────► Claude (extract) ──► JobProfile ──► Embed ──► cos search ─┘
                                                              │
                                             top-10 candidates│
                                                              ▼
                                              Claude (rerank + explain)
                                                              │
                                              top-3 MatchResults ──► stdout
```

## Quick Start

### 1. Start PostgreSQL with pgvector

```bash
docker run --name pgvector \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=talent_match \
  -p 5432:5432 \
  -d ankane/pgvector:latest
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set environment variables

```bash
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY
export $(cat .env | xargs)
```

### 4. Run the pipeline

```bash
python main.py
```

## File Layout

| File | Purpose |
|------|---------|
| `data.py` | 5 sample resumes + 5 job descriptions |
| `models.py` | Pydantic models: `CandidateProfile`, `JobProfile`, `MatchResult` |
| `extractor.py` | Claude tool-use extraction for resumes and JDs |
| `embeddings.py` | sentence-transformers embedding generation |
| `db.py` | pgvector upsert + cosine similarity search |
| `matcher.py` | Two-stage matching: vector recall → LLM rerank |
| `main.py` | Pipeline orchestrator |

## Key Design Decisions

**Why embed the extracted profile, not the raw resume?**  
The LLM normalizes terminology (e.g., "Hugging Face" and "transformers" become consistent), removes noise (addresses, formatting), and condenses signal. Embedding the structured text produces tighter, more comparable vectors.

**Why two-stage retrieval?**  
Vector search is fast but context-unaware. LLM reranking can weigh factors like seniority fit, rare skill combinations, and domain relevance that pure cosine similarity misses. This is the same pattern used in production search systems.

**Why prompt caching on the system prompt?**  
The system prompt is identical across all extraction calls. Marking it with `cache_control: ephemeral` means Anthropic caches it for 5 minutes, reducing input tokens billed on calls 2–N significantly.
