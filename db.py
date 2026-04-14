"""
pgvector database layer.

Schema:
  candidates — stores structured profile + embedding for each candidate
  jobs        — stores structured profile + embedding for each job (optional; used for batch runs)

Vector similarity uses cosine distance (<=>), which works well with
normalized embeddings (normalize_embeddings=True in embeddings.py).

The table is created on first use via ensure_tables().
"""

import json
import os
from typing import List, Tuple

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

from models import CandidateProfile, CandidateRecord

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def _get_conn():
    """
    Open a connection using DATABASE_URL env var.
    Example: postgresql://postgres:password@localhost:5432/talent_match
    """
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    register_vector(conn)  # teach psycopg2 how to handle the vector type
    return conn


# ---------------------------------------------------------------------------
# Schema setup
# ---------------------------------------------------------------------------

CREATE_TABLES_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS candidates (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    raw_resume  TEXT NOT NULL,
    profile_json JSONB NOT NULL,       -- full CandidateProfile as JSON
    embedding   vector(384)            -- all-MiniLM-L6-v2 dimensionality
);

-- IVFFlat index for fast approximate nearest-neighbor search.
-- lists=10 is reasonable for small datasets; increase for larger corpora.
-- Re-run with DROP INDEX + CREATE after inserting bulk data.
CREATE INDEX IF NOT EXISTS candidates_embedding_idx
    ON candidates
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 10);
"""


def ensure_tables() -> None:
    """Create tables and vector index if they don't exist yet."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLES_SQL)
        conn.commit()
    print("  Database tables ready.")


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------

def upsert_candidate(record: CandidateRecord) -> None:
    """
    Insert or update a candidate record.
    Uses ON CONFLICT DO UPDATE so re-running the pipeline is idempotent.
    """
    sql = """
    INSERT INTO candidates (id, name, raw_resume, profile_json, embedding)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (id) DO UPDATE SET
        name         = EXCLUDED.name,
        raw_resume   = EXCLUDED.raw_resume,
        profile_json = EXCLUDED.profile_json,
        embedding    = EXCLUDED.embedding;
    """
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (
                record.id,
                record.name,
                record.raw_resume,
                json.dumps(record.profile.model_dump()),
                record.embedding,
            ))
        conn.commit()


def clear_candidates() -> None:
    """Remove all candidate rows (useful for test resets)."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM candidates;")
        conn.commit()


# ---------------------------------------------------------------------------
# Read / search operations
# ---------------------------------------------------------------------------

def search_candidates(
    query_embedding: List[float],
    top_k: int = 10,
) -> List[Tuple[CandidateRecord, float]]:
    """
    Find the top-k most similar candidates to a query embedding using
    cosine distance (lower = more similar; we return 1 - distance as score).

    Returns a list of (CandidateRecord, similarity_score) sorted desc by score.
    """
    sql = """
    SELECT
        id,
        name,
        raw_resume,
        profile_json,
        embedding,
        1 - (embedding <=> %s::vector) AS similarity
    FROM candidates
    ORDER BY embedding <=> %s::vector
    LIMIT %s;
    """
    results = []
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(sql, (query_embedding, query_embedding, top_k))
            rows = cur.fetchall()

    for row in rows:
        profile = CandidateProfile(**row["profile_json"])
        record = CandidateRecord(
            id=row["id"],
            name=row["name"],
            raw_resume=row["raw_resume"],
            profile=profile,
            embedding=list(row["embedding"]),
        )
        results.append((record, float(row["similarity"])))

    return results


def get_all_candidates() -> List[CandidateRecord]:
    """Fetch all stored candidates (for debugging / inspection)."""
    sql = "SELECT id, name, raw_resume, profile_json, embedding FROM candidates;"
    records = []
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(sql)
            rows = cur.fetchall()

    for row in rows:
        profile = CandidateProfile(**row["profile_json"])
        records.append(CandidateRecord(
            id=row["id"],
            name=row["name"],
            raw_resume=row["raw_resume"],
            profile=profile,
            embedding=list(row["embedding"]),
        ))
    return records
