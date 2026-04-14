"""
Mini Talent-Matching Pipeline
==============================

Orchestrates the full pipeline:

  INDEXING PHASE
  ─────────────
  For each resume:
    1. Extract structured CandidateProfile (GPT-4o-mini function calling)
    2. Generate embedding (sentence-transformers all-MiniLM-L6-v2)
    3. Store in pgvector

  QUERY PHASE
  ───────────
  For each job description:
    1. Extract structured JobProfile (GPT-4o-mini function calling)
    2. Embed the JD
    3. Vector search → top-10 candidates by cosine similarity
    4. LLM rerank → top-3 candidates with explanations

Run:
  python main.py

Prerequisites:
  pip install -r requirements.txt
  export ANTHROPIC_API_KEY=sk-ant-...
  export DATABASE_URL=postgresql://postgres:password@localhost:5432/talent_match
  createdb talent_match   (or: docker-compose up)
"""

import os
import sys
import textwrap

from data import JOB_DESCRIPTIONS, RESUMES
from db import ensure_tables, upsert_candidate
from embeddings import embed_candidate
from extractor import extract_candidate, extract_job
from matcher import find_top_candidates
from models import CandidateRecord


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _divider(char="─", width=70):
    print(char * width)


def _print_match(match, rank: int):
    print(f"\n  #{rank}  {match.candidate_name}  (similarity: {match.similarity_score:.4f})")
    print(f"       Skills matched: {', '.join(match.skill_overlap[:6])}")
    wrapped = textwrap.fill(match.llm_explanation, width=66, initial_indent="       ", subsequent_indent="       ")
    print(wrapped)


# ---------------------------------------------------------------------------
# Phase 1: Index all candidates
# ---------------------------------------------------------------------------

def index_candidates():
    _divider("═")
    print("PHASE 1: INDEXING CANDIDATES")
    _divider("═")

    ensure_tables()

    for resume_data in RESUMES:
        print(f"\n  Processing: {resume_data['name']}")

        # Step 1 — LLM extraction (structured output via tool use)
        print("    → Extracting profile with GPT-4o-mini...")
        profile = extract_candidate(resume_data["text"])
        print(f"      Role: {profile.current_role} | Exp: {profile.years_exp} yrs")
        print(f"      Skills: {', '.join(profile.skills[:6])}{'...' if len(profile.skills) > 6 else ''}")

        # Step 2 — Embedding
        print("    → Generating embedding...")
        embedding = embed_candidate(profile)

        # Step 3 — Store in pgvector
        record = CandidateRecord(
            id=resume_data["id"],
            name=resume_data["name"],
            raw_resume=resume_data["text"],
            profile=profile,
            embedding=embedding,
        )
        upsert_candidate(record)
        print(f"    ✓ Stored in pgvector (dim={len(embedding)})")

    print(f"\n  Indexed {len(RESUMES)} candidates successfully.")


# ---------------------------------------------------------------------------
# Phase 2: Match candidates to each JD
# ---------------------------------------------------------------------------

def match_jobs():
    _divider("═")
    print("PHASE 2: MATCHING CANDIDATES TO JOB DESCRIPTIONS")
    _divider("═")

    for jd_data in JOB_DESCRIPTIONS:
        print(f"\n\nJOB: {jd_data['title']}")
        _divider()

        # Step 1 — Extract structured job profile
        print("  → Extracting job profile with GPT-4o-mini...")
        job_profile = extract_job(jd_data["text"])
        print(f"     Min exp: {job_profile.min_years_exp} yrs | Skills: {', '.join(job_profile.required_skills[:5])}...")

        # Step 2 + 3: Vector search + LLM rerank (in matcher.find_top_candidates)
        print("  → Running vector search + LLM reranking...")
        matches = find_top_candidates(job_profile, top_n=3)

        print("\n  TOP 3 CANDIDATES:")
        for match in matches:
            _print_match(match, match.rank)

    print("\n")
    _divider("═")
    print("Pipeline complete.")
    _divider("═")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # Validate required env vars before starting
    missing = [v for v in ("OPENAI_API_KEY", "DATABASE_URL") if not os.environ.get(v)]
    if missing:
        print(f"ERROR: Missing environment variables: {', '.join(missing)}")
        print("Set them in your shell or in a .env file (see .env.example)")
        sys.exit(1)

    index_candidates()
    match_jobs()


if __name__ == "__main__":
    main()
