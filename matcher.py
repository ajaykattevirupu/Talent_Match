"""
Talent matching pipeline: vector search → LLM reranking + explanation.

Two-stage retrieval:
  Stage 1 — pgvector cosine similarity: fast, retrieves top-k candidates
  Stage 2 — LLM reranking: Claude reads the JD + candidate summaries and
             produces a ranked list with per-candidate explanations

This mirrors production RAG reranking patterns (retrieve-then-rerank).
"""

import json
import os
from typing import List

from openai import OpenAI

from db import search_candidates
from embeddings import embed_job
from models import CandidateRecord, JobProfile, MatchResult

# How many candidates to pull from vector search before LLM reranking
VECTOR_RECALL_K = 10


def _get_client() -> OpenAI:
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ---------------------------------------------------------------------------
# Stage 1: Vector similarity retrieval
# ---------------------------------------------------------------------------

def vector_search(job_profile: JobProfile, top_k: int = VECTOR_RECALL_K):
    """Embed the JD and retrieve the top-k candidates by cosine similarity."""
    job_embedding = embed_job(job_profile)
    return search_candidates(job_embedding, top_k=top_k)


# ---------------------------------------------------------------------------
# Stage 2: LLM reranking + explanation
# ---------------------------------------------------------------------------

RERANK_FUNCTION = {
    "name": "rank_candidates",
    "description": (
        "Given a job description and a list of candidates, rank the top 3 "
        "best-fit candidates and explain why each is a strong match."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "rankings": {
                "type": "array",
                "description": "Ordered list of top 3 candidates, best match first.",
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate_id": {
                            "type": "string",
                            "description": "The candidate's ID string",
                        },
                        "explanation": {
                            "type": "string",
                            "description": (
                                "2-3 sentence explanation of why this candidate is a "
                                "strong fit, citing specific skills and experience."
                            ),
                        },
                        "skill_overlap": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Skills the candidate has that the JD requires.",
                        },
                    },
                    "required": ["candidate_id", "explanation", "skill_overlap"],
                },
                "minItems": 3,
                "maxItems": 3,
            }
        },
        "required": ["rankings"],
    },
}


def _build_rerank_prompt(
    job_profile: JobProfile,
    candidates: List[tuple[CandidateRecord, float]],
) -> str:
    """Serialize JD + candidate list into a prompt for the reranker."""
    lines = [
        f"JOB: {job_profile.title}",
        f"Minimum experience required: {job_profile.min_years_exp} years",
        f"Required skills: {', '.join(job_profile.required_skills)}",
        f"Description: {job_profile.summary}",
        "",
        "CANDIDATES (pre-filtered by semantic similarity):",
    ]

    for record, score in candidates:
        p = record.profile
        lines.append(
            f"\n[ID: {record.id}] {p.name} | {p.current_role} | {p.years_exp} yrs exp\n"
            f"  Skills: {', '.join(p.skills)}\n"
            f"  Summary: {p.summary}\n"
            f"  Vector similarity score: {score:.4f}"
        )

    lines.append(
        "\nRank the top 3 candidates for this role. "
        "Prioritize technical skill match, relevant experience, and seniority fit."
    )
    return "\n".join(lines)


def llm_rerank(
    job_profile: JobProfile,
    candidates: List[tuple[CandidateRecord, float]],
) -> List[dict]:
    """
    Ask GPT-4o-mini to rerank candidates and produce structured explanations.
    Returns the raw rankings list from the function call.
    """
    client = _get_client()
    prompt = _build_rerank_prompt(job_profile, candidates)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=1500,
        tools=[{"type": "function", "function": RERANK_FUNCTION}],
        tool_choice={"type": "function", "function": {"name": "rank_candidates"}},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert technical recruiter. Evaluate candidates objectively "
                    "based on their skills, experience, and fit for the role. "
                    "Be specific and cite evidence from the candidate profiles."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    tool_call = response.choices[0].message.tool_calls[0]
    data = json.loads(tool_call.function.arguments)
    return data["rankings"]


# ---------------------------------------------------------------------------
# Full match pipeline
# ---------------------------------------------------------------------------

def find_top_candidates(
    job_profile: JobProfile,
    top_n: int = 3,
) -> List[MatchResult]:
    """
    End-to-end matching for a job:
      1. Vector search to recall top candidates
      2. LLM reranking with explanations
      3. Returns top_n MatchResult objects

    Args:
        job_profile: Structured JobProfile (already extracted by extractor.py)
        top_n: Number of final candidates to return (default 3)
    """
    # Stage 1: vector recall
    recalled = vector_search(job_profile, top_k=VECTOR_RECALL_K)
    if not recalled:
        return []

    # Build a lookup map for quick access during reranking
    candidate_map = {record.id: (record, score) for record, score in recalled}

    # Stage 2: LLM rerank
    rankings = llm_rerank(job_profile, recalled)

    # Assemble final MatchResult objects
    results = []
    for rank_idx, ranking in enumerate(rankings[:top_n], start=1):
        cid = ranking["candidate_id"]
        if cid not in candidate_map:
            continue  # guard against hallucinated IDs
        record, sim_score = candidate_map[cid]
        results.append(
            MatchResult(
                rank=rank_idx,
                candidate_id=cid,
                candidate_name=record.name,
                similarity_score=sim_score,
                llm_explanation=ranking["explanation"],
                skill_overlap=ranking["skill_overlap"],
            )
        )

    return results
