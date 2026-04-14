"""
Embedding generation using sentence-transformers.

We embed a rich text representation of each candidate that combines:
  - their skills list
  - their role and experience
  - their summary

This "embedding document" is more signal-dense than embedding the raw resume,
since the LLM has already filtered noise and normalized terminology.

Model: all-MiniLM-L6-v2 — 384 dimensions, fast, good quality for semantic search.
"""

from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer

from models import CandidateProfile, JobProfile

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load model once and cache it for the lifetime of the process."""
    print(f"  Loading embedding model: {MODEL_NAME}")
    return SentenceTransformer(MODEL_NAME)


def _candidate_document(profile: CandidateProfile) -> str:
    """
    Build the text we embed for a candidate.
    Putting skills first biases the embedding toward technical match,
    which is what matters most for this use case.
    """
    skills_str = ", ".join(profile.skills)
    return (
        f"Role: {profile.current_role}. "
        f"Experience: {profile.years_exp} years. "
        f"Skills: {skills_str}. "
        f"{profile.summary}"
    )


def _job_document(profile: JobProfile) -> str:
    """Build the text we embed for a job description."""
    skills_str = ", ".join(profile.required_skills)
    return (
        f"Role: {profile.title}. "
        f"Minimum experience: {profile.min_years_exp} years. "
        f"Required skills: {skills_str}. "
        f"{profile.summary}"
    )


def embed_candidate(profile: CandidateProfile) -> List[float]:
    """Return a 384-dim embedding for a candidate profile."""
    model = _get_model()
    doc = _candidate_document(profile)
    vector = model.encode(doc, normalize_embeddings=True)
    return vector.tolist()


def embed_job(profile: JobProfile) -> List[float]:
    """Return a 384-dim embedding for a job profile."""
    model = _get_model()
    doc = _job_document(profile)
    vector = model.encode(doc, normalize_embeddings=True)
    return vector.tolist()


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Batch-embed arbitrary texts. Useful for ad-hoc queries."""
    model = _get_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]
