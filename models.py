"""
Pydantic models for structured data throughout the pipeline.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class CandidateProfile(BaseModel):
    """Structured profile extracted from a raw resume by the LLM."""

    name: str = Field(description="Full name of the candidate")
    current_role: str = Field(description="Most recent job title")
    years_exp: float = Field(description="Total years of professional experience")
    skills: List[str] = Field(description="List of technical skills (tools, languages, frameworks)")
    summary: str = Field(description="2-3 sentence summary of the candidate's strengths and focus areas")


class JobProfile(BaseModel):
    """Structured profile extracted from a raw job description by the LLM."""

    title: str = Field(description="Job title")
    min_years_exp: float = Field(description="Minimum years of experience required")
    required_skills: List[str] = Field(description="Required technical skills")
    summary: str = Field(description="2-3 sentence summary of the role and ideal candidate")


class CandidateRecord(BaseModel):
    """A candidate stored in the database, including embedding."""

    id: str
    name: str
    raw_resume: str
    profile: CandidateProfile
    embedding: Optional[List[float]] = None


class MatchResult(BaseModel):
    """A single match result returned by the matcher."""

    rank: int
    candidate_id: str
    candidate_name: str
    similarity_score: float
    llm_explanation: str
    skill_overlap: List[str]
