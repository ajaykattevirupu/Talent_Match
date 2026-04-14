"""
LLM-based structured extraction using OpenAI function calling.

For each resume or JD, we ask GPT-4o-mini to call a predefined function whose
schema matches our Pydantic model. This guarantees structured, type-safe output.
"""

import json
import os

from openai import OpenAI

from models import CandidateProfile, JobProfile

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


# ---------------------------------------------------------------------------
# Function schemas (mirrors Pydantic models; OpenAI will populate these fields)
# ---------------------------------------------------------------------------

CANDIDATE_FUNCTION = {
    "name": "extract_candidate_profile",
    "description": "Extract structured information about a job candidate from their resume.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Full name of the candidate"},
            "current_role": {"type": "string", "description": "Most recent job title"},
            "years_exp": {
                "type": "number",
                "description": "Total years of professional experience (approximate)",
            },
            "skills": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of technical skills: programming languages, frameworks, tools, platforms",
            },
            "summary": {
                "type": "string",
                "description": "2-3 sentence summary of the candidate's strengths and focus areas",
            },
        },
        "required": ["name", "current_role", "years_exp", "skills", "summary"],
    },
}

JOB_FUNCTION = {
    "name": "extract_job_profile",
    "description": "Extract structured information from a job description.",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Job title"},
            "min_years_exp": {
                "type": "number",
                "description": "Minimum years of experience required",
            },
            "required_skills": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Required technical skills listed or implied by the JD",
            },
            "summary": {
                "type": "string",
                "description": "2-3 sentence summary of the role and ideal candidate profile",
            },
        },
        "required": ["title", "min_years_exp", "required_skills", "summary"],
    },
}

SYSTEM_PROMPT = (
    "You are a precise talent-data extraction engine. "
    "When given a resume or job description, call the provided function to extract "
    "structured information. Be accurate and comprehensive with skills — include "
    "all languages, frameworks, cloud services, and tools you can identify. "
    "For years_exp, calculate from the earliest job start date to present (2026). "
    "Do not invent information not present in the text."
)


def extract_candidate(resume_text: str) -> CandidateProfile:
    """
    Send a resume through GPT-4o-mini with function calling to extract a CandidateProfile.
    Returns a validated Pydantic model.
    """
    client = _get_client()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=1024,
        tools=[{"type": "function", "function": CANDIDATE_FUNCTION}],
        tool_choice={"type": "function", "function": {"name": "extract_candidate_profile"}},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract structured information from this resume:\n\n{resume_text}"},
        ],
    )

    # function call is guaranteed by tool_choice
    tool_call = response.choices[0].message.tool_calls[0]
    data = json.loads(tool_call.function.arguments)
    return CandidateProfile(**data)


def extract_job(jd_text: str) -> JobProfile:
    """
    Send a job description through GPT-4o-mini with function calling to extract a JobProfile.
    Returns a validated Pydantic model.
    """
    client = _get_client()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=1024,
        tools=[{"type": "function", "function": JOB_FUNCTION}],
        tool_choice={"type": "function", "function": {"name": "extract_job_profile"}},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract structured information from this job description:\n\n{jd_text}"},
        ],
    )

    tool_call = response.choices[0].message.tool_calls[0]
    data = json.loads(tool_call.function.arguments)
    return JobProfile(**data)
