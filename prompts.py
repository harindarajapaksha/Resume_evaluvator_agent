from __future__ import annotations

import os
from textwrap import dedent
from typing import Literal

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

if os.environ.get("LOAD_DOTENV", "").lower() in {"1", "true", "yes"}:
    load_dotenv()

__all__ = [
    "CategoryScore",
    "Categories",
    "Evaluation",
    "EvaluationOutput",
    "resume_eveluator_prompt",
    "redaction_prompt",
]


class CategoryScore(BaseModel):
    """Score and confidence for a single evaluation category."""
    score: float = Field(..., ge=0, le=100, description="Score from 0–100")
    confidence: float = Field(..., ge=0, le=100, description="Confidence from 0–100")


class Categories(BaseModel):
    """All category scores. Keys must match exactly for downstream parsing."""
    Technical_Skills: CategoryScore
    Domain_Knowledge: CategoryScore
    Experience_Level: CategoryScore
    Tools_and_Technologies: CategoryScore
    Education_and_Certifications: CategoryScore
    Soft_Skills: CategoryScore


class Evaluation(BaseModel):
    """Overall evaluation rollups."""
    categories: Categories
    overall_match_score: float = Field(..., ge=0, le=100)
    cumulative_confidence: float = Field(..., ge=0, le=100)
    fit_classification: Literal["Strong Fit", "Moderate Fit", "Weak Fit"]


class EvaluationOutput(BaseModel):
    """Final LLM output contract."""
    evaluation: Evaluation
    summary: str = Field(
        ...,
        description="1–3 sentence summary explaining key strengths, weaknesses, and overall alignment",
    )


def resume_eveluator_prompt() -> ChatPromptTemplate:
    """
    Build the resume evaluator prompt.
    Returns a ChatPromptTemplate that instructs the model to strictly output JSON
    conforming to the EvaluationOutput schema with the exact keys as defined.
    """
    weights_section = dedent(
        """
        Use these exact JSON keys and weights:
        - Technical_Skills (30%)
        - Domain_Knowledge (20%)
        - Experience_Level (20%)
        - Tools_and_Technologies (15%)
        - Education_and_Certifications (10%)
        - Soft_Skills (5%)
        """
    ).strip()

    template = dedent(
        f"""
        You are an expert recruiter and data-driven talent evaluator.
        Compare the candidate resume to the job description and produce a structured, numeric evaluation.

        {weights_section}

        For each category provide:
        - score: 0–100 (100 = perfect alignment; 50 = partial; 0 = no evidence)
        - confidence: 0–100 (100 = very high confidence; 50 = partial; 0 = no confidence)

        Then compute:
        - overall_match_score: 0–100
        - cumulative_confidence: 0–100
        - fit_classification: one of "Strong Fit", "Moderate Fit", "Weak Fit"

        Provide a 1–3 sentence summary of strengths, weaknesses, and overall alignment.

        Output requirements:
        - Output ONLY valid JSON matching this structure with these exact keys:
          evaluation.categories.Technical_Skills.score
          evaluation.categories.Technical_Skills.confidence
          evaluation.categories.Domain_Knowledge.score
          evaluation.categories.Domain_Knowledge.confidence
          evaluation.categories.Experience_Level.score
          evaluation.categories.Experience_Level.confidence
          evaluation.categories.Tools_and_Technologies.score
          evaluation.categories.Tools_and_Technologies.confidence
          evaluation.categories.Education_and_Certifications.score
          evaluation.categories.Education_and_Certifications.confidence
          evaluation.categories.Soft_Skills.score
          evaluation.categories.Soft_Skills.confidence
          evaluation.overall_match_score
          evaluation.cumulative_confidence
          evaluation.fit_classification
          summary
        - Do not include any additional fields, comments, or explanations.

        Candidate Resume:
        {{resume}}

        Job Description:
        {{job_description}}
        """
    ).strip()

    return ChatPromptTemplate.from_template(template)


def _redaction_system_instructions() -> str:
    """
    Private builder for safe, deterministic PII redaction instructions.
    """
    return dedent(
        """
        You are a meticulous PII redactor.
        Task: Replace any personally identifiable information with the placeholder REDACTION_TOKEN.
        Categories to redact include (non-exhaustive):
        - Person names
        - Email addresses
        - Phone numbers
        - URLs and social profiles
        - Gender, religion, ethnicity, marital status, age
        - Citizenship, visa/immigration status
        - Physical addresses and specific locations (street, city, suburb, postcode)

        Rules:
        - Preserve original whitespace, line breaks, and all non-PII text exactly.
        - Do not add, remove, or reorder lines.
        - Do not redact job titles, generic role names, company names, or technologies unless they reveal PII.
        - Apply the same REDACTION_TOKEN consistently for each occurrence.
        - Return only the redacted text with no explanations.
        """
    ).strip()


def redaction_prompt() -> ChatPromptTemplate:
    """
    Build the PII redaction prompt.
    Returns a ChatPromptTemplate that takes a single input variable: 'input'.
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", _redaction_system_instructions()),
            ("human", "{input}"),
        ]
    )
