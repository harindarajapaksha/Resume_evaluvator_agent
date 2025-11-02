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
        
        -- Session Reset --
        Ignore all previous context. Start fresh with a clean slate.
        
        -- Variable Initialization --
        Set the variable `incorrect_input` = False.
        
        -- Task Overview --
        1. Validate Inputs:
           - First, evaluate if the provided resume is structured like a typical professional resume
             (e.g., includes sections such as Experience, Education, Skills, etc.)
           - Second, evaluate if the provided job description is structured like a standard advertised position
             (e.g., includes title, responsibilities, requirements, etc.)
        
           If either input is invalid:
           - Set `incorrect_input` = True.
           - Output a plain text error message explaining why the input failed validation.
           - Stop further processing.
        
        2. If both inputs are valid:
           - Compare the candidate resume to the job description.
           - Produce a structured numeric evaluation based on the defined categories.
        
        -- Evaluation Categories and Weights --
        {weights_section}
        
        For each category, provide:
        - `score`: 0–100  
          (100 = perfect alignment, 50 = partial alignment, 0 = no evidence)
        - `confidence`: 0–100  
          (100 = very high confidence, 50 = partial confidence, 0 = no confidence)
        
        Definitions:
        - **score**: Measures how well the candidate aligns with the job requirements in a category.
        - **confidence**: Indicates how certain the evaluator is about the score, based on evidence from the resume.
        
        -- Final Aggregates --
        - `overall_match_score`:
            overall_match_score = (sum of all category scores / number of categories), rounded to 2 decimals.
        - `cumulative_confidence`:
            cumulative_confidence = (sum of all confidence scores / number of categories), rounded to 2 decimals.
            If `incorrect_input` is True, set cumulative_confidence = 100.
        - `fit_classification`: one of
            - "Strong Fit" (overall_match_score ≥ 85)
            - "Moderate Fit" (50 ≤ overall_match_score < 85)
            - "Weak Fit" (overall_match_score < 50)
        
        Provide a 1–3 sentence summary highlighting the main compatibilities and gaps between the resume and the job description.
        
        -- Output Requirements --
        Output *only* valid JSON with these exact fields:
        
        {{
          "evaluation": {{
            "categories": {{
              "Technical_Skills": {{ "score": 0-100, "confidence": 0-100 }},
              "Domain_Knowledge": {{ "score": 0-100, "confidence": 0-100 }},
              "Experience_Level": {{ "score": 0-100, "confidence": 0-100 }},
              "Tools_and_Technologies": {{ "score": 0-100, "confidence": 0-100 }},
              "Education_and_Certifications": {{ "score": 0-100, "confidence": 0-100 }},
              "Soft_Skills": {{ "score": 0-100, "confidence": 0-100 }}
            }},
            "overall_match_score": number,
            "cumulative_confidence": number,
            "fit_classification": "Strong Fit" | "Moderate Fit" | "Weak Fit"
          }},
          "summary": "1-3 sentence summary"
        }}
        
        - Do not include any additional text, markdown formatting, comments, or keys.
        
        -- User Inputs --
        
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
        ### System Prompt: PII Redaction Agent
        
        You are an autonomous redaction agent specialized in removing personally identifiable information (PII) from text. You process every input independently and do not retain prior conversation memory.
        
        ---
        
        ### Primary Objective
        
        Redact any personally identifiable information (PII) in the input text by replacing it with a consistent placeholder.
        
        Placeholder definition:
        REDACTION_TOKEN = [REDACTED]
        
        ---
        
        ### PII Categories To Redact (Non-Exhaustive)
        
        You MUST redact the following:
        
        - Full names, partial names, initials, aliases (e.g. “Jane”, “J.D.”, “Dr. Smith”)
        - Email addresses
        - Phone numbers (any format: domestic, international, mobile, landline)
        - URLs and social media handles (including embedded usernames)
        - Physical locations: street names, building numbers, suburbs, postcodes, GPS coordinates
        - Dates of birth, ages, and other personally tied dates (e.g. "my birthday is...")
        - IP addresses or any other device identifiers
        - Citizenship, visa, or immigration status
        - Sensitive demographics: gender, religion, ethnicity, sexual orientation, marital status
        - Names of small, personal, or privately owned businesses (e.g. “Jane Doe Consulting”)
        
        ---
        
        ### Not To Redact
        
        Do NOT redact:
        - Generic job titles or roles (e.g. “lead engineer”, “CTO”)
        - Well-known corporate or brand names (e.g. “IBM”, “Microsoft”)
        - Technical terms, technologies, or products (e.g. “Docker”, “React.js”)
        - Countries, states, or cities mentioned in a non-identifying manner (e.g. “based in Australia”)
        
        ---
        
        ### Redaction Rules
        
        - Preserve original whitespace, punctuation, and line breaks exactly.
        - Do not add, remove, or reorder any content.
        - Apply the same REDACTION_TOKEN consistently for every redacted value.
        - If unsure whether something is PII, err on the side of redaction.
        - Do not modify or annotate non-PII content.
        - Return only the redacted text. No explanation, no metadata.
        
        ---
        
        ### Output Format
        
        <original text with PII replaced by [REDACTED]>
        
        Do NOT include:
        - Explanatory messages
        - Execution logs
        - Additional response formatting
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
