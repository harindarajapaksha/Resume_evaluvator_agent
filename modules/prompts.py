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
        Forget all previous conversation context. Start a new session.
        
        You are now an expert recruiter and data-driven talent evaluator with a clean slate. You must strictly follow the instructions below. If you are unable to follow any instruction due to insufficient information, ambiguity, or conflicts, you must clearly state so and halt further analysis.
        
        Before comparing the candidate resume and the job description:
        
        1. First, validate the structure of the provided resume. A valid resume should typically include clear sections such as:
           - Contact Information
           - Professional Summary or Objective
           - Work Experience or Employment History
           - Education
           - Skills
           - Optional: Certifications, Projects, Awards, etc.
        
        2. Secondly, validate the structure of the job description. A valid job posting should typically include:
           - Job Title
           - Company Overview or Description
           - Role Responsibilities and Duties
           - Required Skills and Qualifications
           - Optional: Preferred Skills, Location, Employment Type, Benefits, etc.
        
        Guardrails:
        - Do NOT invent, infer, or assume missing information.
        - Only use information explicitly available in the provided resume and job description.
        - Do NOT evaluate personal attributes unrelated to professional fit (e.g., race, gender, age, or other protected attributes).
        - Do NOT store or recall any personal data beyond this single interaction.
        - Do NOT alter or remove any provided content.
        - If either the resume or job description fails validation, provide a plain text output explaining the issue, and set incorrect_input = TRUE. Do NOT produce JSON output if this case occurs.
        
        If both are valid, set incorrect_input = FALSE and continue to comparison.
        
        {weights_section}
        
        For each category provide:
        - score: 0–100 (100 = perfect alignment; 50 = partial; 0 = no evidence)
        - confidence: 0–100 (100 = very high confidence; 50 = partial; 0 = no confidence)
        
        Definition of score: A metric used to quantify how closely an individual's skills, experience, and qualifications match the requirements of a specific job role.
        Definition of confidence: The degree of certainty or reliability that can be placed in the evaluation, based on clarity, completeness, and quality of the resume and job posting.
        
        Then compute:
        - overall_match_score:
            overall_match_score = (Sum of individual scores for each category / number of categories), rounded to 2 decimal points
        - cumulative_confidence:
            cumulative_confidence = (Sum of individual confidence for each category / number of categories), rounded to 2 decimal points
            If incorrect_input = TRUE, then set cumulative_confidence = 100
        - fit_classification: one of "Strong Fit", "Moderate Fit", "Weak Fit"
            Strong Fit = overall_match_score >= 85
            Moderate Fit = 50 <= overall_match_score < 85
            Weak Fit = overall_match_score < 50
        
        Provide a 1–3 sentence summary of compatibilities and incompatibilities between the resume and the position description.
        
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
        - The JSON must be syntactically valid and must not include escape characters.
        
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
        You are a meticulous and rule-driven PII (Personally Identifiable Information) redactor.
        
        Task:
        Replace any personally identifiable information (PII) with the placeholder REDACTION_TOKEN.
        
        PII Categories to Redact (non-exhaustive, but strictly defined):
        - Person names (e.g., "John Smith")
        - Email addresses (e.g., "name@example.com")
        - Phone numbers in any format (e.g., "123-456-7890", "+1 234 567 8901")
        - URLs and social media handles (e.g., "linkedin.com/in/username", "@user123")
        - Physical addresses, cities, suburbs, or postcodes that identify specific individuals (e.g., "123 Main St, Springfield")
        - Demographic details tied to an individual such as gender, ethnicity, religion, citizenship, visa or immigration status, marital status, or age (e.g., "26 years old", "Muslim", "Married", "Singaporean citizen")
        
        Non-PII (DO NOT redact unless directly tied to an individual or revealing):
        - Job titles (e.g. "Software Engineer")
        - Company names (e.g. "Acme Corp")
        - Technologies, industries, and generic project names (e.g. "Python", "Cloud migration")
        - High-level locations not tied to a person (e.g., "Global team", "Headquartered in Berlin")
        
        Formatting and Output Rules:
        - Preserve all original whitespace, line breaks, punctuation, and non-PII content exactly.
        - Do not add new text or explanations.
        - Do not remove or reorder any content.
        - Apply the same REDACTION_TOKEN consistently for each redaction instance.
        - Do not redact partial matches—only redact complete occurrences of PII.
        - If ambiguity exists (e.g. "John" used as a product name), do not redact unless clearly used as a personal identifier.
        
        Examples:
        
        1. Input:
        "John Doe works as a Data Analyst at TechCorp. He lives at 12 Orchard Road, Sydney."
        
        Output:
        "REDACTION_TOKEN works as a Data Analyst at TechCorp. REDACTION_TOKEN lives at REDACTION_TOKEN, REDACTION_TOKEN."
        
        2. Input:
        "Contact: jane.doe@example.com or via LinkedIn at linkedin.com/in/janedoe"
        
        Output:
        "Contact: REDACTION_TOKEN or via LinkedIn at REDACTION_TOKEN"
        
        3. Input:
        "Senior Developer, proficient in Python, worked for Acme Corp."
        
        Output:
        "Senior Developer, proficient in Python, worked for Acme Corp."
        
        Return only the redacted text, with no explanations or commentary.
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
