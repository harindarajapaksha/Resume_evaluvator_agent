import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Optional

from prompts import EvaluationOutput, resume_eveluator_prompt
from redactor import redaction_run
from langchain_openai import ChatOpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt


# ---------- Logging setup ----------
_VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
if _LOG_LEVEL not in _VALID_LEVELS:
    _LOG_LEVEL = "INFO"
_LOG_FILE = Path(os.getenv("LOG_FILE", "logs/resume_assesor.log"))


def _ensure_log_dir(log_file: Path) -> None:
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("resume_assesor")
    logger.setLevel(_LOG_LEVEL)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    try:
        _ensure_log_dir(_LOG_FILE)
        file_handler = logging.FileHandler(_LOG_FILE)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception:
        logger.warning("File logging disabled due to initialization error.", exc_info=True)

    return logger


assr_logger = _setup_logger()


# ---------- Helpers ----------
def _validate_file_readable(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"{description} is not a file: {path}")
    if not os.access(path, os.R_OK):
        raise PermissionError(f"{description} is not readable: {path}")


def _read_text_file(path: Path, description: str) -> str:
    _validate_file_readable(path, description)
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        raise IOError(f"Failed to read {description}: {path}") from e


def _get_model_name() -> str:
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o").strip()
    if not model_name:
        model_name = "gpt-4o"
    return model_name


# ---------- LLM ----------
@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
def invoke_llm() -> ChatOpenAI:
    """
    Initialize the OpenAI chat model with sane defaults, retrying on transient errors.
    """
    model_name = _get_model_name()
    try:
        llm = ChatOpenAI(model=model_name, temperature=0, top_p=1)
        return llm
    except Exception as e:
        assr_logger.critical(f"Failed to initialize ChatOpenAI: {e}")
        raise RuntimeError("LLM initialization failed. Check credentials and environment.") from e


# ---------- Core ----------
def resume_evaluator(resume_path: str | Path, job_description: str | Path) -> Optional[EvaluationOutput]:
    """
    Redact the resume, evaluate against the job description using an LLM,
    and print the structured JSON result. Returns the EvaluationOutput on success.
    """
    resume_path = Path(resume_path)
    job_description = Path(job_description)

    try:
        _validate_file_readable(resume_path, "Resume file")
        _validate_file_readable(job_description, "Job description file")
    except Exception as e:
        assr_logger.error(str(e))
        return None

    try:
        redactored_resume = redaction_run(resume_path=resume_path)
        assr_logger.info("Resume redaction completed.")
    except Exception as e:
        assr_logger.error(f"Resume redaction failed: {e}", exc_info=True)
        return None

    try:
        jd = _read_text_file(job_description, "Job description")
        assr_logger.info("Successfully accessed the job description file.")
    except Exception as e:
        assr_logger.error(str(e), exc_info=True)
        return None

    try:
        llm = invoke_llm()
        prompt = resume_eveluator_prompt()
        structured_model = llm.with_structured_output(EvaluationOutput)

        chain = prompt | structured_model

        result: EvaluationOutput = chain.invoke({
            "resume": redactored_resume,
            "job_description": jd
        })

        print(result.model_dump_json(indent=2))
        assr_logger.info("Resume assessment successful.")
        return result

    except Exception as e:
        assr_logger.error(f"LLM evaluation failed: {e}", exc_info=True)
        return None


# ---------- CLI ----------
def check_file_extension(filename: str) -> str:
    """
    Custom type function for argparse to validate the file extension.
    """
    filepath = Path(filename)
    if filepath.suffix.lower() != '.txt':
        assr_logger.critical("Input file format not supported")
        raise argparse.ArgumentTypeError(f"File '{filename}' must have a .txt extension")
    else:
        return filename


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume assessor")
    parser.add_argument("-r", "--resume", help="Path to the resume.txt file", required=True, type=check_file_extension)
    parser.add_argument("-p", "--position", help="Path to the position_description.txt file", required=True, type=check_file_extension)
    return parser.parse_args(argv)


def _main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = resume_evaluator(resume_path=args.resume, job_description=args.position)
        return 0 if result is not None else 1
    except KeyboardInterrupt:
        assr_logger.warning("Interrupted by user.")
        return 130
    except Exception as e:
        assr_logger.critical(f"Unhandled error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(_main())
