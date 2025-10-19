import os
import logging
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt, before_sleep_log
from langchain_openai import ChatOpenAI

# Attempt to import OpenAI base error (works across openai versions); fallback to Exception
try:
    from openai import OpenAIError as _OpenAIError
except Exception:
    class _OpenAIError(Exception): 
        pass


# Load environment variables
dotenv_path = os.getenv("DOTENV_PATH")
if dotenv_path:
    load_dotenv(dotenv_path)
# load_dotenv(os.getenv("DOTENV_PATH", ".env"))


def _setup_logger() -> logging.Logger:
    """Configure and return module logger with console and rotating file handlers."""
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    if log_level not in valid_levels:
        log_level = "INFO"
    logger = logging.getLogger("redactor")
    logger.setLevel(log_level)

    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)

    log_file = os.getenv("LOG_FILE", "logs/redactor.log")
    try:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    except Exception as e: 
        logger.warning(f"File logging disabled due to error: {e}")

    return logger


rdc_logger = _setup_logger()


def _require_env(var_name: str) -> str:
    """Fetch required environment variable or raise RuntimeError with guidance."""
    val = os.getenv(var_name)
    if not val:
        raise RuntimeError(
            f"Missing required environment variable: {var_name}. "
            f"Ensure your .env or deployment configuration sets it."
        )
    return val


def _read_text_file(path: Path, *, encoding: str = "utf-8", max_bytes: int = 2 * 1024 * 1024) -> str:
    """Read text file safely with size guard."""
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    size = path.stat().st_size
    if size == 0:
        raise ValueError("Input file is empty.")
    if size > max_bytes:
        raise ValueError(f"Input file is too large: {size} bytes (limit {max_bytes} bytes).")
    with path.open("r", encoding=encoding, errors="replace") as fh:
        return fh.read()


def load_llm() -> ChatOpenAI:
    """Initialise and return a ChatOpenAI client configured from environment."""
    try:
        _require_env("OPENAI_API_KEY")
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))
        top_p = float(os.getenv("OPENAI_TOP_P", "1"))
        llm = ChatOpenAI(model=model_name, temperature=temperature, top_p=top_p, max_retries=0)
        return llm
    except Exception as llm_e:
        rdc_logger.critical(f"Failed to initialize LLM: {llm_e}", exc_info=True)
        raise RuntimeError(f"Failed to initialize LLM: {llm_e}") from llm_e


@retry(
    wait=wait_random_exponential(min=1, max=10),
    stop=stop_after_attempt(5),
    reraise=True,
    before_sleep=before_sleep_log(rdc_logger, logging.WARNING),
)
def invoke_llm_with_retry(llm: ChatOpenAI, msg: Any) -> str:
    """Invoke the LLM with retries and return content as string."""
    response = llm.invoke(msg)
    content: Optional[str] = getattr(response, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("LLM returned empty response.")
    return content


def redaction_run(resume_path: str) -> str:
    """Run redaction for the given resume file path and return redacted content."""
    from prompts import redaction_prompt

    prompt = redaction_prompt()
    llm = load_llm()

    try:
        path_obj = Path(resume_path).expanduser().resolve()
        resume_text = _read_text_file(path_obj)
        rdc_logger.info("Successfully accessed the resume file")

        msg = prompt.format_messages(input=resume_text)
        llm_response = invoke_llm_with_retry(llm=llm, msg=msg)
        rdc_logger.info("Redaction completed successfully.")
        return llm_response
    except _OpenAIError as openai_error:
        rdc_logger.error(f"OpenAI error: {openai_error}", exc_info=True)
        raise
    except (FileNotFoundError, ValueError) as io_err:
        rdc_logger.error(f"Input error: {io_err}")
        raise
    except Exception as e:
        rdc_logger.exception(f"Unexpected error during redaction: {e}")
        raise
