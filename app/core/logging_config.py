"""Centralised logging configuration for the claims processing pipeline.

Call ``get_logger(__name__)`` in any module to get a named logger that writes to both
the rotating log file (``logs/claims_processor.log``) and the console.

Log format (every line):
    TIMESTAMP | LEVEL    | module:function:line | message

This gives enough context to trace any log line back to its exact source without an IDE.
"""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
_LOG_FILE = _LOG_DIR / "claims_processor.log"
_FMT = "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _configure() -> None:
    global _configured
    if _configured:
        return
    _configured = True

    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(_FMT, datefmt=_DATE_FMT)

    # Rotating file: 5 MB per file, keep 3 backups.
    # File-only — no StreamHandler so logs never appear in the Streamlit UI.
    file_handler = logging.handlers.RotatingFileHandler(
        _LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    root = logging.getLogger("app")
    root.setLevel(logging.DEBUG)
    if not root.handlers:
        root.addHandler(file_handler)

    eval_logger = logging.getLogger("eval")
    eval_logger.setLevel(logging.DEBUG)
    if not eval_logger.handlers:
        eval_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger.  Always pass ``__name__`` so the module path appears in logs."""
    _configure()
    return logging.getLogger(name)
