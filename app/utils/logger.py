import logging
import os
from typing import Optional


_configured = False


def _configure_logging(level: Optional[str] = None) -> None:
    global _configured
    if _configured:
        return
    log_level = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    _configured = True


def get_logger(name: str) -> logging.Logger:
    _configure_logging()
    return logging.getLogger(name)
