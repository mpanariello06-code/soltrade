"""Logging setup: rotating file log plus a console stream."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-22s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 3

_configured = False


def setup_logging(log_file: Path, level: str = "INFO", console: bool = True) -> logging.Logger:
    """Configure root logging once and return the project logger.

    Uses a rotating handler so ``system_log.txt`` cannot grow without bound
    during long unattended runs.
    """
    global _configured
    root = logging.getLogger()
    if _configured:
        return logging.getLogger("xauusd")

    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    try:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError as exc:  # logging must never take the process down
        print(f"WARNING: could not open log file {log_file}: {exc}", file=sys.stderr)

    if console:
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(formatter)
        root.addHandler(stream)

    # third-party noise
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

    _configured = True
    return logging.getLogger("xauusd")


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced child logger."""
    return logging.getLogger(f"xauusd.{name}")
