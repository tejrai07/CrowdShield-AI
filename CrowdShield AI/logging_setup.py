"""Structured logging setup for CrowdShield.

Deliberately minimal per project instructions — no log aggregation,
no external logging services. Just consistent, readable structured logs.
"""
from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure root logger once. Safe to call multiple times (idempotent)."""
    logger = logging.getLogger("crowdshield")

    if logger.handlers:
        # Already configured — avoid duplicate handlers on repeated calls.
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    return logger
