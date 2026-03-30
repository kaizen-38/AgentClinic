"""
Structured JSON logging for the medical_agent pipeline.
Every module emits JSON-formatted log records compatible with log aggregators.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }
        # Attach structured 'data' dict if provided via extra={"data": {...}}
        if hasattr(record, "data") and record.data:
            log_obj["data"] = record.data
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Human-readable formatter for development."""

    FMT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"

    def format(self, record: logging.LogRecord) -> str:
        formatter = logging.Formatter(self.FMT, datefmt="%H:%M:%S")
        return formatter.format(record)


def get_logger(name: str, level: str = "INFO", fmt: str = "json") -> logging.Logger:
    """
    Return a configured logger.

    Args:
        name:  Logger name (typically module name, e.g. "hypothesis_engine").
        level: Logging level string ("DEBUG", "INFO", "WARNING", "ERROR").
        fmt:   "json" (default) or "text".
    """
    logger = logging.getLogger(f"medical_agent.{name}")
    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


class PipelineLogger:
    """
    Convenience wrapper that adds turn-level context to every log call.

    Usage:
        logger = PipelineLogger("hypothesis_engine", config)
        logger.log("hypothesis_update", turn=5, data={"top": "SLE", "confidence": 0.62})
    """

    def __init__(self, module_name: str, level: str = "INFO", fmt: str = "json"):
        self._logger = get_logger(module_name, level, fmt)
        self._module = module_name

    def log(
        self,
        event: str,
        turn: int = 0,
        data: Optional[Dict[str, Any]] = None,
        level: str = "INFO",
    ) -> None:
        record_data = {"turn": turn, "event": event}
        if data:
            record_data.update(data)
        log_fn = getattr(self._logger, level.lower(), self._logger.info)
        log_fn(event, extra={"data": record_data})

    def info(self, event: str, turn: int = 0, data: Optional[Dict] = None) -> None:
        self.log(event, turn=turn, data=data, level="INFO")

    def warning(self, event: str, turn: int = 0, data: Optional[Dict] = None) -> None:
        self.log(event, turn=turn, data=data, level="WARNING")

    def error(self, event: str, turn: int = 0, data: Optional[Dict] = None) -> None:
        self.log(event, turn=turn, data=data, level="ERROR")

    def debug(self, event: str, turn: int = 0, data: Optional[Dict] = None) -> None:
        self.log(event, turn=turn, data=data, level="DEBUG")
