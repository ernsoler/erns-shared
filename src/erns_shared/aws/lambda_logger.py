import json
import logging
import os
import sys
from typing import Any


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def _resolve_level() -> int:
    raw = os.environ.get("LOG_LEVEL", "INFO").upper()
    return logging.DEBUG if raw == "DEBUG" else logging.INFO


def get_lambda_logger(name: str | None = None) -> logging.Logger:
    logger_name = name or os.environ.get("AWS_LAMBDA_FUNCTION_NAME", "lambda")
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(_resolve_level())
    return logger
