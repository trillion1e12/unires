import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional

_metrics_logger: Optional[logging.Logger] = None


class MetricsFilter(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith("METRICS|")


def setup_logger(
    log_dir: str = "logs",
    log_file: str = "training.log",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    console_level: str = "INFO",
    file_level: str = "DEBUG",
) -> logging.Logger:
    global _metrics_logger

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    log_path = os.path.join(log_dir, log_file)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=max_bytes, backupCount=backup_count
    )
    file_handler.setLevel(getattr(logging, file_level.upper()))
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, console_level.upper()))
    console_handler.addFilter(MetricsFilter())
    console_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    root.addHandler(console_handler)

    _metrics_logger = logging.getLogger("metrics")
    _metrics_logger.propagate = False
    metrics_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=max_bytes, backupCount=backup_count
    )
    metrics_handler.setLevel(logging.DEBUG)
    metrics_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    metrics_handler.setFormatter(metrics_formatter)
    _metrics_logger.addHandler(metrics_handler)

    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_metrics(**kwargs) -> None:
    if _metrics_logger:
        parts = "|".join(f"{k}={v}" for k, v in kwargs.items())
        _metrics_logger.info(f"METRICS|{parts}")


def get_log_path(log_dir: str = "logs", log_file: str = "training.log") -> str:
    return os.path.join(log_dir, log_file)
