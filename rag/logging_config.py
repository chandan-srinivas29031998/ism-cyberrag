import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    logger = logging.getLogger()
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)

    # Avoid duplicate handlers in notebooks
    if not logger.handlers:
        logger.addHandler(handler)
