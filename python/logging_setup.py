import logging
from typing import Optional


def setup_logging(level: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger("videoforge")
    resolved = (level or "info").upper()
    numeric = getattr(logging, resolved, logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        logger.addHandler(handler)

    logger.setLevel(numeric)
    logger.propagate = False
    return logger

