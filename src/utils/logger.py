"""Logger simples para treinamento e experimentos."""

import logging
from typing import Dict


def setup_logger(config: Dict) -> logging.Logger:
    logger = logging.getLogger("lane_follower")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(config.get("log_level", "INFO"))
    return logger
