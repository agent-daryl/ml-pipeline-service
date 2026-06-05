#!/usr/bin/env python3
"""Start the FastAPI prediction service."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

if __name__ == "__main__":
    uvicorn.run(
        "src.serving.app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
