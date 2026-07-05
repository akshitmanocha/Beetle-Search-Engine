"""Shared pytest configuration: make the repository root importable.

Lets tests do ``from src.config import CONFIG`` and ``from eval.metrics import ...``
regardless of the working directory pytest is invoked from.
"""

import os
# faiss-cpu and torch each ship their own OpenMP (libomp) runtime. Importing both
# in one process on macOS segfaults / aborts when several test modules touch
# faiss and torch in sequence. The empirically-verified fix needs BOTH of:
#   * OMP_NUM_THREADS=1        -- single OpenMP thread, no cross-runtime race
#   * KMP_DUPLICATE_LIB_OK     -- tolerate the duplicate libomp load
# Either one alone still aborts on the full suite; together the suite is green.
# These MUST be set before faiss/torch are first imported, hence at the very top
# of conftest, before any other import below.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
