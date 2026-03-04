"""
Start ASD Detection API Server

Simple script to start the FastAPI server for ASD detection predictions.

Usage:
    python run_api.py
    
Then access:
    - API docs: http://localhost:8000/docs
    - Health: http://localhost:8000/health
    - Models: http://localhost:8000/models

Author: Bimidu Gunathilake
"""

# ── macOS OpenMP segfault fix ────────────────────────────────────────────────
# On macOS, CTranslate2 (used by faster-whisper) ships Intel's libomp, which
# conflicts with the LLVM/Apple OpenMP already linked into numpy/scipy.  Having
# two OpenMP runtimes active in the same process causes a SIGSEGV (exit 139).
# KMP_DUPLICATE_LIB_OK=TRUE tells Intel's runtime to tolerate the duplicate.
# OMP_NUM_THREADS=1 prevents the runtimes from spawning competing thread pools.
# These must be set BEFORE any native libraries are imported.
# On Linux there is only one OpenMP runtime (libgomp), so this has no effect.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
# ────────────────────────────────────────────────────────────────────────────

import uvicorn
from pathlib import Path

if __name__ == "__main__":
    print("\n" + "="*70)
    print("Starting ASD Detection API Server")
    print("="*70)
    print("\nAPI Documentation will be available at:")
    print("  [BOOK] Swagger UI: http://localhost:8000/docs")
    print("  [BOOK] ReDoc: http://localhost:8000/redoc")
    print("\nEndpoints:")
    print("  [DIAGNOSIS] Health Check: http://localhost:8000/health")
    print("  [ML] List Models: http://localhost:8000/models")
    print("  [PREDICT] Predictions: http://localhost:8000/predict")
    print("\n" + "="*70 + "\n")
    
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )

