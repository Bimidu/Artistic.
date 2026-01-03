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
    print("  🔮 Predictions: http://localhost:8000/predict")
    print("\n" + "="*70 + "\n")
    
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )

