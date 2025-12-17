#!/usr/bin/env python
"""
Test script to verify local setup and fix any issues
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set environment variables for optimal CPU performance
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'
os.environ['VECLIB_MAXIMUM_THREADS'] = '8'

print("="*70)
print("ARTISTIC LOCAL SETUP TEST")
print("="*70)

# Test 1: System information
print("\n[1/10] Checking system information...")
try:
    import platform
    import psutil

    print(f"  ✓ Platform: {platform.platform()}")
    print(f"  ✓ Python: {sys.version.split()[0]}")
    print(f"  ✓ CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    print(f"  ✓ RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
except Exception as e:
    print(f"  ✗ Error: {e}")

# Test 2: Core libraries
print("\n[2/10] Testing core libraries...")
try:
    import numpy as np
    import pandas as pd
    import scipy
    print(f"  ✓ NumPy: {np.__version__}")
    print(f"  ✓ Pandas: {pd.__version__}")
    print(f"  ✓ SciPy: {scipy.__version__}")
except ImportError as e:
    print(f"  ✗ Missing library: {e}")
    print("  Run: pip install numpy pandas scipy")

# Test 3: NLP libraries
print("\n[3/10] Testing NLP libraries...")
try:
    import nltk
    print(f"  ✓ NLTK: {nltk.__version__}")

    # Test NLTK data
    try:
        from nltk.corpus import wordnet
        wordnet.synsets('test')
        print("  ✓ NLTK WordNet data available")
    except LookupError:
        print("  ⚠ NLTK WordNet data missing")
        print("    Run: python -c \"import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')\"")

except ImportError:
    print("  ✗ NLTK not installed")
    print("  Run: pip install nltk")

try:
    import spacy
    print(f"  ✓ spaCy: {spacy.__version__}")

    # Test spaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
        print("  ✓ spaCy model 'en_core_web_sm' available")
    except OSError:
        print("  ⚠ spaCy model not found")
        print("    Run: python -m spacy download en_core_web_sm")

except ImportError:
    print("  ✗ spaCy not installed")
    print("  Run: pip install spacy")

try:
    import textstat
    print(f"  ✓ textstat: {textstat.__version__}")
except ImportError:
    print("  ✗ textstat not installed")
    print("  Run: pip install textstat")

# Test 4: ML libraries
print("\n[4/10] Testing ML libraries...")
try:
    import sklearn
    print(f"  ✓ scikit-learn: {sklearn.__version__}")
except ImportError:
    print("  ✗ scikit-learn not installed")
    print("  Run: pip install scikit-learn")

try:
    import xgboost as xgb
    print(f"  ✓ XGBoost: {xgb.__version__}")
except ImportError:
    print("  ✗ XGBoost not installed")
    print("  Run: pip install xgboost")

try:
    import lightgbm as lgb
    print(f"  ✓ LightGBM: {lgb.__version__}")
except ImportError:
    print("  ✗ LightGBM not installed")
    print("  Run: pip install lightgbm")

try:
    import imblearn
    print(f"  ✓ imbalanced-learn: {imblearn.__version__}")
except ImportError:
    print("  ⚠ imbalanced-learn not installed (optional)")
    print("  Run: pip install imbalanced-learn")

# Test 5: Visualization libraries
print("\n[5/10] Testing visualization libraries...")
try:
    import matplotlib
    print(f"  ✓ matplotlib: {matplotlib.__version__}")
except ImportError:
    print("  ✗ matplotlib not installed")
    print("  Run: pip install matplotlib")

try:
    import seaborn as sns
    print(f"  ✓ seaborn: {sns.__version__}")
except ImportError:
    print("  ✗ seaborn not installed")
    print("  Run: pip install seaborn")

try:
    import plotly
    print(f"  ✓ plotly: {plotly.__version__}")
except ImportError:
    print("  ⚠ plotly not installed (optional)")
    print("  Run: pip install plotly")

# Test 6: Jupyter
print("\n[6/10] Testing Jupyter...")
try:
    import jupyter
    print(f"  ✓ Jupyter installed")
    import notebook
    print(f"  ✓ Jupyter Notebook: {notebook.__version__}")
except ImportError:
    print("  ✗ Jupyter not installed")
    print("  Run: pip install jupyter notebook")

# Test 7: Project modules
print("\n[7/10] Testing project modules...")
try:
    from src.parsers.chat_parser import CHATParser
    print("  ✓ CHATParser module")
except ImportError as e:
    print(f"  ✗ CHATParser import failed: {e}")

try:
    from src.features.feature_extractor import FeatureExtractor
    print("  ✓ FeatureExtractor module")
except ImportError as e:
    print(f"  ✗ FeatureExtractor import failed: {e}")

try:
    from src.features.syntactic_semantic import SyntacticSemanticFeatures
    print("  ✓ SyntacticSemanticFeatures module")
except ImportError as e:
    print(f"  ✗ SyntacticSemanticFeatures import failed: {e}")

try:
    from src.models.syntactic_semantic import SyntacticSemanticPreprocessor, SyntacticSemanticTrainer
    print("  ✓ Preprocessor and Trainer modules")
except ImportError as e:
    print(f"  ✗ Model modules import failed: {e}")

try:
    from config import config
    print("  ✓ Config module")
except ImportError as e:
    print(f"  ✗ Config import failed: {e}")

# Test 8: Data directory
print("\n[8/10] Checking data directory...")
data_dir = Path("data")
if data_dir.exists():
    cha_files = list(data_dir.rglob("*.cha"))
    if cha_files:
        print(f"  ✓ Data directory found: {data_dir}")
        print(f"  ✓ Found {len(cha_files)} .cha files")

        # List datasets
        datasets = set([f.parts[1] for f in cha_files if len(f.parts) > 1])
        for ds in sorted(datasets):
            count = len([f for f in cha_files if len(f.parts) > 1 and f.parts[1] == ds])
            print(f"    - {ds}: {count} files")
    else:
        print(f"  ⚠ Data directory exists but no .cha files found")
else:
    print(f"  ⚠ Data directory not found: {data_dir}")
    print("    Please ensure CHAT transcript files are in 'data/' directory")

# Test 9: Output directory
print("\n[9/10] Checking output directory...")
output_dir = Path("output")
if not output_dir.exists():
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ✓ Created output directory: {output_dir}")
else:
    print(f"  ✓ Output directory exists: {output_dir}")

# Test 10: Quick functionality test
print("\n[10/10] Testing feature extraction (if data available)...")
try:
    data_dir = Path("data")
    cha_files = list(data_dir.rglob("*.cha"))

    if cha_files:
        # Test with first file
        test_file = cha_files[0]
        print(f"  Testing with: {test_file.name}")

        from src.parsers.chat_parser import CHATParser
        from src.features.feature_extractor import FeatureExtractor

        parser = CHATParser()
        transcript = parser.parse_file(test_file)
        print(f"  ✓ Parsed transcript: {transcript.total_utterances} utterances")

        extractor = FeatureExtractor(categories=['syntactic_semantic'])
        feature_set = extractor.extract_from_transcript(transcript)
        print(f"  ✓ Extracted {len(feature_set.features)} features")

        print(f"\n  Sample features:")
        for i, (name, value) in enumerate(list(feature_set.features.items())[:5]):
            print(f"    - {name}: {value:.4f}")

    else:
        print("  ⚠ No .cha files found, skipping functional test")

except Exception as e:
    print(f"  ✗ Functional test failed: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "="*70)
print("SETUP TEST COMPLETE")
print("="*70)

print("\nNext steps:")
print("  1. Fix any missing dependencies shown above")
print("  2. Ensure .cha files are in the 'data/' directory")
print("  3. Run: jupyter notebook Artistic_Local_Analysis.ipynb")
print("\nFor full setup, run: ./setup_macos_gpu.sh")
print("="*70)
