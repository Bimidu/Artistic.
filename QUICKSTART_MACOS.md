# Quick Start Guide for MacBook Pro

## Hardware Acceleration Setup for MacBook Pro 15" 2018 (Radeon Pro 555X)

This guide will help you set up and run the Artistic syntactic/semantic analysis locally with hardware optimization.

---

## 📋 Prerequisites

- MacBook Pro 15" 2018 with Radeon Pro 555X
- macOS 10.14 or later
- Python 3.8+
- Homebrew (recommended)

---

## 🚀 Quick Start (3 Steps)

### Step 1: Run Hardware Setup Script

```bash
cd /Users/user/PycharmProjects/Artistic.

# Make script executable (if not already)
chmod +x setup_macos_gpu.sh

# Run setup script
./setup_macos_gpu.sh
```

This will:
- Install optimized ML libraries
- Configure 8-core CPU parallelization
- Set up environment variables
- Test hardware acceleration

### Step 2: Activate Environment

```bash
source .venv/bin/activate
source .env.local  # Load optimized settings
```

### Step 3: Run Jupyter Notebook

```bash
jupyter notebook Artistic_Local_Analysis.ipynb
```

Then click "Cell" → "Run All" in Jupyter

---

## 🔧 Manual Setup (Alternative)

### 1. Activate Virtual Environment

```bash
cd /Users/user/PycharmProjects/Artistic.
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
# Core ML libraries with optimizations
pip install --upgrade pip

# Data processing
pip install pandas numpy scipy

# NLP libraries
pip install nltk spacy textstat
python -m spacy download en_core_web_sm

# ML frameworks (optimized for CPU)
pip install scikit-learn
pip install xgboost
pip install lightgbm
pip install imbalanced-learn

# Visualization
pip install matplotlib seaborn plotly

# Jupyter
pip install jupyter jupyterlab ipywidgets
jupyter nbextension enable --py widgetsnbextension

# Utilities
pip install psutil python-dotenv tqdm joblib
```

### 3. Download NLTK Data

```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

### 4. Set Environment Variables

```bash
# For optimal CPU performance (8 cores)
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8
```

---

## 💻 Running the Analysis

### Option 1: Jupyter Notebook (Recommended)

```bash
jupyter notebook Artistic_Local_Analysis.ipynb
```

Features:
- Interactive visualizations
- Step-by-step execution
- Easy debugging
- Save/resume work

### Option 2: JupyterLab (Advanced)

```bash
jupyter lab
```

### Option 3: Run Test Script

```bash
python test_syntactic_semantic.py
```

---

## ⚡ Performance Optimization

### CPU Optimization (Already Applied)

The setup script configures:
- **8-thread parallelization** for all ML libraries
- **Optimized histogram method** for XGBoost
- **Column-wise optimization** for LightGBM
- **Intel MKL** acceleration for NumPy/scikit-learn

### Expected Performance

With your MacBook Pro 2018:
- **Feature extraction**: ~5-10 files/second
- **Model training** (5 models): ~2-5 minutes
- **Total pipeline** (100 files): ~10-15 minutes

### GPU Acceleration (Limited)

**Note:** The Radeon Pro 555X is primarily used by macOS for display acceleration. For ML workloads:

- ✅ **CPU**: Fully optimized with 8-core parallelization
- ⚠️ **GPU**: Limited support (macOS doesn't expose AMD GPUs for ML)
- 💡 **Tip**: For best performance, close other applications

### Monitor Resources

```bash
# In a separate terminal
python -c "import psutil; import time;
while True:
    print(f'CPU: {psutil.cpu_percent()}% | RAM: {psutil.virtual_memory().percent}%');
    time.sleep(2)"
```

---

## 📊 Running Examples

### Test with Small Dataset

```python
# In Jupyter notebook or Python
from pathlib import Path
from src.parsers.chat_parser import CHATParser
from src.features.feature_extractor import FeatureExtractor

# Parse a single file
parser = CHATParser()
transcript = parser.parse_file("data/asdbank_eigsti/Eigsti/ASD/Brett/060819.cha")

# Extract features
extractor = FeatureExtractor(categories=['syntactic_semantic'])
features = extractor.extract_from_transcript(transcript)

print(f"Extracted {len(features.features)} features")
```

### Process Full Dataset

See `Artistic_Local_Analysis.ipynb` for complete pipeline.

---

## 🐛 Troubleshooting

### Issue: "spaCy model not found"

```bash
python -m spacy download en_core_web_sm
```

### Issue: "NLTK data not found"

```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### Issue: "Module not found"

Ensure you're in the project directory and virtual environment is activated:

```bash
cd /Users/user/PycharmProjects/Artistic.
source .venv/bin/activate
pip install -e .
```

### Issue: Jupyter kernel crashes

Reduce memory usage:
- Process fewer files at once
- Close other applications
- Restart Jupyter kernel

### Issue: Slow performance

Check CPU usage:
```bash
top -o cpu
```

If CPU usage is low, verify environment variables:
```bash
echo $OMP_NUM_THREADS  # Should be 8
```

---

## 📁 Output Files

After running the notebook, you'll find:

```
output/
├── syntactic_semantic_features.csv      # Extracted features
├── model_evaluation_results.csv         # Model performance
├── feature_importance.csv               # Feature rankings
├── best_model_*.pkl                     # Trained model
└── preprocessor.pkl                     # Preprocessing pipeline
```

---

## 🎯 What Gets Optimized

### Hardware Level
- ✅ 8-core CPU parallelization
- ✅ SIMD vectorization (via NumPy/MKL)
- ✅ Memory-efficient processing
- ⚠️ GPU (limited macOS support for AMD)

### Software Level
- ✅ Optimized tree methods (XGBoost/LightGBM)
- ✅ Multi-threaded scikit-learn
- ✅ Batch processing for feature extraction
- ✅ Efficient data structures (pandas/numpy)

### Algorithms
- Random Forest: Parallel tree building
- XGBoost: Histogram-based splitting
- LightGBM: Leaf-wise growth with column optimization
- SVM: Multi-core kernel computation
- Logistic: Parallel gradient descent

---

## 📚 Additional Resources

- **Project Repository**: https://github.com/Bimidu/Artistic
- **spaCy Documentation**: https://spacy.io
- **XGBoost Optimization**: https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html
- **Apple Metal**: https://developer.apple.com/metal/ (for future GPU support)

---

## ✅ Verification

Test your setup:

```bash
python macos_optimization.py
```

Expected output:
```
✓ Using Metal Performance Shaders (MPS) for PyTorch (if available)
✓ XGBoost with CPU optimizations
✓ LightGBM with CPU optimizations
✓ scikit-learn with Intel MKL
```

---

## 🎉 You're Ready!

Open the notebook and start analyzing:

```bash
jupyter notebook Artistic_Local_Analysis.ipynb
```

**Happy analyzing! 🚀**
