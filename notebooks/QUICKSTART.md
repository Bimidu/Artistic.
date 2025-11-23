# Quick Start Guide - Syntactic Semantic Model Analysis Notebook

## Important: Python Version Requirement

The notebook requires **Python 3.8+** (it uses `Union` type hints compatible with Python 3.7+).

If you previously encountered a `TypeError: unsupported operand type(s) for |: 'type' and 'type'` error, this has been **FIXED**. The issue was due to Python 3.10+ style type hints (`str | Path`) which have been updated to use `Union[str, Path]` for compatibility.

## Prerequisites

### 1. Python Environment
```bash
# Check Python version (should be 3.8+)
python --version

# If using virtual environment, activate it
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows
```

### 2. Install Dependencies

```bash
# Essential packages
pip install jupyter notebook pandas numpy matplotlib seaborn

# ML libraries
pip install scikit-learn xgboost

# NLP libraries
pip install spacy nltk textstat
python -m spacy download en_core_web_sm

# Optional (for full model training - may require additional setup)
# pip install lightgbm  # Requires libomp on macOS
```

### Note on LightGBM (macOS users)
If you encounter errors with LightGBM on macOS:
```bash
# Install libomp via Homebrew
brew install libomp

# Then install lightgbm
pip install lightgbm
```

The notebook will work fine without LightGBM - it will just skip that specific model type.

## Running the Notebook

### Option 1: Jupyter Notebook (Recommended)

```bash
# Navigate to project directory
cd /Users/user/PycharmProjects/Artistic.

# Start Jupyter Notebook
python -m jupyter notebook notebooks/syntactic_semantic_model_analysis.ipynb
```

This will:
1. Open your browser automatically
2. Load the notebook
3. You can now run cells using `Shift + Enter`

### Option 2: Jupyter Lab

```bash
python -m jupyter lab
```

Then navigate to `notebooks/` folder and open the `.ipynb` file.

### Option 3: VS Code

If you're using VS Code with the Python extension:
1. The `.ipynb` file should already be open
2. Select Python kernel (Python 3.8+)
3. Click "Run All" or run cells individually

## Fixing the Import Error

**The type annotation errors have been fixed!** Changes made:

1. **chat_parser.py**: Updated `str | Path` → `Union[str, Path]`
2. **feature_extractor.py**: Updated type annotations
3. **model_trainer.py**: Updated type annotations
4. **preprocessor.py**: Updated type annotations

### If you still see the error:

**Restart your Jupyter kernel:**
1. In Jupyter: `Kernel` → `Restart`
2. Then run all cells again

The fixes have been applied to the source code, but Jupyter caches imported modules.

## What the Notebook Does

The notebook provides:

### 1. Documentation
- All 27 syntactic & semantic features explained
- Feature extraction pipeline
- Model architecture details
- Dataset information

### 2. Demonstrations
- Feature extraction from CHAT files
- Data preprocessing steps
- Multiple ML model training
- Performance evaluation

### 3. Visualizations
- Feature distributions by diagnosis
- Correlation heatmaps
- Feature importance charts
- Model performance comparisons
- Confusion matrices

### 4. Code Examples
- Complete end-to-end pipeline
- Feature extraction code
- Model training examples
- Prediction examples

## Running with Your Own Data

To use the notebook with your actual data:

1. **Ensure data is in place:**
   ```
   Artistic./
   ├── data/
   │   ├── asdbank_aac/AAC/*.cha
   │   ├── asdbank_eigsti/*.cha
   │   └── ...
   ```

2. **Update paths in cells:**
   - Most cells use `config.paths.data_dir` which reads from `config.py`
   - Check that paths point to your data location

3. **Run cells sequentially:**
   - Some cells depend on previous cells
   - Run from top to bottom for best results

## Troubleshooting

### Problem: Import errors
**Solution:** Restart the Jupyter kernel and run again

### Problem: "Module not found"
**Solution:**
```bash
pip install <missing-module>
```

### Problem: "Data files not found"
**Solution:** Check that `.cha` files exist in `data/` directories

### Problem: "Kernel not found"
**Solution:**
```bash
pip install ipykernel
python -m ipykernel install --user --name=artistic
```
Then select the "artistic" kernel in Jupyter.

### Problem: LightGBM errors on macOS
**Solution:** Either:
1. Install libomp: `brew install libomp`
2. Or comment out LightGBM imports (notebook works without it)

## Notebook Structure

The notebook has 12 sections:

1. **Setup** - Imports and configuration
2. **Dataset Info** - Data sources and formats
3. **Features** - All 27 features documented
4. **Extraction** - How features are extracted
5. **Preprocessing** - Data cleaning and validation
6. **Models** - 7 ML algorithms explained
7. **Training** - Training configuration
8. **Visualizations** - Feature analysis plots
9. **Performance** - Model evaluation metrics
10. **Importance** - Feature importance analysis
11. **Usage** - Complete code examples
12. **Conclusion** - Summary and references

## Tips

1. **Run cells in order** - Some cells depend on previous ones
2. **Clear output** if notebook is slow: `Cell` → `All Output` → `Clear`
3. **Restart kernel** if you make changes to source code
4. **Save frequently** - Jupyter auto-saves but manual save is safer

## Support

For issues:
1. Check this guide first
2. Check the main `notebooks/README.md`
3. Review source code documentation in `src/`
4. Check error messages carefully - they usually point to the issue

## Summary

The type annotation errors have been **FIXED**. To run:

```bash
# Quick start (from project root)
python -m jupyter notebook notebooks/syntactic_semantic_model_analysis.ipynb
```

Then restart the kernel if you see any cached import errors!

---

**Last Updated:** 2025-11-17
**Status:** ✓ Type annotation fixes applied
