# Notebooks Directory

This directory contains Jupyter notebooks for analysis and documentation of the ASD detection system.

## Available Notebooks

### 1. syntactic_semantic_model_analysis.ipynb
Comprehensive analysis and documentation of the Syntactic & Semantic model component.

**Topics covered:**
- 27 syntactic and semantic features
- Feature extraction pipeline
- Data preprocessing
- Model training (7 ML algorithms)
- Performance evaluation
- Feature importance analysis
- Visualizations and plots

## Getting Started

### Prerequisites

Make sure you have the following installed:

```bash
# Install Jupyter
pip install jupyter notebook jupyterlab

# Install visualization libraries
pip install matplotlib seaborn

# Install ML libraries (if not already installed)
pip install scikit-learn xgboost lightgbm

# Install NLP libraries
pip install spacy nltk textstat
python -m spacy download en_core_web_sm
```

### Running the Notebooks

**Option 1: Jupyter Notebook**
```bash
cd /Users/user/PycharmProjects/Artistic
jupyter notebook
```
Then navigate to `notebooks/` and open the desired notebook.

**Option 2: Jupyter Lab**
```bash
cd /Users/user/PycharmProjects/Artistic
jupyter lab
```

**Option 3: VS Code**
- Open the `.ipynb` file in VS Code
- VS Code will show a notebook interface
- Click "Run All" or run cells individually

### Running Specific Notebook

To directly open a specific notebook:
```bash
jupyter notebook notebooks/syntactic_semantic_model_analysis.ipynb
```

## Notebook Structure

Each notebook follows this structure:
1. **Setup and Imports** - Library imports and configuration
2. **Data Loading** - Loading datasets and samples
3. **Analysis** - Feature extraction and analysis
4. **Visualizations** - Charts and plots
5. **Model Training** - ML model training (if applicable)
6. **Results** - Performance metrics and conclusions
7. **Examples** - Usage examples and code snippets

## Tips for Running

### Memory Management
Some notebooks may require significant memory. If you encounter issues:
- Close other applications
- Restart the kernel: `Kernel` → `Restart`
- Clear output: `Cell` → `All Output` → `Clear`

### Data Paths
The notebooks assume the following directory structure:
```
Artistic/
├── data/
│   ├── asdbank_aac/
│   ├── asdbank_eigsti/
│   └── ...
├── notebooks/
│   └── syntactic_semantic_model_analysis.ipynb
├── src/
└── config.py
```

Make sure your data is in the correct location.

### Virtual Environment
It's recommended to run notebooks in a virtual environment:

```bash
# Create virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Run Jupyter
jupyter notebook
```

## Troubleshooting

### Kernel Issues
If the kernel doesn't start:
```bash
# Install ipykernel
pip install ipykernel

# Add your virtual environment as a kernel
python -m ipykernel install --user --name=artistic --display-name="Python (Artistic)"
```

Then select the "Python (Artistic)" kernel in Jupyter.

### Import Errors
If you get import errors:
1. Make sure you're in the project root directory
2. Check that all dependencies are installed
3. Restart the kernel: `Kernel` → `Restart`

### Data Not Found
If you get "file not found" errors:
1. Check the data directory path in `config.py`
2. Verify that `.cha` files exist in the data directories
3. Update the file paths in the notebook cells if needed

## Contributing

When adding new notebooks:
1. Follow the standard structure outlined above
2. Include clear documentation and comments
3. Add visualizations where appropriate
4. Update this README with notebook information

## Contact

For questions or issues:
- Check the main project README
- Review the code documentation in `src/`
- Contact: Randil Haturusinghe (Syntactic/Semantic)

---

**Last Updated:** 2025-11-17
