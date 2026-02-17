# Cloud Storage Setup Guide

This guide explains how to set up and use cloud storage for datasets and models using HuggingFace Hub.

## Table of Contents

- [Why Cloud Storage?](#why-cloud-storage)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Why Cloud Storage?

**Before (Local Only):**
- ❌ 2.6GB of datasets stored locally
- ❌ Models tied to local machine
- ❌ No version control for data/models
- ❌ Difficult to share and collaborate
- ❌ Manual backup required

**After (Cloud-Enabled):**
- ✅ Datasets hosted on HuggingFace Hub
- ✅ Models accessible from anywhere
- ✅ Built-in versioning and Git-like workflow
- ✅ Easy sharing and collaboration
- ✅ Automatic caching for fast access
- ✅ Fallback to local storage if needed

## Prerequisites

1. **HuggingFace Account**
   - Create a free account at [huggingface.co](https://huggingface.co/join)
   - No credit card required

2. **HuggingFace CLI**
   ```bash
   # Already included in requirements.txt
   pip install huggingface-hub
   ```

3. **Authentication Token**
   - Get your token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Create a token with "write" permissions

## Quick Start

### 1. Login to HuggingFace

```bash
huggingface-cli login
```

Enter your token when prompted. This only needs to be done once per machine.

### 2. Create Repositories

Go to [huggingface.co/new](https://huggingface.co/new) and create:

1. **Dataset Repository**: `your-username/artistic-asd-datasets`
   - Select "Dataset" type
   - Make it public or private

2. **Model Repository**: `your-username/artistic-asd-models`
   - Select "Model" type
   - Make it public or private

### 3. Configure Your App

Create a `.env` file (copy from `.env.example`):

```bash
cp .env.example .env
```

Edit `.env` and update:

```env
USE_CLOUD_STORAGE=True
HF_DATASET_REPO=your-username/artistic-asd-datasets
HF_MODEL_REPO=your-username/artistic-asd-models
```

### 4. Upload Your Data

```bash
# Upload all models
python scripts/cloud_sync.py upload-models

# Upload all datasets (this may take a while - 2.6GB)
python scripts/cloud_sync.py upload-datasets

# Check status
python scripts/cloud_sync.py status
```

### 5. Verify

Visit your repositories on HuggingFace to see your uploaded data:
- `https://huggingface.co/datasets/your-username/artistic-asd-datasets`
- `https://huggingface.co/your-username/artistic-asd-models`

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_CLOUD_STORAGE` | `True` | Enable cloud storage |
| `CLOUD_FALLBACK_LOCAL` | `True` | Use local storage if cloud fails |
| `HF_DATASET_REPO` | Required | Your HF dataset repository |
| `HF_MODEL_REPO` | Required | Your HF model repository |
| `CLOUD_AUTO_SYNC` | `False` | Auto-upload models after training |

### Programmatic Configuration

```python
from src.cloud import HFConfig, get_hf_manager

# Custom configuration
config = HFConfig(
    dataset_repo="my-org/my-datasets",
    model_repo="my-org/my-models",
    use_cloud=True,
    fallback_to_local=True
)

hf_manager = get_hf_manager(config)
```

## Usage

### Command Line Interface

```bash
# Upload commands
python scripts/cloud_sync.py upload-models      # Upload all models
python scripts/cloud_sync.py upload-datasets    # Upload all datasets

# Download commands
python scripts/cloud_sync.py download-model pragmatic_conversational_xgboost
python scripts/cloud_sync.py download-dataset asdbank_aac

# List commands
python scripts/cloud_sync.py list-models        # List cloud models
python scripts/cloud_sync.py list-datasets      # List cloud datasets

# Status
python scripts/cloud_sync.py status             # Show cloud status
```

### Python API

#### Download and Use a Model

```python
from src.models.model_registry import ModelRegistry

# Create registry with cloud enabled
registry = ModelRegistry(use_cloud=True)

# Load model (automatically downloads from cloud if not local)
model = registry.load_model("pragmatic_conversational_xgboost")

# Make predictions
predictions = model.predict(features)
```

#### Upload a New Model

```python
from src.models.model_registry import ModelRegistry, ModelMetadata

registry = ModelRegistry(use_cloud=True)

# Train your model
model = train_my_model()

# Create metadata
metadata = ModelMetadata(
    model_name="my_new_model",
    model_type="xgboost",
    accuracy=0.85,
    f1_score=0.83,
    # ... other metadata
)

# Register and automatically upload to cloud
registry.register_model(
    model=model,
    metadata=metadata,
    upload_to_cloud=True  # Upload immediately
)
```

#### Download a Dataset

```python
from src.cloud import get_hf_manager

hf_manager = get_hf_manager()

# Download dataset
dataset_path = hf_manager.download_dataset("asdbank_aac")

if dataset_path:
    print(f"Dataset available at: {dataset_path}")
    # Use the dataset...
```

### API Integration

The FastAPI backend automatically uses cloud storage when enabled:

```python
# In your API (already integrated)
from src.api.app import app

# The model registry automatically loads from cloud
# No code changes needed!
```

## Best Practices

### 1. Repository Organization

```
your-username/artistic-asd-datasets/
├── asdbank_aac/
│   ├── file1.cha
│   └── file2.cha
├── asdbank_nadig/
│   └── ...
└── td/
    └── ...

your-username/artistic-asd-models/
├── pragmatic_conversational_xgboost/
│   ├── model.joblib
│   ├── metadata.json
│   └── preprocessor.joblib
└── acoustic_prosodic_random_forest/
    └── ...
```

### 2. Versioning

HuggingFace Hub uses Git for versioning:

```bash
# Every upload creates a new commit
# View history on HuggingFace website
# Rollback if needed
```

### 3. Caching

Models and datasets are cached locally after first download:

```python
# First time: downloads from cloud
model = registry.load_model("my_model")

# Subsequent times: uses cached version
model = registry.load_model("my_model")

# Force re-download
dataset_path = hf_manager.download_dataset("my_dataset", force_download=True)
```

### 4. Selective Sync

Don't upload everything at once:

```bash
# Upload only specific models
from src.cloud import get_hf_manager

hf_manager = get_hf_manager()
hf_manager.upload_model("best_model")
```

### 5. Private Repositories

For sensitive data, make repositories private:
- Go to repository settings on HuggingFace
- Change visibility to "Private"
- Only authenticated users can access

## Troubleshooting

### Authentication Issues

**Problem**: "Not authenticated with HuggingFace Hub"

**Solution**:
```bash
# Re-login
huggingface-cli login

# Or set token manually
export HF_TOKEN=your_token_here
```

### Repository Not Found

**Problem**: "Repository not found: 404"

**Solution**:
1. Verify repository exists on HuggingFace
2. Check repository name in `.env` matches exactly
3. Ensure you have access permissions

### Upload Failures

**Problem**: Upload times out or fails

**Solution**:
```bash
# For large datasets, upload in smaller batches
# Or use HuggingFace CLI directly:
huggingface-cli upload your-username/repo-name ./data/asdbank_aac asdbank_aac
```

### Cache Issues

**Problem**: Using old cached version

**Solution**:
```python
# Force re-download
hf_manager.download_model("model_name", force_download=True)

# Or clear cache manually
rm -rf cache/hf_cache/*
```

### Fallback Behavior

When cloud storage fails, the system automatically falls back to local storage if `CLOUD_FALLBACK_LOCAL=True`.

Check logs:
```python
from src.utils.logger import get_logger

logger = get_logger(__name__)
# Look for "Falling back to local" messages
```

## Migration Guide

### Migrating from Local-Only Setup

If you already have local data and models:

1. **Backup your data**
   ```bash
   tar -czf backup.tar.gz data/ models/
   ```

2. **Enable cloud storage**
   ```bash
   # Edit .env
   USE_CLOUD_STORAGE=True
   ```

3. **Upload existing data**
   ```bash
   python scripts/cloud_sync.py upload-models
   python scripts/cloud_sync.py upload-datasets
   ```

4. **Verify upload**
   ```bash
   python scripts/cloud_sync.py status
   ```

5. **(Optional) Remove local data**
   ```bash
   # Only after verifying cloud upload!
   # Keep models local, remove large datasets
   rm -rf data/asdbank_*
   rm -rf data/td/
   
   # Models are small (2.9MB), keep them local too
   ```

### Hybrid Approach (Recommended)

- **Keep models local** (only 2.9MB)
- **Store datasets in cloud** (2.6GB)
- **Use cache for downloaded data**

This gives you:
- Fast model loading (local)
- Space savings (cloud datasets)
- Offline capability (cached data)

## Advanced Usage

### Custom Storage Backends

While HuggingFace Hub is the default, the architecture supports other backends:

```python
# Example: Add S3 backend (future enhancement)
class S3StorageManager:
    def upload_model(self, model_name, model_dir):
        # Upload to S3
        pass
    
    def download_model(self, model_name):
        # Download from S3
        pass
```

### Continuous Integration

Integrate with CI/CD:

```yaml
# .github/workflows/train-and-upload.yml
name: Train and Upload

on:
  push:
    branches: [main]

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Train models
        run: python train.py
      - name: Upload to HuggingFace
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          huggingface-cli login --token $HF_TOKEN
          python scripts/cloud_sync.py upload-models
```

## Cost Considerations

HuggingFace Hub is **FREE** for:
- Public repositories (unlimited storage)
- Private repositories (limited storage)

Paid plans available for:
- Large private repositories
- Enterprise features

For this project (2.6GB + 2.9MB), the free tier is sufficient.

## Support

- **HuggingFace Documentation**: [huggingface.co/docs/hub](https://huggingface.co/docs/hub)
- **Community Forum**: [discuss.huggingface.co](https://discuss.huggingface.co)
- **Project Issues**: Use the GitHub issues for project-specific questions

## Summary

✅ **Setup**: 5 minutes (create account, login, configure)  
✅ **Upload**: 10-30 minutes (depending on data size)  
✅ **Use**: Automatic (code loads from cloud transparently)  
✅ **Cost**: Free for this project size  
✅ **Benefits**: Versioning, sharing, collaboration, backup  

Cloud storage is now the standard practice for ML projects. This setup makes your project production-ready and collaboration-friendly! 🚀
