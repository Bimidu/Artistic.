# Cloud Storage Migration Guide

This guide walks you through migrating from local-only storage to cloud-based storage using HuggingFace Hub.

## Overview

**What's Changing:**
- ✅ Models and datasets can be stored in the cloud (HuggingFace Hub)
- ✅ Automatic fallback to local storage if cloud is unavailable
- ✅ Backward compatible - existing code continues to work
- ✅ No breaking changes - cloud is optional

**What's NOT Changing:**
- Your existing code and workflows
- Local storage (still available as fallback)
- API endpoints and interfaces

## Migration Steps

### Step 1: Understand Current State

```bash
# Check what you have locally
ls -lh data/      # 2.6GB of datasets
ls -lh models/    # 2.9MB of trained models

# These will remain available even after enabling cloud storage
```

### Step 2: Install Requirements

Already done! The required packages are in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Step 3: Set Up HuggingFace Account

1. **Create Account** (if you don't have one)
   - Visit [huggingface.co/join](https://huggingface.co/join)
   - Free signup, no credit card required

2. **Get Access Token**
   - Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Click "New token"
   - Name it "artistic-asd-project"
   - Select "Write" permission
   - Copy the token

3. **Login**
   ```bash
   huggingface-cli login
   # Paste your token when prompted
   ```

### Step 4: Create Repositories

1. **Create Dataset Repository**
   - Go to [huggingface.co/new-dataset](https://huggingface.co/new-dataset)
   - Name: `artistic-asd-datasets`
   - Visibility: Choose public or private
   - Click "Create dataset"

2. **Create Model Repository**
   - Go to [huggingface.co/new](https://huggingface.co/new)
   - Name: `artistic-asd-models`
   - Type: Model
   - Visibility: Choose public or private
   - Click "Create model"

### Step 5: Configure Your Environment

1. **Copy environment template**
   ```bash
   cp .env.example .env
   ```

2. **Edit .env file**
   ```bash
   # Update these lines with your HuggingFace username/org
   HF_DATASET_REPO=your-username/artistic-asd-datasets
   HF_MODEL_REPO=your-username/artistic-asd-models
   
   # Enable cloud storage
   USE_CLOUD_STORAGE=True
   
   # Keep local fallback enabled (recommended)
   CLOUD_FALLBACK_LOCAL=True
   
   # Optional: Auto-upload models after training
   CLOUD_AUTO_SYNC=False
   ```

### Step 6: Test the Setup

```bash
# Run integration tests
python tests/test_cloud_storage.py

# Should see:
# ✓ PASS Authentication
# ✓ PASS Configuration
# ✓ PASS Model Registry
# ✓ PASS Cloud Listing
# ✓ PASS Local Fallback
```

### Step 7: Upload Existing Data (Optional)

**Option A: Upload Everything**
```bash
# This uploads all 2.6GB of datasets + 2.9MB of models
# Will take 10-30 minutes depending on your connection

python scripts/cloud_sync.py upload-models
python scripts/cloud_sync.py upload-datasets
```

**Option B: Upload Only Models** (Recommended)
```bash
# Models are small (2.9MB), upload them
python scripts/cloud_sync.py upload-models

# Keep datasets local for now (2.6GB)
# Upload specific datasets as needed
```

**Option C: Selective Upload**
```bash
# Upload only specific datasets
python scripts/cloud_sync.py upload-dataset asdbank_aac
python scripts/cloud_sync.py upload-dataset asdbank_nadig
```

### Step 8: Verify Upload

```bash
# Check what's in the cloud
python scripts/cloud_sync.py status

# You should see:
# Cloud Storage Status:
# - Models: X uploaded
# - Datasets: Y uploaded
```

Visit your repositories:
- Models: `https://huggingface.co/your-username/artistic-asd-models`
- Datasets: `https://huggingface.co/datasets/your-username/artistic-asd-datasets`

### Step 9: Update Your Workflow

**No code changes needed!** But here's what happens now:

**Before (Local Only):**
```python
from src.models.model_registry import ModelRegistry

registry = ModelRegistry()
model = registry.load_model("my_model")  # Loads from ./models/my_model
```

**After (Cloud Enabled):**
```python
from src.models.model_registry import ModelRegistry

registry = ModelRegistry()  # Cloud automatically enabled from .env
model = registry.load_model("my_model")  
# 1. Checks local first
# 2. If not found, downloads from cloud
# 3. Caches locally for future use
```

### Step 10: Test Your Application

```bash
# Start the API server
python run_api.py

# Try making predictions - models will load from cloud if needed
curl -X POST http://localhost:8000/predict/audio \
  -F "audio_file=@test_audio.wav"
```

## Post-Migration

### Recommended Setup

We recommend a **hybrid approach**:

1. **Keep models local** (2.9MB - tiny!)
   - Fast loading
   - No network dependency
   - Backup in cloud

2. **Store large datasets in cloud** (2.6GB)
   - Free up local disk space
   - Download specific datasets as needed
   - Automatic caching

3. **Enable fallback** (CLOUD_FALLBACK_LOCAL=True)
   - Works offline
   - No disruption if cloud is down

### Clean Up Local Data (Optional)

After verifying cloud upload:

```bash
# Remove large datasets from local storage
# ⚠️ Only do this after confirming cloud upload!

# List what you have
du -sh data/*

# Remove specific datasets
rm -rf data/asdbank_aac
rm -rf data/asdbank_nadig
rm -rf data/td

# Keep models local (they're small)
# Keep one sample dataset for testing
```

## Using Cloud Storage

### Training New Models

**With Auto-Sync Disabled** (default):
```python
from src.models.model_registry import ModelRegistry, ModelMetadata

registry = ModelRegistry()

# Train model
model = train_my_model()

# Save locally and manually upload
metadata = ModelMetadata(...)
registry.register_model(model, metadata)

# Upload to cloud
registry.sync_to_cloud(["my_new_model"])
```

**With Auto-Sync Enabled** (set CLOUD_AUTO_SYNC=True):
```python
# Automatically uploads to cloud after registration
registry.register_model(model, metadata)
# Done! Model is now in both local and cloud
```

### Loading Models

```python
# Automatically uses cloud if needed
model = registry.load_model("any_model_name")

# Force cloud (even if available locally)
model = registry.load_model("model_name", prefer_cloud=True)
```

### Downloading Datasets

```python
from src.cloud import get_hf_manager

hf_manager = get_hf_manager()

# Download dataset
path = hf_manager.download_dataset("asdbank_aac")
# Cached locally at: cache/hf_cache/...

# Use the dataset
chat_parser.parse_files(str(path))
```

## Rollback Plan

If you need to disable cloud storage:

1. **Update .env**
   ```env
   USE_CLOUD_STORAGE=False
   ```

2. **Ensure local data exists**
   ```bash
   # If you deleted local data, download from cloud first
   python scripts/cloud_sync.py download-models
   python scripts/cloud_sync.py download-datasets
   ```

3. **Restart your application**
   ```bash
   # Everything works as before, using local storage only
   python run_api.py
   ```

## Troubleshooting

### "Not authenticated with HuggingFace Hub"

```bash
# Re-login
huggingface-cli login
```

### "Model not found in registry or cloud"

```bash
# List what's available
python scripts/cloud_sync.py list-models

# Download specific model
python scripts/cloud_sync.py download-model model_name
```

### Upload is very slow

```bash
# Upload in background
nohup python scripts/cloud_sync.py upload-datasets &

# Or upload specific datasets only
python scripts/cloud_sync.py upload-dataset asdbank_aac
```

### Cache is taking too much space

```bash
# Clear cache
rm -rf cache/hf_cache/*

# Models will re-download on next use
```

## Best Practices

1. **Keep Cloud Fallback Enabled**
   - Ensures your app works even without internet
   - Set `CLOUD_FALLBACK_LOCAL=True`

2. **Upload Models, Keep Datasets Local Initially**
   - Models are small (2.9MB) - upload all
   - Datasets are large (2.6GB) - upload selectively

3. **Use Auto-Sync for Production**
   - In production, enable `CLOUD_AUTO_SYNC=True`
   - In development, keep it off for faster iteration

4. **Regular Backups**
   - Even with cloud storage, keep local backups
   - Cloud is not a replacement for version control

5. **Monitor Disk Usage**
   ```bash
   # Check cache size
   du -sh cache/hf_cache
   
   # Clear if needed
   rm -rf cache/hf_cache/*
   ```

## Timeline

- **Setup**: 10 minutes
  - Create account, login, configure

- **Upload Models**: 1-2 minutes
  - Only 2.9MB total

- **Upload Datasets**: 15-30 minutes
  - 2.6GB, depends on your connection
  - Optional - can upload selectively

- **Testing**: 5 minutes
  - Run tests, verify everything works

**Total**: ~30 minutes for complete migration

## Questions?

- Read the full documentation: [docs/CLOUD_SETUP.md](docs/CLOUD_SETUP.md)
- Check HuggingFace docs: [huggingface.co/docs/hub](https://huggingface.co/docs/hub)
- Test the setup: `python tests/test_cloud_storage.py`

## Summary

✅ **Backward Compatible**: Existing code works unchanged  
✅ **Optional**: Cloud storage can be disabled anytime  
✅ **Hybrid Approach**: Use both local and cloud storage  
✅ **Fast**: Automatic caching for downloaded resources  
✅ **Free**: No cost for your project size  
✅ **Production Ready**: Standard practice for ML projects  

You now have a modern, cloud-enabled ML project! 🚀
