# Cloud Storage Feature

## 🚀 New: Cloud Storage Support

The ASD Detection system now supports **cloud-based storage** for models and datasets using [HuggingFace Hub](https://huggingface.co). This is the industry-standard approach for ML projects.

### Why Cloud Storage?

**Before:**
- ❌ 2.6GB datasets on local disk
- ❌ Models tied to one machine
- ❌ No version control
- ❌ Difficult to share/collaborate

**After:**
- ✅ Models and data in the cloud
- ✅ Access from anywhere
- ✅ Built-in versioning
- ✅ Easy collaboration
- ✅ **FREE** for this project!

### Quick Start (5 minutes)

```bash
# 1. Run automated setup
bash scripts/quick_setup_cloud.sh

# 2. Upload your models (2.9MB - quick!)
python scripts/cloud_sync.py upload-models

# 3. (Optional) Upload datasets (2.6GB - slower)
python scripts/cloud_sync.py upload-datasets

# 4. Done! Your code works unchanged
python run_api.py
```

### Features

✅ **Automatic Fallback**: Works offline with local storage  
✅ **Smart Caching**: Downloaded data cached locally  
✅ **Backward Compatible**: Existing code unchanged  
✅ **CLI Tools**: Easy upload/download commands  
✅ **Free Storage**: No cost for your data size  
✅ **Versioning**: Git-like version control  

### Configuration

Create `.env` from template:
```bash
cp .env.example .env
```

Edit these lines:
```env
USE_CLOUD_STORAGE=True
HF_DATASET_REPO=your-username/artistic-asd-datasets
HF_MODEL_REPO=your-username/artistic-asd-models
```

### Usage

**No code changes needed!** The system automatically uses cloud storage:

```python
from src.models.model_registry import ModelRegistry

# Cloud is automatically enabled from .env
registry = ModelRegistry()

# This now:
# 1. Checks local storage
# 2. Downloads from cloud if needed
# 3. Caches locally
model = registry.load_model("my_model")
```

### Documentation

- **📖 Quick Start**: [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)
- **📚 Full Docs**: [docs/CLOUD_SETUP.md](docs/CLOUD_SETUP.md)
- **🔧 CLI Reference**: `python scripts/cloud_sync.py --help`
- **✅ Test Suite**: `python tests/test_cloud_storage.py`

### Command Line

```bash
# Upload
python scripts/cloud_sync.py upload-models
python scripts/cloud_sync.py upload-datasets

# Download
python scripts/cloud_sync.py download-model <name>
python scripts/cloud_sync.py download-dataset <name>

# List
python scripts/cloud_sync.py list-models
python scripts/cloud_sync.py list-datasets

# Status
python scripts/cloud_sync.py status
```

### Setup Requirements

1. **HuggingFace Account** (free): [huggingface.co/join](https://huggingface.co/join)
2. **Authentication**: `huggingface-cli login`
3. **Create Repositories**: 
   - Dataset: [huggingface.co/new-dataset](https://huggingface.co/new-dataset)
   - Model: [huggingface.co/new](https://huggingface.co/new)

### Hybrid Approach (Recommended)

Keep best of both worlds:
- **Models**: Keep local (2.9MB - tiny!)
- **Large Datasets**: Store in cloud (2.6GB)
- **Enable Fallback**: Works offline
- **Auto-Cache**: Fast subsequent access

### Disable Cloud Storage

Want to use local-only? Just set:
```env
USE_CLOUD_STORAGE=False
```

Everything continues to work!

### Cost

**FREE** ✅ for your project size

HuggingFace provides:
- Free public repositories
- Generous private repository limits
- No hidden costs

### Need Help?

- **Quick Setup**: Run `bash scripts/quick_setup_cloud.sh`
- **Test Setup**: Run `python tests/test_cloud_storage.py`
- **Read Docs**: See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)

---

**TL;DR**: Cloud storage is now available, free, optional, and works transparently. Run `bash scripts/quick_setup_cloud.sh` to get started! 🚀
