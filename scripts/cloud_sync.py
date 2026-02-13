#!/usr/bin/env python3
"""
Cloud Sync Utility

Command-line tool for syncing datasets and models with HuggingFace Hub.

Usage:
    # Upload all models to cloud
    python scripts/cloud_sync.py upload-models

    # Upload all datasets to cloud
    python scripts/cloud_sync.py upload-datasets

    # Download specific model from cloud
    python scripts/cloud_sync.py download-model <model_name>

    # List cloud models
    python scripts/cloud_sync.py list-models

    # List cloud datasets
    python scripts/cloud_sync.py list-datasets

    # Get cloud status
    python scripts/cloud_sync.py status

Author: Bimidu Gunathilake
Date: 2026-02-13
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.cloud import get_hf_manager, HFConfig, reset_hf_manager
from src.models.model_registry import ModelRegistry
from config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Reset HF manager to ensure it reads fresh values from .env
reset_hf_manager()


def upload_models():
    """Upload all models to HuggingFace Hub."""
    print("\n" + "="*70)
    print("UPLOADING MODELS TO HUGGINGFACE HUB")
    print("="*70 + "\n")
    
    hf_manager = get_hf_manager()
    
    if not hf_manager.is_authenticated:
        print("❌ Error: Not authenticated with HuggingFace Hub")
        print("   Please run: huggingface-cli login")
        return 1
    
    results = hf_manager.upload_all_models()
    
    print("\n" + "="*70)
    print("UPLOAD RESULTS")
    print("="*70)
    
    for model_name, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {model_name}")
    
    successful = sum(1 for v in results.values() if v)
    print(f"\nTotal: {successful}/{len(results)} models uploaded successfully")
    
    return 0 if successful == len(results) else 1


def upload_datasets():
    """Upload all datasets to HuggingFace Hub."""
    print("\n" + "="*70)
    print("UPLOADING DATASETS TO HUGGINGFACE HUB")
    print("="*70 + "\n")
    
    hf_manager = get_hf_manager()
    
    if not hf_manager.is_authenticated:
        print("❌ Error: Not authenticated with HuggingFace Hub")
        print("   Please run: huggingface-cli login")
        return 1
    
    results = hf_manager.upload_all_datasets()
    
    print("\n" + "="*70)
    print("UPLOAD RESULTS")
    print("="*70)
    
    for dataset_name, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {dataset_name}")
    
    successful = sum(1 for v in results.values() if v)
    print(f"\nTotal: {successful}/{len(results)} datasets uploaded successfully")
    
    return 0 if successful == len(results) else 1


def download_model(model_name: str):
    """Download a specific model from HuggingFace Hub."""
    print(f"\nDownloading model: {model_name}")
    
    hf_manager = get_hf_manager()
    
    if not hf_manager.is_authenticated:
        print("❌ Error: Not authenticated with HuggingFace Hub")
        print("   Please run: huggingface-cli login")
        return 1
    
    model_path = hf_manager.download_model(model_name)
    
    if model_path:
        print(f"✓ Model downloaded to: {model_path}")
        
        # Sync to local registry
        registry = ModelRegistry()
        registry._sync_model_from_cloud(model_name, model_path)
        print(f"✓ Model synced to local registry")
        
        return 0
    else:
        print(f"✗ Failed to download model: {model_name}")
        return 1


def download_dataset(dataset_name: str):
    """Download a specific dataset from HuggingFace Hub."""
    print(f"\nDownloading dataset: {dataset_name}")
    
    hf_manager = get_hf_manager()
    
    if not hf_manager.is_authenticated:
        print("❌ Error: Not authenticated with HuggingFace Hub")
        print("   Please run: huggingface-cli login")
        return 1
    
    dataset_path = hf_manager.download_dataset(dataset_name)
    
    if dataset_path:
        print(f"✓ Dataset downloaded to: {dataset_path}")
        return 0
    else:
        print(f"✗ Failed to download dataset: {dataset_name}")
        return 1


def list_models():
    """List all models available on HuggingFace Hub."""
    print("\n" + "="*70)
    print("CLOUD MODELS (HUGGINGFACE HUB)")
    print("="*70 + "\n")
    
    hf_manager = get_hf_manager()
    
    if not hf_manager.is_authenticated:
        print("❌ Error: Not authenticated with HuggingFace Hub")
        print("   Please run: huggingface-cli login")
        return 1
    
    cloud_models = hf_manager.list_cloud_models()
    
    if cloud_models:
        for i, model_name in enumerate(cloud_models, 1):
            print(f"{i:3d}. {model_name}")
        print(f"\nTotal: {len(cloud_models)} models")
    else:
        print("No models found on HuggingFace Hub")
    
    return 0


def list_datasets():
    """List all datasets available on HuggingFace Hub."""
    print("\n" + "="*70)
    print("CLOUD DATASETS (HUGGINGFACE HUB)")
    print("="*70 + "\n")
    
    hf_manager = get_hf_manager()
    
    if not hf_manager.is_authenticated:
        print("❌ Error: Not authenticated with HuggingFace Hub")
        print("   Please run: huggingface-cli login")
        return 1
    
    cloud_datasets = hf_manager.list_cloud_datasets()
    
    if cloud_datasets:
        for i, dataset_name in enumerate(cloud_datasets, 1):
            print(f"{i:3d}. {dataset_name}")
        print(f"\nTotal: {len(cloud_datasets)} datasets")
    else:
        print("No datasets found on HuggingFace Hub")
    
    return 0


def show_status():
    """Show cloud storage status and configuration."""
    print("\n" + "="*70)
    print("CLOUD STORAGE STATUS")
    print("="*70 + "\n")
    
    hf_manager = get_hf_manager()
    info = hf_manager.get_repo_info()
    
    print(f"Dataset Repository: {info['dataset_repo']}")
    print(f"Model Repository:   {info['model_repo']}")
    print(f"Authenticated:      {info['authenticated']}")
    print(f"Cloud Enabled:      {info['use_cloud']}")
    print(f"Fallback to Local:  {info['fallback_enabled']}")
    print(f"Cache Directory:    {info['cache_dir']}")
    
    if info['authenticated'] and info['use_cloud']:
        print(f"\nCloud Models:       {len(info.get('cloud_models', []))}")
        print(f"Cloud Datasets:     {len(info.get('cloud_datasets', []))}")
    else:
        print("\n⚠️  Cloud storage not available")
        print("   Run 'huggingface-cli login' to authenticate")
    
    # Show local status
    print("\n" + "-"*70)
    print("LOCAL STORAGE")
    print("-"*70 + "\n")
    
    registry = ModelRegistry(use_cloud=False)
    print(f"Local Models:       {len(registry.models_)}")
    
    import os
    data_dirs = [d for d in config.paths.data_dir.iterdir() if d.is_dir()]
    print(f"Local Datasets:     {len(data_dirs)}")
    
    print("\n" + "="*70 + "\n")
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Cloud Sync Utility for HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s upload-models              Upload all models to cloud
  %(prog)s upload-datasets            Upload all datasets to cloud
  %(prog)s download-model model_name  Download specific model
  %(prog)s list-models                List cloud models
  %(prog)s status                     Show cloud status

Setup:
  1. Install HuggingFace CLI: pip install huggingface-hub
  2. Login: huggingface-cli login
  3. Update config: Edit HF_MODEL_REPO and HF_DATASET_REPO in .env
        """
    )
    
    parser.add_argument(
        'command',
        choices=[
            'upload-models', 'upload-datasets',
            'download-model', 'download-dataset',
            'list-models', 'list-datasets',
            'status'
        ],
        help='Command to execute'
    )
    
    parser.add_argument(
        'name',
        nargs='?',
        help='Model or dataset name (for download commands)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.command == 'upload-models':
            return upload_models()
        elif args.command == 'upload-datasets':
            return upload_datasets()
        elif args.command == 'download-model':
            if not args.name:
                print("Error: model name required")
                return 1
            return download_model(args.name)
        elif args.command == 'download-dataset':
            if not args.name:
                print("Error: dataset name required")
                return 1
            return download_dataset(args.name)
        elif args.command == 'list-models':
            return list_models()
        elif args.command == 'list-datasets':
            return list_datasets()
        elif args.command == 'status':
            return show_status()
        else:
            parser.print_help()
            return 1
    
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
