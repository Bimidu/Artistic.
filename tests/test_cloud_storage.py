"""
Test Cloud Storage Integration

Simple tests to verify HuggingFace Hub integration works correctly.

Author: Bimidu Gunathilake
Date: 2026-02-13
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.cloud import get_hf_manager, HFConfig
from src.models.model_registry import ModelRegistry
from config import config


def test_hf_authentication():
    """Test HuggingFace authentication."""
    print("\n" + "="*70)
    print("TEST: HuggingFace Authentication")
    print("="*70)
    
    try:
        hf_manager = get_hf_manager()
        
        if hf_manager.is_authenticated:
            print("✓ Authenticated with HuggingFace Hub")
            return True
        else:
            print("✗ Not authenticated with HuggingFace Hub")
            print("  Run: huggingface-cli login")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_cloud_config():
    """Test cloud configuration."""
    print("\n" + "="*70)
    print("TEST: Cloud Configuration")
    print("="*70)
    
    try:
        print(f"USE_CLOUD_STORAGE:     {config.cloud.use_cloud}")
        print(f"CLOUD_FALLBACK_LOCAL:  {config.cloud.fallback_to_local}")
        print(f"HF_DATASET_REPO:       {config.cloud.hf_dataset_repo}")
        print(f"HF_MODEL_REPO:         {config.cloud.hf_model_repo}")
        print(f"CLOUD_AUTO_SYNC:       {config.cloud.auto_sync}")
        
        if config.cloud.hf_dataset_repo == "your-username/artistic-asd-datasets":
            print("\n⚠️  Warning: Using default repository names")
            print("   Update HF_DATASET_REPO and HF_MODEL_REPO in .env")
            return False
        
        print("\n✓ Cloud configuration looks good")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_model_registry_cloud():
    """Test model registry with cloud support."""
    print("\n" + "="*70)
    print("TEST: Model Registry with Cloud Support")
    print("="*70)
    
    try:
        # Create registry with cloud enabled
        registry = ModelRegistry(use_cloud=True)
        
        print(f"Cloud Enabled:  {registry.use_cloud}")
        print(f"Local Models:   {len(registry.models_)}")
        
        if registry.hf_manager:
            print("✓ HuggingFace Manager initialized")
        else:
            print("✗ HuggingFace Manager not available")
            return False
        
        # List local models
        if registry.models_:
            print(f"\nLocal Models Available:")
            for i, model_name in enumerate(list(registry.models_.keys())[:5], 1):
                print(f"  {i}. {model_name}")
            if len(registry.models_) > 5:
                print(f"  ... and {len(registry.models_) - 5} more")
        
        print("\n✓ Model registry with cloud support works")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cloud_list():
    """Test listing cloud resources."""
    print("\n" + "="*70)
    print("TEST: List Cloud Resources")
    print("="*70)
    
    try:
        hf_manager = get_hf_manager()
        
        if not hf_manager.is_authenticated:
            print("⚠️  Skipping - not authenticated")
            return False
        
        # List cloud models
        print("Fetching cloud models...")
        cloud_models = hf_manager.list_cloud_models()
        print(f"Cloud Models: {len(cloud_models)}")
        
        if cloud_models:
            for i, model_name in enumerate(cloud_models[:3], 1):
                print(f"  {i}. {model_name}")
            if len(cloud_models) > 3:
                print(f"  ... and {len(cloud_models) - 3} more")
        else:
            print("  (No models uploaded yet)")
        
        # List cloud datasets
        print("\nFetching cloud datasets...")
        cloud_datasets = hf_manager.list_cloud_datasets()
        print(f"Cloud Datasets: {len(cloud_datasets)}")
        
        if cloud_datasets:
            for i, dataset_name in enumerate(cloud_datasets[:3], 1):
                print(f"  {i}. {dataset_name}")
            if len(cloud_datasets) > 3:
                print(f"  ... and {len(cloud_datasets) - 3} more")
        else:
            print("  (No datasets uploaded yet)")
        
        print("\n✓ Cloud listing works")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fallback():
    """Test fallback to local storage."""
    print("\n" + "="*70)
    print("TEST: Fallback to Local Storage")
    print("="*70)
    
    try:
        # Create config with cloud disabled but fallback enabled
        test_config = HFConfig(
            use_cloud=False,
            fallback_to_local=True
        )
        
        hf_manager = get_hf_manager()
        
        # Try to download (should fallback to local)
        print("Attempting to download with cloud disabled...")
        
        # Get first local model name
        registry = ModelRegistry(use_cloud=False)
        if registry.models_:
            model_name = list(registry.models_.keys())[0]
            print(f"Testing fallback for model: {model_name}")
            
            model_path = hf_manager.download_model(model_name)
            
            if model_path and model_path.exists():
                print(f"✓ Fallback successful: {model_path}")
                return True
            else:
                print("✗ Fallback failed")
                return False
        else:
            print("⚠️  No local models to test fallback")
            return False
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all cloud storage tests."""
    print("\n" + "="*70)
    print("CLOUD STORAGE INTEGRATION TESTS")
    print("="*70)
    
    tests = [
        ("Authentication", test_hf_authentication),
        ("Configuration", test_cloud_config),
        ("Model Registry", test_model_registry_cloud),
        ("Cloud Listing", test_cloud_list),
        ("Local Fallback", test_fallback),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70 + "\n")
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:10s} {test_name}")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Cloud storage is ready to use.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the output above.")
        print("\nCommon fixes:")
        print("  1. Run: huggingface-cli login")
        print("  2. Update .env with your HuggingFace repository names")
        print("  3. Create repositories on huggingface.co")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
