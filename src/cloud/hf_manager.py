"""
Hugging Face Hub Manager for Cloud Storage

This module handles uploading and downloading datasets and models to/from
Hugging Face Hub for cloud-based storage and retrieval.

Features:
- Upload/download datasets
- Upload/download models with metadata
- Versioning support
- Automatic caching
- Fallback to local storage

Author: Bimidu Gunathilake
Date: 2026-02-13
"""

import os
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass
import joblib

from huggingface_hub import (
    HfApi,
    hf_hub_download,
    upload_file,
    upload_folder,
    create_repo,
    snapshot_download,
    login,
)
from huggingface_hub.utils import HfHubHTTPError

try:
    # Try new API (huggingface_hub >= 0.20.0)
    from huggingface_hub import get_token
    HF_TOKEN_GETTER = get_token
except ImportError:
    # Fallback to old API
    try:
        from huggingface_hub import HfFolder
        HF_TOKEN_GETTER = HfFolder.get_token
    except ImportError:
        # Very old version
        def HF_TOKEN_GETTER():
            import os
            return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

from src.utils.logger import get_logger
from config import config

logger = get_logger(__name__)


@dataclass
class HFConfig:
    """Configuration for Hugging Face Hub integration."""
    
    # Repository names
    dataset_repo: str = "your-username/artistic-asd-datasets"
    model_repo: str = "your-username/artistic-asd-models"
    
    # Cache directory
    cache_dir: Path = config.paths.cache_dir / "hf_cache"
    
    # Enable/disable cloud storage
    use_cloud: bool = True
    
    # Fallback to local if cloud fails
    fallback_to_local: bool = True


class HuggingFaceManager:
    """
    Manager for Hugging Face Hub operations.
    
    Handles uploading/downloading datasets and models to/from HF Hub.
    """
    
    def __init__(self, hf_config: Optional[HFConfig] = None):
        """
        Initialize Hugging Face Manager.
        
        Args:
            hf_config: Configuration for HF Hub (None = use defaults from config.cloud)
        """
        # If no config provided, create from config.cloud (reads from .env)
        if hf_config is None:
            hf_config = HFConfig(
                dataset_repo=config.cloud.hf_dataset_repo,
                model_repo=config.cloud.hf_model_repo,
                use_cloud=config.cloud.use_cloud,
                fallback_to_local=config.cloud.fallback_to_local,
                cache_dir=config.paths.cache_dir / "hf_cache"
            )
        self.config = hf_config
        self._token = None  # Will be set during authentication check
        self.api = HfApi()
        self.logger = logger
        
        # Create cache directory
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Check authentication (checks .env first, then cache)
        self.is_authenticated = self._check_authentication()
        
        if self.config.use_cloud and not self.is_authenticated:
            self.logger.warning(
                "Not authenticated with Hugging Face Hub. "
                "Set HF_TOKEN in .env file or run 'python scripts/hf_login.py' to authenticate. "
                "Falling back to local storage."
            )
            if not self.config.fallback_to_local:
                raise RuntimeError("HF Hub authentication required but not available")
        
        self.logger.info(
            f"HuggingFaceManager initialized - "
            f"Cloud: {self.config.use_cloud and self.is_authenticated}, "
            f"Fallback: {self.config.fallback_to_local}"
        )
    
    def _check_authentication(self) -> bool:
        """Check if user is authenticated with HF Hub."""
        try:
            # First check environment variable (from .env)
            import os
            token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
            
            # If not in env, try getting from cache
            if not token:
                token = HF_TOKEN_GETTER()
            
            if token:
                # Try to get user info to verify token
                self.api.whoami(token=token)
                # Store token for API calls
                self._token = token
                return True
        except Exception as e:
            self.logger.debug(f"HF Hub authentication check failed: {e}")
        return False
    
    def _get_token(self):
        """Get authentication token (from env or cache)."""
        import os
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if not token:
            token = HF_TOKEN_GETTER()
        return token
    
    # ==================== DATASET OPERATIONS ====================
    
    def upload_dataset(
        self,
        dataset_path: Union[str, Path],
        dataset_name: str,
        commit_message: Optional[str] = None
    ) -> bool:
        """
        Upload a dataset to Hugging Face Hub.
        
        Args:
            dataset_path: Path to dataset directory or file
            dataset_name: Name of dataset (used in repo path)
            commit_message: Optional commit message
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.config.use_cloud or not self.is_authenticated:
            self.logger.warning("Cloud storage not available, skipping upload")
            return False
        
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            self.logger.error(f"Dataset path not found: {dataset_path}")
            return False
        
        try:
            # Get token for API calls
            token = self._get_token()
            
            # Create repository if it doesn't exist
            try:
                self.api.create_repo(
                    repo_id=self.config.dataset_repo,
                    repo_type="dataset",
                    exist_ok=True,
                    token=token
                )
                self.logger.info(f"Dataset repo ready: {self.config.dataset_repo}")
            except Exception as e:
                self.logger.warning(f"Repo creation warning: {e}")
            
            # Upload folder or file
            commit_msg = commit_message or f"Upload dataset: {dataset_name}"
            
            if dataset_path.is_dir():
                # Upload entire folder
                self.logger.info(f"Uploading dataset folder: {dataset_path}")
                self.api.upload_folder(
                    folder_path=str(dataset_path),
                    path_in_repo=dataset_name,
                    repo_id=self.config.dataset_repo,
                    repo_type="dataset",
                    commit_message=commit_msg,
                    token=token
                )
            else:
                # Upload single file
                self.logger.info(f"Uploading dataset file: {dataset_path}")
                self.api.upload_file(
                    path_or_fileobj=str(dataset_path),
                    path_in_repo=f"{dataset_name}/{dataset_path.name}",
                    repo_id=self.config.dataset_repo,
                    repo_type="dataset",
                    commit_message=commit_msg,
                    token=token
                )
            
            self.logger.info(f"✓ Dataset uploaded: {dataset_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to upload dataset {dataset_name}: {e}")
            return False
    
    def download_dataset(
        self,
        dataset_name: str,
        force_download: bool = False
    ) -> Optional[Path]:
        """
        Download a dataset from Hugging Face Hub.
        
        Args:
            dataset_name: Name of dataset to download
            force_download: Force re-download even if cached
        
        Returns:
            Path to downloaded dataset, or None if failed
        """
        if not self.config.use_cloud or not self.is_authenticated:
            if self.config.fallback_to_local:
                local_path = config.paths.data_dir / dataset_name
                if local_path.exists():
                    self.logger.info(f"Using local dataset: {local_path}")
                    return local_path
                else:
                    self.logger.error(f"Dataset not found locally: {local_path}")
                    return None
            return None
        
        try:
            self.logger.info(f"Downloading dataset: {dataset_name}")
            
            # Get token for download
            token = self._get_token()
            
            # Download entire folder
            cache_path = snapshot_download(
                repo_id=self.config.dataset_repo,
                repo_type="dataset",
                allow_patterns=f"{dataset_name}/**",
                cache_dir=self.config.cache_dir,
                force_download=force_download,
                token=token
            )
            
            dataset_path = Path(cache_path) / dataset_name
            
            if not dataset_path.exists():
                self.logger.error(f"Dataset not found in repo: {dataset_name}")
                return None
            
            self.logger.info(f"✓ Dataset downloaded: {dataset_path}")
            return dataset_path
            
        except HfHubHTTPError as e:
            if e.response.status_code == 404:
                self.logger.error(f"Dataset not found on HF Hub: {dataset_name}")
            else:
                self.logger.error(f"Failed to download dataset: {e}")
            
            # Fallback to local
            if self.config.fallback_to_local:
                local_path = config.paths.data_dir / dataset_name
                if local_path.exists():
                    self.logger.info(f"Falling back to local dataset: {local_path}")
                    return local_path
            
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error downloading dataset: {e}")
            return None
    
    # ==================== MODEL OPERATIONS ====================
    
    def upload_model(
        self,
        model_name: str,
        model_dir: Optional[Path] = None,
        commit_message: Optional[str] = None
    ) -> bool:
        """
        Upload a trained model to Hugging Face Hub.
        
        Args:
            model_name: Name of model to upload
            model_dir: Directory containing model files (None = use registry)
            commit_message: Optional commit message
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.config.use_cloud or not self.is_authenticated:
            self.logger.warning("Cloud storage not available, skipping upload")
            return False
        
        # Use registry directory if not specified
        if model_dir is None:
            model_dir = config.paths.models_dir / model_name
        else:
            model_dir = Path(model_dir)
        
        if not model_dir.exists():
            self.logger.error(f"Model directory not found: {model_dir}")
            return False
        
        try:
            # Get token for API calls
            token = self._get_token()
            
            # Create repository if it doesn't exist
            try:
                self.api.create_repo(
                    repo_id=self.config.model_repo,
                    repo_type="model",
                    exist_ok=True,
                    token=token
                )
                self.logger.info(f"Model repo ready: {self.config.model_repo}")
            except Exception as e:
                self.logger.warning(f"Repo creation warning: {e}")
            
            # Upload model folder
            commit_msg = commit_message or f"Upload model: {model_name}"
            
            self.logger.info(f"Uploading model: {model_name}")
            self.api.upload_folder(
                folder_path=str(model_dir),
                path_in_repo=model_name,
                repo_id=self.config.model_repo,
                repo_type="model",
                commit_message=commit_msg,
                token=token
            )
            
            self.logger.info(f"✓ Model uploaded: {model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to upload model {model_name}: {e}")
            return False
    
    def download_model(
        self,
        model_name: str,
        force_download: bool = False
    ) -> Optional[Path]:
        """
        Download a model from Hugging Face Hub.
        
        Args:
            model_name: Name of model to download
            force_download: Force re-download even if cached
        
        Returns:
            Path to downloaded model directory, or None if failed
        """
        if not self.config.use_cloud or not self.is_authenticated:
            if self.config.fallback_to_local:
                local_path = config.paths.models_dir / model_name
                if local_path.exists():
                    self.logger.info(f"Using local model: {local_path}")
                    return local_path
                else:
                    self.logger.error(f"Model not found locally: {local_path}")
                    return None
            return None
        
        try:
            self.logger.info(f"Downloading model: {model_name}")
            
            # Get token for download
            token = self._get_token()
            
            # Download entire model folder
            cache_path = snapshot_download(
                repo_id=self.config.model_repo,
                repo_type="model",
                allow_patterns=f"{model_name}/**",
                cache_dir=self.config.cache_dir,
                force_download=force_download,
                token=token
            )
            
            model_path = Path(cache_path) / model_name
            
            if not model_path.exists():
                self.logger.error(f"Model not found in repo: {model_name}")
                return None
            
            self.logger.info(f"✓ Model downloaded: {model_path}")
            return model_path
            
        except HfHubHTTPError as e:
            if e.response.status_code == 404:
                self.logger.error(f"Model not found on HF Hub: {model_name}")
            else:
                self.logger.error(f"Failed to download model: {e}")
            
            # Fallback to local
            if self.config.fallback_to_local:
                local_path = config.paths.models_dir / model_name
                if local_path.exists():
                    self.logger.info(f"Falling back to local model: {local_path}")
                    return local_path
            
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error downloading model: {e}")
            return None
    
    def upload_all_models(self) -> Dict[str, bool]:
        """
        Upload all models from the local registry to HF Hub.
        
        Returns:
            Dict mapping model_name to success status
        """
        results = {}
        models_dir = config.paths.models_dir
        
        # Find all model directories
        model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
        
        self.logger.info(f"Uploading {len(model_dirs)} models to HF Hub...")
        
        for model_dir in model_dirs:
            model_name = model_dir.name
            success = self.upload_model(model_name, model_dir)
            results[model_name] = success
        
        successful = sum(1 for v in results.values() if v)
        self.logger.info(f"Upload complete: {successful}/{len(results)} successful")
        
        return results
    
    def upload_all_datasets(self) -> Dict[str, bool]:
        """
        Upload all datasets from the local data directory to HF Hub.
        
        Returns:
            Dict mapping dataset_name to success status
        """
        results = {}
        data_dir = config.paths.data_dir
        
        # Find all dataset directories (exclude files)
        dataset_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
        
        self.logger.info(f"Uploading {len(dataset_dirs)} datasets to HF Hub...")
        
        for dataset_dir in dataset_dirs:
            dataset_name = dataset_dir.name
            success = self.upload_dataset(dataset_dir, dataset_name)
            results[dataset_name] = success
        
        successful = sum(1 for v in results.values() if v)
        self.logger.info(f"Upload complete: {successful}/{len(results)} successful")
        
        return results
    
    # ==================== UTILITY OPERATIONS ====================
    
    def list_cloud_models(self) -> List[str]:
        """
        List all models available on HF Hub.
        
        Returns:
            List of model names
        """
        if not self.config.use_cloud or not self.is_authenticated:
            self.logger.warning("Cloud storage not available")
            return []
        
        try:
            token = self._get_token()
            files = self.api.list_repo_files(
                repo_id=self.config.model_repo,
                repo_type="model",
                token=token
            )
            
            # Extract unique model names (first path component)
            model_names = set()
            for file_path in files:
                parts = file_path.split('/')
                if len(parts) > 0 and parts[0]:
                    model_names.add(parts[0])
            
            return sorted(list(model_names))
            
        except Exception as e:
            self.logger.error(f"Failed to list cloud models: {e}")
            return []
    
    def list_cloud_datasets(self) -> List[str]:
        """
        List all datasets available on HF Hub.
        
        Returns:
            List of dataset names
        """
        if not self.config.use_cloud or not self.is_authenticated:
            self.logger.warning("Cloud storage not available")
            return []
        
        try:
            token = self._get_token()
            files = self.api.list_repo_files(
                repo_id=self.config.dataset_repo,
                repo_type="dataset",
                token=token
            )
            
            # Extract unique dataset names
            dataset_names = set()
            for file_path in files:
                parts = file_path.split('/')
                if len(parts) > 0 and parts[0]:
                    dataset_names.add(parts[0])
            
            return sorted(list(dataset_names))
            
        except Exception as e:
            self.logger.error(f"Failed to list cloud datasets: {e}")
            return []
    
    def get_repo_info(self) -> Dict[str, Any]:
        """Get information about the configured HF repositories."""
        info = {
            "dataset_repo": self.config.dataset_repo,
            "model_repo": self.config.model_repo,
            "authenticated": self.is_authenticated,
            "use_cloud": self.config.use_cloud,
            "fallback_enabled": self.config.fallback_to_local,
            "cache_dir": str(self.config.cache_dir),
        }
        
        if self.is_authenticated and self.config.use_cloud:
            try:
                info["cloud_models"] = self.list_cloud_models()
                info["cloud_datasets"] = self.list_cloud_datasets()
            except Exception as e:
                self.logger.warning(f"Could not fetch repo info: {e}")
        
        # Show token source (for debugging, don't show actual token)
        import os
        if os.getenv("HF_TOKEN"):
            info["token_source"] = ".env file"
        else:
            info["token_source"] = "cached token (~/.cache/huggingface/token)"
        
        return info


# Global instance (lazy-loaded)
_hf_manager: Optional[HuggingFaceManager] = None


def reset_hf_manager():
    """Reset the global HuggingFaceManager instance (useful when config changes)."""
    global _hf_manager
    _hf_manager = None


def get_hf_manager(hf_config: Optional[HFConfig] = None, force_reload: bool = False) -> HuggingFaceManager:
    """
    Get or create the global HuggingFaceManager instance.
    
    Args:
        hf_config: Configuration (only used on first call or if force_reload=True)
                   If None, uses values from config.cloud (which reads from .env)
        force_reload: Force reload even if instance exists (useful when .env changes)
    
    Returns:
        HuggingFaceManager instance
    """
    global _hf_manager
    
    if _hf_manager is None or force_reload:
        # If no config provided, create one from config.cloud (reads from .env)
        if hf_config is None:
            hf_config = HFConfig(
                dataset_repo=config.cloud.hf_dataset_repo,
                model_repo=config.cloud.hf_model_repo,
                use_cloud=config.cloud.use_cloud,
                fallback_to_local=config.cloud.fallback_to_local,
                cache_dir=config.paths.cache_dir / "hf_cache"
            )
        _hf_manager = HuggingFaceManager(hf_config)
    
    return _hf_manager
