"""
Model Registry Module

This module provides model management and persistence for trained models.

Key functionalities:
- Save/load models with metadata
- Model versioning
- Model selection and retrieval
- Model metadata management

Author: Bimidu Gunathilake
"""

import joblib
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime

from src.utils.logger import get_logger
from config import config

logger = get_logger(__name__)


@dataclass
class ModelMetadata:
    """
    Metadata for a trained model.
    
    Attributes:
        model_name: Name of the model
        model_type: Type of model (e.g., 'random_forest', 'xgboost')
        version: Model version
        created_at: Creation timestamp
        accuracy: Test accuracy
        f1_score: Test F1 score
        feature_names: List of feature names
        n_features: Number of features
        hyperparameters: Model hyperparameters
        training_samples: Number of training samples
        description: Model description
    """
    model_name: str
    model_type: str
    version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    accuracy: float = 0.0
    f1_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    roc_auc: Optional[float] = None
    matthews_corr: float = 0.0
    confusion_matrix: List[List[int]] = field(default_factory=list)
    feature_names: List[str] = field(default_factory=list)
    n_features: int = 0
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_samples: int = 0
    component: str = "pragmatic_conversational"
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)


class ModelRegistry:
    """
    Model registry for managing trained models.
    
    Handles saving, loading, versioning, and metadata management.
    """
    
    def __init__(self, registry_dir: Optional[Path] = None):
        """
        Initialize model registry.
        
        Args:
            registry_dir: Directory for storing models (None = use config)
        """
        self.registry_dir = registry_dir or config.paths.models_dir
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.registry_dir / "registry.json"
        self.models_: Dict[str, ModelMetadata] = {}
        
        self.logger = logger
        
        # Load existing registry
        self._load_registry()
        
        self.logger.info(f"ModelRegistry initialized - Directory: {self.registry_dir}")
    
    def _load_registry(self):
        """Load registry metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                
                self.models_ = {
                    name: ModelMetadata(**metadata)
                    for name, metadata in data.items()
                }
                
                self.logger.info(f"Loaded registry with {len(self.models_)} models")
            except Exception as e:
                self.logger.error(f"Error loading registry: {e}")
                self.models_ = {}
        else:
            self.logger.info("No existing registry found, starting fresh")
    
    def _save_registry(self):
        """Save registry metadata to disk."""
        try:
            data = {
                name: metadata.to_dict()
                for name, metadata in self.models_.items()
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.debug("Registry metadata saved")
        except Exception as e:
            self.logger.error(f"Error saving registry: {e}")
    
    def register_model(
        self,
        model: Any,
        metadata: ModelMetadata,
        preprocessor: Optional[Any] = None
    ):
        """
        Register and save a trained model.
        
        Args:
            model: Trained model to save
            metadata: Model metadata
            preprocessor: Optional preprocessor to save with model
        """
        self.logger.info(f"Registering model: {metadata.model_name}")
        
        # Create model directory
        model_dir = self.registry_dir / metadata.model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / "model.joblib"
        joblib.dump(model, model_path)
        self.logger.info(f"Model saved to {model_path}")
        
        # Save preprocessor if provided
        if preprocessor is not None:
            preprocessor_path = model_dir / "preprocessor.joblib"
            joblib.dump(preprocessor, preprocessor_path)
            self.logger.info(f"Preprocessor saved to {preprocessor_path}")
        
        # Save individual metadata
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        # Update registry
        self.models_[metadata.model_name] = metadata
        self._save_registry()
        
        self.logger.info(f"Model {metadata.model_name} registered successfully")
    
    def load_model(
        self,
        model_name: str,
        load_preprocessor: bool = False
    ) -> Any:
        """
        Load a registered model.
        
        Args:
            model_name: Name of model to load
            load_preprocessor: Whether to load preprocessor as well
        
        Returns:
            Loaded model (and preprocessor if requested)
        """
        if model_name not in self.models_:
            raise ValueError(f"Model {model_name} not found in registry")
        
        self.logger.info(f"Loading model: {model_name}")
        
        model_dir = self.registry_dir / model_name
        
        # Load model
        model_path = model_dir / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = joblib.load(model_path)
        self.logger.info(f"Model loaded from {model_path}")
        
        if load_preprocessor:
            # Load preprocessor
            preprocessor_path = model_dir / "preprocessor.joblib"
            if preprocessor_path.exists():
                preprocessor = joblib.load(preprocessor_path)
                self.logger.info(f"Preprocessor loaded from {preprocessor_path}")
                return model, preprocessor
            else:
                self.logger.warning("Preprocessor file not found")
                return model, None
        
        return model
    
    def get_model_metadata(self, model_name: str) -> ModelMetadata:
        """
        Get metadata for a registered model.
        
        Args:
            model_name: Name of model
        
        Returns:
            ModelMetadata: Model metadata
        """
        if model_name not in self.models_:
            raise ValueError(f"Model {model_name} not found in registry")
        
        return self.models_[model_name]
    
    def list_models(self) -> List[str]:
        """
        List all registered models.
        
        Returns:
            List of model names
        """
        return list(self.models_.keys())
    
    def get_best_model(
        self,
        metric: str = 'f1_score'
    ) -> Tuple[str, ModelMetadata]:
        """
        Get the best performing model based on a metric.
        
        Args:
            metric: Metric to use for comparison ('accuracy' or 'f1_score')
        
        Returns:
            Tuple of (model_name, metadata)
        """
        if not self.models_:
            raise ValueError("No models in registry")
        
        best_name = None
        best_score = -1.0
        
        for name, metadata in self.models_.items():
            score = getattr(metadata, metric, 0.0)
            if score > best_score:
                best_score = score
                best_name = name
        
        if best_name is None:
            raise ValueError("Could not determine best model")
        
        return best_name, self.models_[best_name]
    
    def delete_model(self, model_name: str):
        """
        Delete a registered model.
        
        Args:
            model_name: Name of model to delete
        """
        if model_name not in self.models_:
            raise ValueError(f"Model {model_name} not found in registry")
        
        self.logger.info(f"Deleting model: {model_name}")
        
        # Delete model directory
        model_dir = self.registry_dir / model_name
        if model_dir.exists():
            import shutil
            shutil.rmtree(model_dir)
            self.logger.info(f"Deleted model directory: {model_dir}")
        
        # Remove from registry
        del self.models_[model_name]
        self._save_registry()
        
        self.logger.info(f"Model {model_name} deleted from registry")
    
    def get_registry_summary(self) -> pd.DataFrame:
        """
        Get summary of all registered models.
        
        Returns:
            pd.DataFrame: Summary DataFrame
        """
        if not self.models_:
            return pd.DataFrame()
        
        data = []
        for name, metadata in self.models_.items():
            data.append({
                'Model Name': name,
                'Type': metadata.model_type,
                'Version': metadata.version,
                'Accuracy': metadata.accuracy,
                'F1-Score': metadata.f1_score,
                'Features': metadata.n_features,
                'Created': metadata.created_at,
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('F1-Score', ascending=False)
        
        return df
    
    def print_summary(self):
        """Print registry summary."""
        print("\n" + "="*70)
        print("MODEL REGISTRY SUMMARY")
        print("="*70)
        
        if not self.models_:
            print("\nNo models registered")
        else:
            print(f"\nTotal Models: {len(self.models_)}")
            print(f"\nRegistered Models:")
            
            df = self.get_registry_summary()
            print(df.to_string(index=False))
            
            # Show best model
            try:
                best_name, best_meta = self.get_best_model()
                print(f"\nBest Model (by F1-Score):")
                print(f"  Name: {best_name}")
                print(f"  Type: {best_meta.model_type}")
                print(f"  Accuracy: {best_meta.accuracy:.4f}")
                print(f"  F1-Score: {best_meta.f1_score:.4f}")
            except:
                pass
        
        print("\n" + "="*70 + "\n")

