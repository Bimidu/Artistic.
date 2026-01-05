"""
Model Fusion Module

This module provides the fusion mechanism for combining predictions
from multiple component models into a single ASD prediction.

Fusion strategies:
1. Voting: Simple majority or weighted voting
2. Averaging: Average probabilities from all components
3. Stacking: Meta-learner on component predictions
4. Weighted: Component-specific weights based on performance

Author: Bimidu Gunathilake
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Literal
from dataclasses import dataclass, field
from pathlib import Path
import joblib

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ComponentPrediction:
    """
    Prediction from a single component model.
    
    Attributes:
        component: Component name (e.g., 'pragmatic_conversational')
        prediction: Class prediction
        probability: Probability of positive class
        probabilities: Full probability distribution
        model_name: Name of the model used
        confidence: Prediction confidence
    """
    component: str
    prediction: str
    probability: float
    probabilities: Dict[str, float]
    model_name: str = ""
    confidence: float = 1.0
    
    @property
    def is_asd(self) -> bool:
        """Check if prediction is ASD."""
        return self.prediction.upper() == 'ASD'


@dataclass
class FusionResult:
    """
    Result from model fusion.
    
    Attributes:
        final_prediction: Final fused prediction
        final_probability: Final probability of ASD
        final_probabilities: Full probability distribution
        confidence: Overall confidence in prediction
        component_predictions: Individual component predictions
        fusion_method: Method used for fusion
        component_weights: Weights used for each component
        explanation: Human-readable explanation
    """
    final_prediction: str
    final_probability: float
    final_probabilities: Dict[str, float]
    confidence: float
    component_predictions: List[ComponentPrediction]
    fusion_method: str
    component_weights: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""
    
    @property
    def is_asd(self) -> bool:
        """Check if final prediction is ASD."""
        return self.final_prediction.upper() == 'ASD'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'prediction': self.final_prediction,
            'probability': self.final_probability,
            'probabilities': self.final_probabilities,
            'confidence': self.confidence,
            'fusion_method': self.fusion_method,
            'component_predictions': [
                {
                    'component': cp.component,
                    'prediction': cp.prediction,
                    'probability': cp.probability,
                    'model_name': cp.model_name,
                }
                for cp in self.component_predictions
            ],
            'explanation': self.explanation,
        }


class ModelFusion:
    """
    Fusion of component models for final ASD prediction.
    
    This class combines predictions from three components:
    - Pragmatic & Conversational (implemented)
    - Acoustic & Prosodic (placeholder)
    - Syntactic & Semantic (placeholder)
    
    Multiple fusion strategies are available, with the ability to
    handle missing components gracefully.
    
    Example:
        >>> fusion = ModelFusion(method='weighted')
        >>> result = fusion.fuse([pred1, pred2, pred3])
        >>> print(f"Final: {result.final_prediction} ({result.confidence:.2f})")
    """
    
    # Default component weights based on expected discriminative power
    DEFAULT_WEIGHTS = {
        'pragmatic_conversational': 0.5,
        'acoustic_prosodic': 0.25,
        'syntactic_semantic': 0.25,
    }
    
    # Fusion methods
    FUSION_METHODS = ['voting', 'averaging', 'weighted', 'max_confidence', 'stacking']
    
    def __init__(
        self,
        method: Literal['voting', 'averaging', 'weighted', 'max_confidence', 'stacking'] = 'weighted',
        component_weights: Optional[Dict[str, float]] = None,
        threshold: float = 0.5,
    ):
        """
        Initialize the model fusion.
        
        Args:
            method: Fusion method to use
            component_weights: Custom weights for each component
            threshold: Probability threshold for ASD classification
        """
        if method not in self.FUSION_METHODS:
            raise ValueError(f"Unknown fusion method: {method}")
        
        self.method = method
        self.threshold = threshold
        self.component_weights = component_weights or self.DEFAULT_WEIGHTS.copy()
        
        # Meta-learner for stacking (trained later)
        self.meta_learner = None
        
        logger.info(f"ModelFusion initialized with method={method}")
    
    def fuse(
        self,
        component_predictions: List[ComponentPrediction],
        component_weights_override: Optional[Dict[str, float]] = None
    ) -> FusionResult:
        """
        Fuse component predictions into final prediction.
        
        Args:
            component_predictions: List of predictions from components
            component_weights_override: Optional weights to override instance weights
            
        Returns:
            FusionResult with final prediction
        """
        if not component_predictions:
            raise ValueError("No component predictions provided")
        
        logger.debug(f"Fusing {len(component_predictions)} component predictions")
        
        # Apply fusion based on method
        if self.method == 'voting':
            return self._fuse_voting(component_predictions)
        elif self.method == 'averaging':
            return self._fuse_averaging(component_predictions)
        elif self.method == 'weighted':
            return self._fuse_weighted(component_predictions, component_weights_override)
        elif self.method == 'max_confidence':
            return self._fuse_max_confidence(component_predictions)
        elif self.method == 'stacking':
            return self._fuse_stacking(component_predictions)
        else:
            raise ValueError(f"Unknown fusion method: {self.method}")
    
    def _fuse_voting(
        self,
        predictions: List[ComponentPrediction]
    ) -> FusionResult:
        """Simple majority voting."""
        asd_votes = sum(1 for p in predictions if p.is_asd)
        total_votes = len(predictions)
        
        is_asd = asd_votes > total_votes / 2
        final_prediction = 'ASD' if is_asd else 'TD'
        
        # Calculate probability as vote ratio
        asd_prob = asd_votes / total_votes
        
        return FusionResult(
            final_prediction=final_prediction,
            final_probability=asd_prob,
            final_probabilities={'ASD': asd_prob, 'TD': 1 - asd_prob},
            confidence=abs(asd_prob - 0.5) * 2,  # 0 at 50%, 1 at 0% or 100%
            component_predictions=predictions,
            fusion_method='voting',
            explanation=f"Majority voting: {asd_votes}/{total_votes} voted ASD",
        )
    
    def _fuse_averaging(
        self,
        predictions: List[ComponentPrediction]
    ) -> FusionResult:
        """Average probabilities from all components."""
        avg_asd_prob = np.mean([p.probability for p in predictions])
        
        is_asd = avg_asd_prob >= self.threshold
        final_prediction = 'ASD' if is_asd else 'TD'
        
        return FusionResult(
            final_prediction=final_prediction,
            final_probability=avg_asd_prob,
            final_probabilities={'ASD': avg_asd_prob, 'TD': 1 - avg_asd_prob},
            confidence=abs(avg_asd_prob - 0.5) * 2,
            component_predictions=predictions,
            fusion_method='averaging',
            explanation=f"Average probability: {avg_asd_prob:.3f}",
        )
    
    def _fuse_weighted(
        self,
        predictions: List[ComponentPrediction],
        component_weights_override: Optional[Dict[str, float]] = None
    ) -> FusionResult:
        """Weighted averaging based on component weights."""
        # Use override weights if provided, otherwise use instance weights
        weights_to_use = component_weights_override if component_weights_override is not None else self.component_weights
        
        total_weight = 0.0
        weighted_prob = 0.0
        
        for pred in predictions:
            weight = weights_to_use.get(pred.component, 0.33)
            # Skip components with zero weight
            if weight == 0:
                continue
            weighted_prob += pred.probability * weight
            total_weight += weight
        
        # Normalize
        if total_weight > 0:
            weighted_prob /= total_weight
        else:
            # Fallback: if all weights are 0, use simple average
            logger.warning("All component weights are 0, falling back to simple average")
            weighted_prob = np.mean([p.probability for p in predictions])
            total_weight = 1.0
        
        is_asd = weighted_prob >= self.threshold
        final_prediction = 'ASD' if is_asd else 'TD'
        
        # Build weight info for explanation
        weight_info = ", ".join([
            f"{p.component}: {weights_to_use.get(p.component, 0.33):.2f}"
            for p in predictions
        ])
        
        return FusionResult(
            final_prediction=final_prediction,
            final_probability=weighted_prob,
            final_probabilities={'ASD': weighted_prob, 'TD': 1 - weighted_prob},
            confidence=abs(weighted_prob - 0.5) * 2,
            component_predictions=predictions,
            fusion_method='weighted',
            component_weights={p.component: weights_to_use.get(p.component, 0.33) for p in predictions},
            explanation=f"Weighted average: {weighted_prob:.3f} (weights: {weight_info})",
        )
    
    def _fuse_max_confidence(
        self,
        predictions: List[ComponentPrediction]
    ) -> FusionResult:
        """Use prediction from most confident component."""
        # Find most confident prediction
        best_pred = max(predictions, key=lambda p: p.confidence)
        
        return FusionResult(
            final_prediction=best_pred.prediction,
            final_probability=best_pred.probability,
            final_probabilities=best_pred.probabilities,
            confidence=best_pred.confidence,
            component_predictions=predictions,
            fusion_method='max_confidence',
            explanation=f"Most confident: {best_pred.component} ({best_pred.confidence:.3f})",
        )
    
    def _fuse_stacking(
        self,
        predictions: List[ComponentPrediction]
    ) -> FusionResult:
        """Use meta-learner on component predictions."""
        if self.meta_learner is None:
            # Fall back to weighted if meta-learner not trained
            logger.warning("Meta-learner not trained, falling back to weighted fusion")
            return self._fuse_weighted(predictions)
        
        # Create feature vector from component predictions
        features = np.array([[p.probability for p in predictions]])
        
        # Get meta-learner prediction
        meta_pred = self.meta_learner.predict(features)[0]
        meta_prob = self.meta_learner.predict_proba(features)[0]
        
        # Determine class indices
        classes = self.meta_learner.classes_
        asd_idx = list(classes).index('ASD') if 'ASD' in classes else 0
        
        asd_prob = meta_prob[asd_idx]
        
        return FusionResult(
            final_prediction=meta_pred,
            final_probability=asd_prob,
            final_probabilities={'ASD': asd_prob, 'TD': 1 - asd_prob},
            confidence=abs(asd_prob - 0.5) * 2,
            component_predictions=predictions,
            fusion_method='stacking',
            explanation="Meta-learner prediction on component outputs",
        )
    
    def train_meta_learner(
        self,
        component_probabilities: np.ndarray,
        true_labels: np.ndarray,
    ):
        """
        Train meta-learner for stacking fusion.
        
        Args:
            component_probabilities: Array of shape (n_samples, n_components)
            true_labels: True class labels
        """
        from sklearn.linear_model import LogisticRegression
        
        logger.info("Training meta-learner for stacking fusion")
        
        self.meta_learner = LogisticRegression(
            random_state=42,
            max_iter=1000
        )
        self.meta_learner.fit(component_probabilities, true_labels)
        
        logger.info("Meta-learner trained successfully")
    
    def save(self, path: str | Path):
        """Save fusion model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        save_data = {
            'method': self.method,
            'threshold': self.threshold,
            'component_weights': self.component_weights,
            'meta_learner': self.meta_learner,
        }
        
        joblib.dump(save_data, path)
        logger.info(f"Fusion model saved to {path}")
    
    @classmethod
    def load(cls, path: str | Path) -> 'ModelFusion':
        """Load fusion model from disk."""
        path = Path(path)
        
        save_data = joblib.load(path)
        
        fusion = cls(
            method=save_data['method'],
            component_weights=save_data['component_weights'],
            threshold=save_data['threshold'],
        )
        fusion.meta_learner = save_data.get('meta_learner')
        
        logger.info(f"Fusion model loaded from {path}")
        
        return fusion
    
    def update_weights(self, new_weights: Dict[str, float]):
        """Update component weights."""
        self.component_weights.update(new_weights)
        logger.info(f"Updated component weights: {self.component_weights}")


__all__ = ["ModelFusion", "FusionResult", "ComponentPrediction"]

