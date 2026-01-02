"""
Diagnosis Mapper for CHAT files

Infers diagnosis labels from:
1. @ID line group field
2. Directory structure (HR/LR, participant names)
3. Dataset-specific metadata
4. 0types.txt files

Author: Bimidu Gunathilake
"""

from pathlib import Path
from typing import Optional, Dict
import re

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DiagnosisMapper:
    """
    Maps file paths and metadata to diagnosis labels.
    
    Handles dataset-specific conventions and directory structures.
    """
    
    # Dataset-specific rules
    DATASET_RULES = {
        # Datasets with ALL ASD participants
        'asdbank_rollins': {
            'default_diagnosis': 'ASD',
            'description': 'All participants are young boys with autism'
        },
        'asdbank_flusberg': {
            'default_diagnosis': 'ASD',
            'description': 'All participants are children with ASD'
        },
        'asdbank_aac': {
            'default_diagnosis': 'ASD',
            'description': 'All participants use AAC and have ASD'
        },
        
        # Datasets with mixed populations
        'asdbank_eigsti': {
            'diagnosis_map': {
                'ASD': 'ASD',
                'DD': 'TD',  # Treat Developmental Delay as non-ASD (TD)
                'TD': 'TD',
                'TYP': 'TD'
            },
            'description': 'Children with ASD, DD (treated as TD), and TD'
        },
        'asdbank_nadig': {
            'diagnosis_map': {
                'ASD': 'ASD',
                'TYP': 'TD',
                'TYPICAL': 'TD',
                'CONTROL': 'TD',
                'TD': 'TD'
            },
            'description': 'Children with ASD and normal controls (TD)'
        },
        
        # High Risk / Low Risk datasets
        'asdbank_quigley_mcnalley': {
            'directory_rules': {
                '/HR/': 'ASD',  # High Risk → ASD
                '/LR/': 'TD',   # Low Risk → TD
            },
            'description': 'Longitudinal study: HR (high risk ASD) vs LR (low risk, TD)'
        },
    }
    
    def __init__(self):
        """Initialize diagnosis mapper."""
        self.logger = logger
        self.logger.info("DiagnosisMapper initialized")
    
    def infer_diagnosis(
        self,
        file_path: Path,
        extracted_diagnosis: Optional[str] = None
    ) -> Optional[str]:
        """
        Infer diagnosis label from file path and metadata.
        
        Args:
            file_path: Path to CHAT file
            extracted_diagnosis: Diagnosis extracted from @ID line (if any)
        
        Returns:
            Normalized diagnosis label ('ASD' or 'TD') or None
        """
        # First, try to use the extracted diagnosis from @ID line
        if extracted_diagnosis:
            normalized = self._normalize_diagnosis(extracted_diagnosis)
            if normalized:
                return normalized
        
        # Identify dataset from path
        dataset_name = self._identify_dataset(file_path)
        if not dataset_name:
            self.logger.warning(f"Could not identify dataset for {file_path}")
            return None
        
        # Apply dataset-specific rules
        rules = self.DATASET_RULES.get(dataset_name)
        if not rules:
            self.logger.debug(f"No specific rules for dataset: {dataset_name}")
            return None
        
        # Check for default diagnosis (all participants same)
        if 'default_diagnosis' in rules:
            diagnosis = rules['default_diagnosis']
            self.logger.debug(
                f"Using default diagnosis '{diagnosis}' for {dataset_name}: {file_path.name}"
            )
            return diagnosis
        
        # Check directory-based rules (e.g., HR/LR)
        if 'directory_rules' in rules:
            for dir_pattern, diagnosis in rules['directory_rules'].items():
                if dir_pattern in str(file_path):
                    self.logger.debug(
                        f"Matched directory pattern '{dir_pattern}' → '{diagnosis}': {file_path.name}"
                    )
                    return diagnosis
        
        # Check diagnosis mapping (normalize extracted diagnosis)
        if 'diagnosis_map' in rules and extracted_diagnosis:
            mapped = rules['diagnosis_map'].get(extracted_diagnosis.upper())
            if mapped:
                self.logger.debug(
                    f"Mapped '{extracted_diagnosis}' → '{mapped}' for {dataset_name}"
                )
                return mapped
        
        self.logger.warning(
            f"Could not infer diagnosis for {file_path.name} in {dataset_name}"
        )
        return None
    
    def _identify_dataset(self, file_path: Path) -> Optional[str]:
        """
        Identify dataset name from file path.
        
        Args:
            file_path: Path to CHAT file
        
        Returns:
            Dataset name (e.g., 'asdbank_rollins') or None
        """
        path_str = str(file_path)
        
        # Look for asdbank_* directory in path
        match = re.search(r'asdbank_[a-z_]+', path_str)
        if match:
            return match.group(0)
        
        # Try to identify from parent directories
        for part in file_path.parts:
            if part.startswith('asdbank_'):
                return part
        
        return None
    
    def _normalize_diagnosis(self, diagnosis: str) -> Optional[str]:
        """
        Normalize diagnosis label to 'ASD' or 'TD'.
        
        Args:
            diagnosis: Raw diagnosis label
        
        Returns:
            Normalized label ('ASD' or 'TD') or None
        """
        if not diagnosis:
            return None
        
        diagnosis_upper = diagnosis.upper().strip()
        
        # Map to binary classification
        if diagnosis_upper in ['ASD', 'AUTISM', 'AUTISTIC']:
            return 'ASD'
        elif diagnosis_upper in ['TD', 'TYP', 'TYPICAL', 'CONTROL', 'DD', 'DEVELOPMENTAL_DELAY']:
            return 'TD'
        
        return None
    
    def get_dataset_info(self, dataset_name: str) -> Optional[Dict]:
        """
        Get information about a dataset.
        
        Args:
            dataset_name: Name of dataset
        
        Returns:
            Dataset rules and description or None
        """
        return self.DATASET_RULES.get(dataset_name)
    
    def print_dataset_rules(self):
        """Print all dataset rules."""
        print("\n" + "="*70)
        print("DIAGNOSIS MAPPING RULES")
        print("="*70)
        
        for dataset, rules in self.DATASET_RULES.items():
            print(f"\n{dataset}:")
            print(f"  Description: {rules.get('description', 'N/A')}")
            
            if 'default_diagnosis' in rules:
                print(f"  Default: {rules['default_diagnosis']}")
            
            if 'directory_rules' in rules:
                print("  Directory Rules:")
                for pattern, diagnosis in rules['directory_rules'].items():
                    print(f"    {pattern} → {diagnosis}")
            
            if 'diagnosis_map' in rules:
                print("  Diagnosis Mapping:")
                for orig, mapped in rules['diagnosis_map'].items():
                    print(f"    {orig} → {mapped}")
        
        print("\n" + "="*70 + "\n")

