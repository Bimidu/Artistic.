"""
Dataset Inventory and Management System

This module provides functionality to build and manage an inventory of all
participants and transcripts across multiple datasets. It creates a structured
database of all available data for easy access and filtering.

Author: Bimidu Gunathilake
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from .chat_parser import CHATParser, TranscriptData
from src.utils.logger import get_logger
from config import config

logger = get_logger(__name__)


@dataclass
class ParticipantInfo:
    """
    Information about a single participant across all their sessions.
    
    Attributes:
        participant_id: Unique identifier for the participant
        dataset: Source dataset name (e.g., 'asdbank_eigsti')
        diagnosis: Clinical diagnosis (ASD, TD, DD, etc.)
        age_months: Age in months (from first session if multiple)
        gender: Participant gender
        num_sessions: Total number of transcript sessions
        session_files: List of transcript file paths
        total_utterances: Total utterances across all sessions
        metadata: Additional participant metadata
    """
    participant_id: str
    dataset: str
    diagnosis: Optional[str] = None
    age_months: Optional[int] = None
    gender: Optional[str] = None
    num_sessions: int = 0
    session_files: List[str] = None
    total_utterances: int = 0
    metadata: Dict = None
    
    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.session_files is None:
            self.session_files = []
        if self.metadata is None:
            self.metadata = {}


class DatasetInventory:
    """
    Build and manage inventory of all datasets and participants.
    
    This class scans all dataset directories, parses transcripts, and creates
    a comprehensive inventory that can be saved and loaded for quick access.
    
    Example:
        >>> inventory = DatasetInventory(data_dir="./data")
        >>> inventory.build_inventory()
        >>> df = inventory.to_dataframe()
        >>> print(df[df['diagnosis'] == 'ASD'].shape)
    """
    
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        cache_file: Optional[Path] = None
    ):
        """
        Initialize the dataset inventory system.
        
        Args:
            data_dir: Root directory containing all datasets
            cache_file: Path to cache file for saving/loading inventory
        """
        self.data_dir = data_dir or config.paths.data_dir
        self.cache_file = cache_file or (config.paths.cache_dir / "inventory.json")
        
        self.parser = CHATParser()
        self.participants: Dict[str, ParticipantInfo] = {}
        self.transcripts: List[TranscriptData] = []
        
        logger.info(f"DatasetInventory initialized with data_dir: {self.data_dir}")
    
    def build_inventory(
        self,
        datasets: Optional[List[str]] = None,
        force_rebuild: bool = False
    ) -> None:
        """
        Build complete inventory of all datasets.
        
        This method scans all specified datasets, parses transcripts, and
        creates a comprehensive inventory of participants and sessions.
        
        Args:
            datasets: List of dataset names to process (default: all from config)
            force_rebuild: If True, rebuild even if cache exists
            
        Example:
            >>> inventory.build_inventory(datasets=['asdbank_eigsti'])
            >>> inventory.build_inventory(force_rebuild=True)
        """
        # Check if cached inventory exists
        if not force_rebuild and self.cache_file.exists():
            logger.info("Loading inventory from cache...")
            self.load_from_cache()
            return
        
        datasets = datasets or config.datasets.datasets
        logger.info(f"Building inventory for {len(datasets)} datasets...")
        
        # Process each dataset
        for dataset_name in datasets:
            dataset_path = self.data_dir / dataset_name
            
            if not dataset_path.exists():
                logger.warning(f"Dataset directory not found: {dataset_path}")
                continue
            
            logger.info(f"Processing dataset: {dataset_name}")
            self._process_dataset(dataset_name, dataset_path)
        
        # Save to cache
        self.save_to_cache()
        
        # Log summary statistics
        self._log_summary()
    
    def _process_dataset(self, dataset_name: str, dataset_path: Path) -> None:
        """
        Process a single dataset directory.
        
        Args:
            dataset_name: Name of the dataset
            dataset_path: Path to dataset directory
        """
        try:
            # Find all .cha files recursively
            cha_files = list(dataset_path.rglob("*.cha"))
            
            if not cha_files:
                logger.warning(f"No .cha files found in {dataset_path}")
                return
            
            logger.info(f"Found {len(cha_files)} transcript files in {dataset_name}")
            
            # Parse each transcript with progress bar
            for file_path in tqdm(cha_files, desc=f"Parsing {dataset_name}"):
                try:
                    transcript = self.parser.parse_file(file_path)
                    self._add_transcript(transcript, dataset_name)
                except Exception as e:
                    logger.error(f"Error processing {file_path.name}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error processing dataset {dataset_name}: {e}")
    
    def _add_transcript(self, transcript: TranscriptData, dataset_name: str) -> None:
        """
        Add a transcript to the inventory.
        
        Args:
            transcript: Parsed transcript data
            dataset_name: Name of source dataset
        """
        # Store transcript
        self.transcripts.append(transcript)
        
        # Get or create participant info
        pid = transcript.participant_id
        
        if pid not in self.participants:
            # Extract diagnosis from speakers info or metadata
            diagnosis = None
            if '_diagnosis' in transcript.speakers:
                diagnosis = transcript.speakers['_diagnosis']
                # Standardize diagnosis label
                diagnosis = config.datasets.diagnosis_mapping.get(
                    diagnosis, diagnosis
                )
            
            # Extract age
            age_months = transcript.age_months
            if age_months is None and '_age_months' in transcript.speakers:
                age_months = transcript.speakers['_age_months']
            
            # Extract gender
            gender = transcript.gender
            if gender is None and '_gender' in transcript.speakers:
                gender = transcript.speakers['_gender']
            
            # Create participant info
            self.participants[pid] = ParticipantInfo(
                participant_id=pid,
                dataset=dataset_name,
                diagnosis=diagnosis,
                age_months=age_months,
                gender=gender,
                num_sessions=0,
                total_utterances=0,
                metadata=transcript.metadata
            )
        
        # Update participant info
        participant = self.participants[pid]
        participant.num_sessions += 1
        participant.session_files.append(str(transcript.file_path))
        participant.total_utterances += transcript.total_utterances
    
    def get_participants_by_diagnosis(
        self,
        diagnosis: str
    ) -> List[ParticipantInfo]:
        """
        Get all participants with a specific diagnosis.
        
        Args:
            diagnosis: Diagnosis code (ASD, TD, DD, etc.)
            
        Returns:
            List of matching participants
            
        Example:
            >>> asd_participants = inventory.get_participants_by_diagnosis('ASD')
            >>> print(f"Found {len(asd_participants)} ASD participants")
        """
        return [
            p for p in self.participants.values()
            if p.diagnosis == diagnosis
        ]
    
    def get_participants_by_age_range(
        self,
        min_age: int,
        max_age: int
    ) -> List[ParticipantInfo]:
        """
        Get participants within a specific age range.
        
        Args:
            min_age: Minimum age in months
            max_age: Maximum age in months
            
        Returns:
            List of matching participants
        """
        return [
            p for p in self.participants.values()
            if p.age_months and min_age <= p.age_months <= max_age
        ]
    
    def get_dataset_summary(self) -> Dict[str, Dict[str, int]]:
        """
        Get summary statistics for each dataset.
        
        Returns:
            Dictionary with counts by dataset and diagnosis
            
        Example:
            >>> summary = inventory.get_dataset_summary()
            >>> print(summary['asdbank_eigsti'])
            {'ASD': 16, 'TD': 16, 'DD': 16}
        """
        summary = {}
        
        for participant in self.participants.values():
            dataset = participant.dataset
            diagnosis = participant.diagnosis or 'Unknown'
            
            if dataset not in summary:
                summary[dataset] = {}
            
            if diagnosis not in summary[dataset]:
                summary[dataset][diagnosis] = 0
            
            summary[dataset][diagnosis] += 1
        
        return summary
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert inventory to pandas DataFrame for analysis.
        
        Returns:
            DataFrame with one row per participant
            
        Example:
            >>> df = inventory.to_dataframe()
            >>> print(df.groupby('diagnosis')['participant_id'].count())
        """
        data = []
        
        for participant in self.participants.values():
            row = {
                'participant_id': participant.participant_id,
                'dataset': participant.dataset,
                'diagnosis': participant.diagnosis,
                'age_months': participant.age_months,
                'gender': participant.gender,
                'num_sessions': participant.num_sessions,
                'total_utterances': participant.total_utterances,
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        return df
    
    def save_to_cache(self) -> None:
        """Save inventory to cache file."""
        try:
            cache_data = {
                'participants': {
                    pid: asdict(p) for pid, p in self.participants.items()
                },
                'metadata': {
                    'total_participants': len(self.participants),
                    'total_transcripts': len(self.transcripts),
                    'build_date': datetime.now().isoformat(),
                }
            }
            
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.info(f"Inventory saved to cache: {self.cache_file}")
            
        except Exception as e:
            logger.error(f"Error saving inventory to cache: {e}")
    
    def load_from_cache(self) -> bool:
        """
        Load inventory from cache file.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if not self.cache_file.exists():
                return False
            
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Restore participants
            self.participants = {
                pid: ParticipantInfo(**data)
                for pid, data in cache_data['participants'].items()
            }
            
            metadata = cache_data.get('metadata', {})
            logger.info(
                f"Loaded inventory from cache: "
                f"{metadata.get('total_participants')} participants, "
                f"{metadata.get('total_transcripts')} transcripts"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading inventory from cache: {e}")
            return False
    
    def export_to_csv(self, output_path: Optional[Path] = None) -> None:
        """
        Export inventory to CSV file.
        
        Args:
            output_path: Path for CSV file (default: output/inventory.csv)
        """
        output_path = output_path or (config.paths.output_dir / "inventory.csv")
        
        df = self.to_dataframe()
        df.to_csv(output_path, index=False)
        
        logger.info(f"Inventory exported to: {output_path}")
    
    def _log_summary(self) -> None:
        """Log summary statistics of the inventory."""
        total_participants = len(self.participants)
        total_transcripts = len(self.transcripts)
        
        # Count by diagnosis
        diagnosis_counts = {}
        for p in self.participants.values():
            dx = p.diagnosis or 'Unknown'
            diagnosis_counts[dx] = diagnosis_counts.get(dx, 0) + 1
        
        logger.info("=" * 50)
        logger.info("INVENTORY SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total Participants: {total_participants}")
        logger.info(f"Total Transcripts: {total_transcripts}")
        logger.info("\nParticipants by Diagnosis:")
        for dx, count in sorted(diagnosis_counts.items()):
            logger.info(f"  {dx}: {count}")
        
        # Dataset summary
        dataset_summary = self.get_dataset_summary()
        logger.info("\nDatasets:")
        for dataset, counts in sorted(dataset_summary.items()):
            logger.info(f"  {dataset}:")
            for dx, count in sorted(counts.items()):
                logger.info(f"    {dx}: {count}")
        logger.info("=" * 50)


__all__ = ["DatasetInventory", "ParticipantInfo"]

