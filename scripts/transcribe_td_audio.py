#!/usr/bin/env python3
"""
Batch Transcription Script for TD Audio Files

Transcribes all TD (Typically Developing) audio files from data/td/
using faster-whisper and saves them in CHAT format.

Features:
- Batch processing with progress tracking
- Resume capability (skips already transcribed files)
- Test mode for trying on a subset
- Error handling and logging
- Summary statistics

Usage:
    # Test mode (50 random files)
    python scripts/transcribe_td_audio.py --test --sample-size 50
    
    # Full batch processing
    python scripts/transcribe_td_audio.py
    
    # Use base model instead of tiny
    python scripts/transcribe_td_audio.py --model base
    
    # Specify custom output directory
    python scripts/transcribe_td_audio.py --output output/my_transcripts

Author: Randil Haturusinghe
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tqdm import tqdm

from src.audio.audio_processor import AudioProcessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TDBatchTranscriber:
    """Batch transcriber for TD audio files."""
    
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        model_size: str = 'tiny',
        backend: str = 'faster-whisper',
        resume: bool = True
    ):
        """
        Initialize batch transcriber.
        
        Args:
            input_dir: Directory containing TD audio files
            output_dir: Directory to save transcripts
            model_size: Whisper model size (tiny/base/small/medium/large)
            backend: Transcription backend (faster-whisper recommended)
            resume: Skip already transcribed files
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.model_size = model_size
        self.backend = backend
        self.resume = resume
        
        # Create output directories
        self.chat_dir = self.output_dir / 'chat_files'
        self.log_dir = self.output_dir / 'logs'
        self.chat_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize audio processor
        logger.info(f"Initializing AudioProcessor with {backend}/{model_size}")
        self.processor = AudioProcessor(
            transcriber_backend=backend,
            whisper_model_size=model_size,
            device='cpu',
            language='en'
        )
        
        # Statistics
        self.stats = {
            'total_files': 0,
            'processed': 0,
            'skipped': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None,
            'error_files': []
        }
    
    def get_audio_files(self, sample_size: int = None, random_sample: bool = False) -> List[Path]:
        """
        Get list of audio files to process.
        
        Args:
            sample_size: Limit to N files (for testing)
            random_sample: Use random sampling instead of first N
            
        Returns:
            List of audio file paths
        """
        audio_files = sorted(self.input_dir.glob('*.wav'))
        
        if sample_size:
            if random_sample:
                audio_files = random.sample(audio_files, min(sample_size, len(audio_files)))
            else:
                audio_files = audio_files[:sample_size]
        
        logger.info(f"Found {len(audio_files)} audio files to process")
        return audio_files
    
    def is_already_transcribed(self, audio_file: Path) -> bool:
        """Check if audio file has already been transcribed."""
        chat_file = self.chat_dir / f"{audio_file.stem}.cha"
        return chat_file.exists()
    
    def process_single_file(self, audio_file: Path) -> bool:
        """
        Process a single audio file.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract participant ID from filename (e.g., F1_01_01 -> F1)
            participant_id = audio_file.stem.split('_')[0]
            
            # Process audio
            result = self.processor.process(
                audio_path=audio_file,
                participant_id=participant_id,
                diagnosis='TD'  # TD = Typically Developing
            )
            
            if not result.success:
                logger.warning(f"No transcription generated for {audio_file.name}")
                return False
            
            # Save CHAT file
            chat_file = self.chat_dir / f"{audio_file.stem}.cha"
            self.processor.save_chat_file(result, chat_file)
            
            logger.debug(f"✓ Transcribed {audio_file.name} -> {chat_file.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {audio_file.name}: {e}")
            self.stats['error_files'].append({
                'file': str(audio_file),
                'error': str(e)
            })
            return False
    
    def process_batch(
        self,
        audio_files: List[Path],
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Process a batch of audio files.
        
        Args:
            audio_files: List of audio file paths
            show_progress: Show progress bar
            
        Returns:
            Statistics dictionary
        """
        self.stats['total_files'] = len(audio_files)
        self.stats['start_time'] = datetime.now()
        
        logger.info(f"Starting batch transcription of {len(audio_files)} files")
        logger.info(f"Model: {self.backend}/{self.model_size}")
        logger.info(f"Output: {self.chat_dir}")
        
        # Process files with progress bar
        iterator = tqdm(audio_files, desc="Transcribing") if show_progress else audio_files
        
        for audio_file in iterator:
            # Skip if already transcribed
            if self.resume and self.is_already_transcribed(audio_file):
                self.stats['skipped'] += 1
                if show_progress:
                    iterator.set_postfix({
                        'processed': self.stats['processed'],
                        'skipped': self.stats['skipped'],
                        'errors': self.stats['errors']
                    })
                continue
            
            # Process file
            success = self.process_single_file(audio_file)
            
            if success:
                self.stats['processed'] += 1
            else:
                self.stats['errors'] += 1
            
            # Update progress bar
            if show_progress:
                iterator.set_postfix({
                    'processed': self.stats['processed'],
                    'skipped': self.stats['skipped'],
                    'errors': self.stats['errors']
                })
        
        self.stats['end_time'] = datetime.now()
        
        # Calculate duration
        duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        self.stats['duration_seconds'] = duration
        self.stats['duration_formatted'] = self._format_duration(duration)
        
        # Calculate rate
        if self.stats['processed'] > 0:
            self.stats['seconds_per_file'] = duration / self.stats['processed']
        
        return self.stats
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    
    def save_summary(self):
        """Save summary statistics to JSON file."""
        summary_file = self.output_dir / 'summary_report.json'
        
        # Convert datetime to string for JSON serialization
        stats_copy = self.stats.copy()
        if stats_copy['start_time']:
            stats_copy['start_time'] = stats_copy['start_time'].isoformat()
        if stats_copy['end_time']:
            stats_copy['end_time'] = stats_copy['end_time'].isoformat()
        
        with open(summary_file, 'w') as f:
            json.dump(stats_copy, f, indent=2)
        
        logger.info(f"Summary saved to: {summary_file}")
    
    def print_summary(self):
        """Print summary statistics."""
        print("\n" + "="*60)
        print("BATCH TRANSCRIPTION SUMMARY")
        print("="*60)
        print(f"Total files:     {self.stats['total_files']}")
        print(f"Processed:       {self.stats['processed']}")
        print(f"Skipped:         {self.stats['skipped']} (already transcribed)")
        print(f"Errors:          {self.stats['errors']}")
        print(f"Duration:        {self.stats.get('duration_formatted', 'N/A')}")
        
        if self.stats.get('seconds_per_file'):
            print(f"Avg time/file:   {self.stats['seconds_per_file']:.1f}s")
        
        print(f"\nOutput directory: {self.chat_dir}")
        
        if self.stats['errors'] > 0:
            print(f"\nErrors logged to: {self.log_dir}")
        
        print("="*60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Batch transcribe TD audio files using Whisper"
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='data/td',
        help='Input directory containing TD audio files (default: data/td)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='output/td_transcripts',
        help='Output directory for transcripts (default: output/td_transcripts)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        default='tiny',
        help='Whisper model size (default: tiny)'
    )
    
    parser.add_argument(
        '--backend',
        type=str,
        choices=['faster-whisper', 'whisper', 'vosk', 'google'],
        default='faster-whisper',
        help='Transcription backend (default: faster-whisper)'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: process only a subset of files'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=50,
        help='Number of files to process in test mode (default: 50)'
    )
    
    parser.add_argument(
        '--random',
        action='store_true',
        help='Use random sampling instead of first N files'
    )
    
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Disable resume mode (re-transcribe all files)'
    )
    
    args = parser.parse_args()
    
    # Initialize transcriber
    transcriber = TDBatchTranscriber(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        model_size=args.model,
        backend=args.backend,
        resume=not args.no_resume
    )
    
    # Get audio files
    sample_size = args.sample_size if args.test else None
    audio_files = transcriber.get_audio_files(
        sample_size=sample_size,
        random_sample=args.random
    )
    
    if not audio_files:
        logger.error("No audio files found!")
        sys.exit(1)
    
    # Show configuration
    print("\n" + "="*60)
    print("BATCH TRANSCRIPTION CONFIGURATION")
    print("="*60)
    print(f"Input directory:  {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Model:            {args.backend}/{args.model}")
    print(f"Mode:             {'TEST' if args.test else 'FULL BATCH'}")
    print(f"Files to process: {len(audio_files)}")
    print(f"Resume mode:      {'ON' if not args.no_resume else 'OFF'}")
    print("="*60 + "\n")
    
    # Estimate duration
    if args.model == 'tiny':
        est_time_per_file = 10  # seconds
    elif args.model == 'base':
        est_time_per_file = 17
    else:
        est_time_per_file = 25
    
    total_est_seconds = est_time_per_file * len(audio_files)
    print(f"Estimated duration: {transcriber._format_duration(total_est_seconds)}")
    print(f"(assuming ~{est_time_per_file}s per file)\n")
    
    # Process batch
    stats = transcriber.process_batch(audio_files, show_progress=True)
    
    # Print and save summary
    transcriber.print_summary()
    transcriber.save_summary()
    
    # Log success
    if stats['errors'] == 0:
        logger.info("✓ All files processed successfully!")
    else:
        logger.warning(f"⚠ Completed with {stats['errors']} errors")
    
    return 0 if stats['errors'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
