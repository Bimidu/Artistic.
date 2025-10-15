"""
Example Usage Scripts for ASD Detection Feature Extraction

This script demonstrates how to use the implemented Phase 1 & 2 functionality:
1. Parse CHAT files
2. Build dataset inventory
3. Extract pragmatic & conversational features

Author: Bimidu Gunathilake
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parsers.chat_parser import CHATParser
from src.parsers.dataset_inventory import DatasetInventory
from src.features.feature_extractor import FeatureExtractor
from src.utils.logger import setup_logger
from config import config


def example_1_parse_single_file():
    """
    Example 1: Parse a single CHAT transcript file
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Parse Single CHAT File")
    print("="*70 + "\n")
    
    # Initialize parser
    parser = CHATParser()
    
    # Parse a file (update path to your data)
    file_path = config.paths.data_dir / "asdbank_eigsti/Eigsti/ASD/1010.cha"
    
    if not file_path.exists():
        print(f"File not found: {file_path}")
        print("Please update the path to point to a valid .cha file")
        return
    
    # Parse the transcript
    transcript = parser.parse_file(file_path)
    
    # Display results
    print(f"Participant ID: {transcript.participant_id}")
    print(f"Diagnosis: {transcript.diagnosis}")
    print(f"Age: {transcript.age_months} months")
    print(f"Total Utterances: {transcript.total_utterances}")
    print(f"Child Utterances: {len(transcript.child_utterances)}")
    print(f"Valid Utterances: {len(transcript.valid_utterances)}")
    
    # Show first few utterances
    print(f"\nFirst 3 child utterances:")
    for i, utt in enumerate(transcript.child_utterances[:3], 1):
        print(f"  {i}. [{utt.speaker}]: {utt.text}")


def example_2_build_inventory():
    """
    Example 2: Build complete dataset inventory
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Build Dataset Inventory")
    print("="*70 + "\n")
    
    # Initialize inventory
    inventory = DatasetInventory()
    
    # Build inventory (this will cache results)
    # Note: This may take a few minutes for first run
    inventory.build_inventory(
        datasets=['asdbank_eigsti'],  # Start with one dataset
        force_rebuild=False  # Use cache if available
    )
    
    # Get summary
    summary = inventory.get_dataset_summary()
    print("Dataset Summary:")
    for dataset, counts in summary.items():
        print(f"\n  {dataset}:")
        for diagnosis, count in counts.items():
            print(f"    {diagnosis}: {count} participants")
    
    # Get DataFrame
    df = inventory.to_dataframe()
    print(f"\nInventory DataFrame shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Filter by diagnosis
    asd_participants = inventory.get_participants_by_diagnosis('ASD')
    td_participants = inventory.get_participants_by_diagnosis('TD')
    
    print(f"\nASD Participants: {len(asd_participants)}")
    print(f"TD Participants: {len(td_participants)}")
    
    # Export to CSV
    output_path = config.paths.output_dir / "inventory.csv"
    inventory.export_to_csv(output_path)
    print(f"\nInventory exported to: {output_path}")


def example_3_extract_features_single():
    """
    Example 3: Extract features from a single transcript
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Extract Features from Single Transcript")
    print("="*70 + "\n")
    
    # Parse transcript
    parser = CHATParser()
    file_path = config.paths.data_dir / "asdbank_eigsti/Eigsti/ASD/1010.cha"
    
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return
    
    transcript = parser.parse_file(file_path)
    
    # Initialize feature extractor (pragmatic & conversational only)
    extractor = FeatureExtractor(categories='pragmatic_conversational')
    
    # Print category info
    extractor.print_category_info()
    
    # Extract features
    feature_set = extractor.extract_from_transcript(transcript)
    
    # Display results
    print(f"Participant: {feature_set.participant_id}")
    print(f"Diagnosis: {feature_set.diagnosis}")
    print(f"Total features extracted: {len(feature_set.features)}")
    print(f"\nSample features:")
    
    # Show some key features
    key_features = [
        'mlu_words',
        'type_token_ratio',
        'turns_per_minute',
        'echolalia_ratio',
        'question_ratio',
        'topic_shift_ratio'
    ]
    
    for feature in key_features:
        if feature in feature_set.features:
            print(f"  {feature}: {feature_set.features[feature]:.3f}")


def example_4_extract_features_batch():
    """
    Example 4: Extract features from multiple files
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Batch Feature Extraction")
    print("="*70 + "\n")
    
    # Initialize extractor
    extractor = FeatureExtractor(categories='pragmatic_conversational')
    
    # Extract from directory
    dataset_dir = config.paths.data_dir / "asdbank_eigsti/Eigsti/ASD"
    
    if not dataset_dir.exists():
        print(f"Directory not found: {dataset_dir}")
        return
    
    # Extract features and save to CSV
    output_file = config.paths.output_dir / "eigsti_asd_features.csv"
    
    print(f"Extracting features from: {dataset_dir}")
    print(f"Output will be saved to: {output_file}\n")
    
    df = extractor.extract_from_directory(
        directory=dataset_dir,
        output_file=output_file
    )
    
    # Display summary
    print(f"\nExtracted features: {df.shape}")
    print(f"\nFeature columns:")
    feature_cols = [c for c in df.columns if c in extractor.all_feature_names]
    for i, col in enumerate(feature_cols[:10], 1):
        print(f"  {i}. {col}")
    if len(feature_cols) > 10:
        print(f"  ... and {len(feature_cols) - 10} more features")
    
    # Show statistics
    summary = extractor.get_feature_summary(df)
    print(f"\nSummary Statistics:")
    print(f"  Total samples: {summary['total_samples']}")
    print(f"  Total features: {summary['feature_count']}")
    
    # Show some feature statistics
    if 'mlu_words' in summary['feature_stats']:
        mlu_stats = summary['feature_stats']['mlu_words']
        print(f"\nMLU Words:")
        print(f"  Mean: {mlu_stats['mean']:.3f}")
        print(f"  Std: {mlu_stats['std']:.3f}")
        print(f"  Range: [{mlu_stats['min']:.3f}, {mlu_stats['max']:.3f}]")


def example_5_compare_asd_vs_td():
    """
    Example 5: Compare features between ASD and TD groups
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Compare ASD vs TD Features")
    print("="*70 + "\n")
    
    # Extract features from both groups
    extractor = FeatureExtractor(categories='pragmatic_conversational')
    
    # ASD group
    asd_dir = config.paths.data_dir / "asdbank_eigsti/Eigsti/ASD"
    # TD group
    td_dir = config.paths.data_dir / "asdbank_eigsti/Eigsti/TD"
    
    if not (asd_dir.exists() and td_dir.exists()):
        print("Required directories not found")
        return
    
    print("Extracting ASD features...")
    asd_df = extractor.extract_from_directory(asd_dir)
    asd_df['diagnosis'] = 'ASD'
    
    print("\nExtracting TD features...")
    td_df = extractor.extract_from_directory(td_dir)
    td_df['diagnosis'] = 'TD'
    
    # Combine
    import pandas as pd
    combined_df = pd.concat([asd_df, td_df], ignore_index=True)
    
    # Compare key features
    print("\n" + "="*70)
    print("FEATURE COMPARISON: ASD vs TD")
    print("="*70)
    
    key_features = [
        'mlu_words',
        'type_token_ratio',
        'echolalia_ratio',
        'question_ratio',
        'pronoun_reversal_count',
        'topic_shift_ratio'
    ]
    
    print(f"\n{'Feature':<30} {'ASD Mean':<15} {'TD Mean':<15} {'Difference':<15}")
    print("-" * 75)
    
    for feature in key_features:
        if feature in combined_df.columns:
            asd_mean = combined_df[combined_df['diagnosis'] == 'ASD'][feature].mean()
            td_mean = combined_df[combined_df['diagnosis'] == 'TD'][feature].mean()
            diff = asd_mean - td_mean
            
            print(f"{feature:<30} {asd_mean:<15.3f} {td_mean:<15.3f} {diff:<15.3f}")
    
    # Save combined features
    output_file = config.paths.output_dir / "asd_vs_td_features.csv"
    combined_df.to_csv(output_file, index=False)
    print(f"\nCombined features saved to: {output_file}")


def main():
    """Run all examples"""
    
    # Setup logging
    setup_logger()
    
    print("\n" + "="*70)
    print("ASD DETECTION FEATURE EXTRACTION - EXAMPLES")
    print("Phase 1 & 2: CHAT Parsing and Feature Extraction")
    print("="*70)
    
    # Uncomment the examples you want to run:
    
    example_1_parse_single_file()
    # example_2_build_inventory()
    # example_3_extract_features_single()
    # example_4_extract_features_batch()
    # example_5_compare_asd_vs_td()
    
    print("\n" + "="*70)
    print("Examples completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

