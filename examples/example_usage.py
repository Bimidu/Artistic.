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


def example_1_parse_multiple_files():
    """
    Example 1: Parse multiple CHAT files from various ASDBank datasets
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Parse Multiple CHAT Files from ASDBank")
    print("="*70 + "\n")
    
    # Initialize parser
    parser = CHATParser()
    
    # Define multiple files to parse from different datasets
    test_files = [
        # Eigsti dataset - ASD group
        ("asdbank_eigsti/Eigsti/ASD/1010.cha", "ASD"),
        ("asdbank_eigsti/Eigsti/ASD/1012.cha", "ASD"),
        ("asdbank_eigsti/Eigsti/ASD/1017.cha", "ASD"),
        ("asdbank_eigsti/Eigsti/ASD/1020.cha", "ASD"),
        
        # Eigsti dataset - TD group
        ("asdbank_eigsti/Eigsti/TD/1011.cha", "TD"),
        ("asdbank_eigsti/Eigsti/TD/1013.cha", "TD"),
        ("asdbank_eigsti/Eigsti/TD/1022.cha", "TD"),
        ("asdbank_eigsti/Eigsti/TD/1024.cha", "TD"),
        
        # Flusberg dataset - different children
        ("asdbank_flusberg/Flusberg/Brett/050800.cha", "ASD"),
        ("asdbank_flusberg/Flusberg/Jack/060900.cha", "ASD"),
        ("asdbank_flusberg/Flusberg/Mark/070700.cha", "ASD"),
        
        # Nadig dataset
        ("asdbank_nadig/Nadig/101.cha", "ASD"),
        ("asdbank_nadig/Nadig/102.cha", "ASD"),
        ("asdbank_nadig/Nadig/103.cha", "ASD"),
        
        # AAC dataset
        ("asdbank_aac/AAC/01_T1_1.cha", "AAC"),
        ("asdbank_aac/AAC/01_T2.cha", "AAC"),
        ("asdbank_aac/AAC/02_T1_1.cha", "AAC"),
        
        # Rollins dataset
        ("asdbank_rollins/Rollins/Carl/030400.cha", "ASD"),
        ("asdbank_rollins/Rollins/Josh/030400.cha", "ASD"),
    ]
    
    parsed_count = 0
    total_files = len(test_files)
    
    print(f"Processing {total_files} CHAT files from multiple ASDBank datasets...\n")
    
    for file_path_str, expected_diagnosis in test_files:
        file_path = config.paths.data_dir / file_path_str
        
        if not file_path.exists():
            print(f"[WARNING]  File not found: {file_path_str}")
            continue
        
        try:
            # Parse the transcript
            transcript = parser.parse_file(file_path)
            parsed_count += 1
            
            # Display results
            print(f"[CHECK] [{parsed_count:2d}/{total_files}] {file_path.name}")
            print(f"   Dataset: {file_path_str.split('/')[0]}")
            print(f"   Participant: {transcript.participant_id}")
            print(f"   Diagnosis: {transcript.diagnosis or 'Unknown'}")
            print(f"   Age: {transcript.age_months or 'Unknown'} months")
            print(f"   Utterances: {transcript.total_utterances} total, {len(transcript.child_utterances)} child")
            print(f"   Valid Utterances: {len(transcript.valid_utterances)}")
            
            # Show a sample utterance if available
            if transcript.child_utterances:
                sample_utt = transcript.child_utterances[0]
                sample_text = sample_utt.text[:50] + "..." if len(sample_utt.text) > 50 else sample_utt.text
                print(f"   Sample: [{sample_utt.speaker}]: {sample_text}")
            
            print()
            
        except Exception as e:
            print(f"[X] Error parsing {file_path.name}: {e}")
            continue
    
    print(f"[CHART] SUMMARY:")
    print(f"   Successfully parsed: {parsed_count}/{total_files} files")
    print(f"   Success rate: {parsed_count/total_files*100:.1f}%")
    print(f"   Datasets covered: Eigsti, Flusberg, Nadig, AAC, Rollins")


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
    Example 4: Extract features from multiple datasets and directories
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Comprehensive Batch Feature Extraction")
    print("="*70 + "\n")
    
    # Initialize extractor
    extractor = FeatureExtractor(categories='pragmatic_conversational')
    
    # Define multiple directories to process
    datasets_to_process = [
        # Eigsti dataset - ASD and TD groups
        ("asdbank_eigsti/Eigsti/ASD", "eigsti_asd_features.csv", "ASD"),
        ("asdbank_eigsti/Eigsti/TD", "eigsti_td_features.csv", "TD"),
        ("asdbank_eigsti/Eigsti/DD", "eigsti_dd_features.csv", "DD"),
        
        # Flusberg dataset - different children
        ("asdbank_flusberg/Flusberg/Brett", "flusberg_brett_features.csv", "ASD"),
        ("asdbank_flusberg/Flusberg/Jack", "flusberg_jack_features.csv", "ASD"),
        ("asdbank_flusberg/Flusberg/Mark", "flusberg_mark_features.csv", "ASD"),
        
        # Nadig dataset
        ("asdbank_nadig/Nadig", "nadig_features.csv", "ASD"),
        
        # AAC dataset (sample)
        ("asdbank_aac/AAC", "aac_features.csv", "AAC"),
        
        # Rollins dataset
        ("asdbank_rollins/Rollins", "rollins_features.csv", "ASD"),
    ]
    
    all_results = []
    total_processed = 0
    
    print(f"Processing {len(datasets_to_process)} datasets...\n")
    
    for dataset_path, output_filename, diagnosis in datasets_to_process:
        dataset_dir = config.paths.data_dir / dataset_path
        
        if not dataset_dir.exists():
            print(f"[WARNING]  Directory not found: {dataset_path}")
            continue
        
        try:
            print(f"[DIR] Processing: {dataset_path}")
            
            # Extract features and save to CSV
            output_file = config.paths.output_dir / output_filename
            
            df = extractor.extract_from_directory(
                directory=dataset_dir,
                output_file=output_file
            )
            
            if df is not None and not df.empty:
                # Add diagnosis column
                df['diagnosis'] = diagnosis
                df['dataset'] = dataset_path.split('/')[0]
                
                all_results.append(df)
                total_processed += len(df)
                
                print(f"   [CHECK] Extracted {len(df)} samples")
                print(f"   [DISK] Saved to: {output_filename}")
                print(f"   [CHART] Features: {len([c for c in df.columns if c in extractor.all_feature_names])}")
            else:
                print(f"   [WARNING]  No valid samples found")
            
            print()
            
        except Exception as e:
            print(f"   [X] Error processing {dataset_path}: {e}")
            continue
    
    # Combine all results
    if all_results:
        print("[REFRESH] Combining all datasets...")
        import pandas as pd
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Save combined dataset
        combined_output = config.paths.output_dir / "all_asdbank_features.csv"
        combined_df.to_csv(combined_output, index=False)
        
        print(f"[CHART] COMPREHENSIVE SUMMARY:")
        print(f"   Total samples processed: {total_processed}")
        print(f"   Total datasets: {len(all_results)}")
        print(f"   Combined features: {combined_df.shape}")
        print(f"   [DISK] Combined dataset saved to: {combined_output}")
        
        # Show diagnosis distribution
        print(f"\n[GRAPH] Diagnosis Distribution:")
        diagnosis_counts = combined_df['diagnosis'].value_counts()
        for diagnosis, count in diagnosis_counts.items():
            print(f"   {diagnosis}: {count} samples")
        
        # Show dataset distribution
        print(f"\n[DATASET] Dataset Distribution:")
        dataset_counts = combined_df['dataset'].value_counts()
        for dataset, count in dataset_counts.items():
            print(f"   {dataset}: {count} samples")
        
        # Show feature statistics for key features
        print(f"\n[CHART] Key Feature Statistics:")
        key_features = ['mlu_words', 'type_token_ratio', 'turns_per_minute', 'echolalia_ratio']
        
        for feature in key_features:
            if feature in combined_df.columns:
                stats = combined_df[feature].describe()
                print(f"   {feature}:")
                print(f"     Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
                print(f"     Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
    
    else:
        print("[X] No datasets were successfully processed")


def example_5_process_all_chat_files():
    """
    Example 5: Process ALL available CHAT files in the ASDBank dataset
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Process ALL CHAT Files in ASDBank")
    print("="*70 + "\n")
    
    # Initialize parser
    parser = CHATParser()
    
    # Find all CHAT files recursively
    data_dir = config.paths.data_dir
    all_cha_files = list(data_dir.rglob("*.cha"))
    
    print(f"[SEARCH] Found {len(all_cha_files)} CHAT files in ASDBank dataset")
    print(f"[DIR] Searching in: {data_dir}")
    
    if not all_cha_files:
        print("[X] No CHAT files found!")
        return
    
    # Process files in batches
    batch_size = 50  # Process 50 files at a time
    total_batches = (len(all_cha_files) + batch_size - 1) // batch_size
    
    all_transcripts = []
    success_count = 0
    error_count = 0
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(all_cha_files))
        batch_files = all_cha_files[start_idx:end_idx]
        
        print(f"\n[BATCH] Processing batch {batch_num + 1}/{total_batches} ({len(batch_files)} files)")
        
        batch_success = 0
        batch_errors = 0
        
        for i, file_path in enumerate(batch_files):
            try:
                # Parse the transcript
                transcript = parser.parse_file(file_path)
                all_transcripts.append(transcript)
                batch_success += 1
                success_count += 1
                
                # Progress indicator
                if (i + 1) % 10 == 0 or (i + 1) == len(batch_files):
                    print(f"   [CHECK] {i + 1}/{len(batch_files)} files processed")
                
            except Exception as e:
                batch_errors += 1
                error_count += 1
                if batch_errors <= 3:  # Show first 3 errors per batch
                    print(f"   [X] Error in {file_path.name}: {str(e)[:50]}...")
        
        print(f"   [CHART] Batch {batch_num + 1} complete: {batch_success} success, {batch_errors} errors")
    
    # Generate comprehensive summary
    print(f"\n" + "="*70)
    print("[CHART] COMPREHENSIVE ASDBANK PROCESSING SUMMARY")
    print("="*70)
    
    print(f"[DIR] Total CHAT files found: {len(all_cha_files)}")
    print(f"[CHECK] Successfully parsed: {success_count}")
    print(f"[X] Errors encountered: {error_count}")
    print(f"[GRAPH] Success rate: {success_count/(success_count + error_count)*100:.1f}%")
    
    if all_transcripts:
        # Analyze dataset distribution
        dataset_counts = {}
        diagnosis_counts = {}
        total_utterances = 0
        total_child_utterances = 0
        
        for transcript in all_transcripts:
            # Count by dataset
            dataset = transcript.file_path.parent.parent.parent.name if transcript.file_path else "unknown"
            dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
            
            # Count by diagnosis
            diagnosis = transcript.diagnosis or "Unknown"
            diagnosis_counts[diagnosis] = diagnosis_counts.get(diagnosis, 0) + 1
            
            # Count utterances
            total_utterances += transcript.total_utterances
            total_child_utterances += len(transcript.child_utterances)
        
        print(f"\n[DATASET] Dataset Distribution:")
        for dataset, count in sorted(dataset_counts.items()):
            print(f"   {dataset}: {count} files")
        
        print(f"\n[DIAGNOSIS] Diagnosis Distribution:")
        for diagnosis, count in sorted(diagnosis_counts.items()):
            print(f"   {diagnosis}: {count} files")
        
        print(f"\n[SPEECH] Utterance Statistics:")
        print(f"   Total utterances: {total_utterances:,}")
        print(f"   Child utterances: {total_child_utterances:,}")
        print(f"   Average utterances per file: {total_utterances/len(all_transcripts):.1f}")
        print(f"   Average child utterances per file: {total_child_utterances/len(all_transcripts):.1f}")
        
        # Show age distribution if available
        ages = [t.age_months for t in all_transcripts if t.age_months is not None]
        if ages:
            print(f"\n[AGE] Age Distribution:")
            print(f"   Age range: {min(ages)} - {max(ages)} months")
            print(f"   Average age: {sum(ages)/len(ages):.1f} months")
    
    print(f"\n[SUCCESS] ASDBank processing complete!")
    print(f"[DISK] All parsed transcripts available in memory for further analysis")


def example_6_compare_asd_vs_td():
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
    """Run comprehensive examples with multiple CHAT files"""
    
    # Setup logging
    setup_logger()
    
    print("\n" + "="*70)
    print("ASD DETECTION FEATURE EXTRACTION - COMPREHENSIVE EXAMPLES")
    print("Phase 1 & 2: CHAT Parsing and Feature Extraction")
    print("Processing Multiple ASDBank Datasets")
    print("="*70)
    
    # Run the comprehensive examples:
    
    # Example 1: Parse multiple files from different datasets
    example_1_parse_multiple_files()
    
    # Example 4: Comprehensive batch feature extraction from multiple datasets
    example_4_extract_features_batch()
    
    # Example 5: Process ALL CHAT files in ASDBank (comprehensive analysis)
    # Uncomment the line below to process ALL available CHAT files (may take longer)
    # example_5_process_all_chat_files()
    
    # Other examples (commented out to save time, uncomment as needed):
    # example_2_build_inventory()           # Build complete dataset inventory
    # example_3_extract_features_single()   # Extract features from single transcript
    # example_6_compare_asd_vs_td()         # Compare ASD vs TD groups
    
    print("\n" + "="*70)
    print("[SUCCESS] COMPREHENSIVE EXAMPLES COMPLETED!")
    print("[CHART] Multiple ASDBank datasets processed successfully")
    print("[DISK] Feature files saved to output directory")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

