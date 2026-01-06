"""
Test script for syntactic_semantic feature extraction

This script demonstrates extracting syntactic and semantic features
from a sample CHAT transcript and displays the output.
"""

import sys
from pathlib import Path
from typing import Union

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.parsers.chat_parser import CHATParser
from src.features.syntactic_semantic.syntactic_semantic import SyntacticSemanticFeatures
from config import config


def main():
    """Extract and display syntactic/semantic features from a sample transcript."""

    print("\n" + "="*80)
    print("SYNTACTIC & SEMANTIC FEATURE EXTRACTION TEST")
    print("="*80 + "\n")

    # Find a sample CHAT file
    sample_file = Path("/Users/user/PycharmProjects/Artistic./data/asdbank_flusberg/Flusberg/Brett/060819.cha")

    if not sample_file.exists():
        print(f"[X] Sample file not found: {sample_file}")
        return

    print(f"[FILE] Processing: {sample_file.name}")
    print(f"[DIR] Path: {sample_file}\n")

    # Parse the transcript
    print("[1/3] Parsing CHAT transcript...")
    parser = CHATParser()
    transcript = parser.parse_file(sample_file)

    print(f"  [CHECK] Parsed successfully")
    print(f"  - Participant ID: {transcript.participant_id}")
    print(f"  - Total utterances: {transcript.total_utterances}")
    print(f"  - Child utterances: {len(transcript.child_utterances)}")
    print()

    # Initialize syntactic/semantic feature extractor
    print("[2/3] Initializing syntactic/semantic feature extractor...")
    extractor = SyntacticSemanticFeatures()
    print(f"  [CHECK] Extractor initialized")
    print(f"  - Features to extract: {len(extractor.feature_names)}\n")

    # Extract features
    print("[3/3] Extracting syntactic/semantic features...")
    feature_result = extractor.extract(transcript)

    print(f"  [CHECK] Features extracted successfully")
    print(f"  - Feature type: {feature_result.feature_type}")
    print(f"  - Status: {feature_result.metadata.get('status', 'unknown')}")
    print(f"  - Child utterances analyzed: {feature_result.metadata.get('num_child_utterances', 0)}")
    print(f"  - Tokens analyzed: {feature_result.metadata.get('num_tokens_analyzed', 0)}")
    print()

    # Display extracted features
    print("="*80)
    print("EXTRACTED FEATURES")
    print("="*80 + "\n")

    # Group features by category
    syntactic_features = [f for f in extractor.feature_names if any(x in f for x in
                         ['dependency', 'clause', 'subordination', 'coordination'])]
    grammatical_features = [f for f in extractor.feature_names if any(x in f for x in
                           ['grammatical', 'tense', 'pos', 'structure', 'diversity'])]
    semantic_features = [f for f in extractor.feature_names if any(x in f for x in
                        ['semantic', 'coherence', 'thematic', 'entity', 'role'])]
    vocabulary_features = [f for f in extractor.feature_names if any(x in f for x in
                          ['vocabulary', 'word', 'lexical', 'content'])]
    structure_features = [f for f in extractor.feature_names if any(x in f for x in
                         ['parse', 'phrase', 'prepositional'])]

    # Display Syntactic Complexity Features
    print("[TREE] Syntactic Complexity Features:")
    for feature in syntactic_features:
        value = feature_result.features.get(feature, 0.0)
        print(f"  {feature:<35} = {value:>8.4f}")
    print()

    # Display Grammatical Features
    print("[CHECK] Grammatical Features:")
    for feature in grammatical_features:
        value = feature_result.features.get(feature, 0.0)
        print(f"  {feature:<35} = {value:>8.4f}")
    print()

    # Display Sentence Structure Features
    print("[BRACKETS] Sentence Structure Features:")
    for feature in structure_features:
        value = feature_result.features.get(feature, 0.0)
        print(f"  {feature:<35} = {value:>8.4f}")
    print()

    # Display Semantic Features
    print("[BRAIN] Semantic Features:")
    for feature in semantic_features:
        value = feature_result.features.get(feature, 0.0)
        print(f"  {feature:<35} = {value:>8.4f}")
    print()

    # Display Vocabulary Semantic Features
    print("[BOOK] Vocabulary Semantic Features:")
    for feature in vocabulary_features:
        value = feature_result.features.get(feature, 0.0)
        print(f"  {feature:<35} = {value:>8.4f}")
    print()

    # Summary statistics
    print("="*80)
    print("SUMMARY STATISTICS")
    print("="*80 + "\n")

    import numpy as np
    feature_values = list(feature_result.features.values())

    print(f"Total features extracted: {len(feature_values)}")
    print(f"Non-zero features: {sum(1 for v in feature_values if v > 0)}")
    print(f"Mean feature value: {np.mean(feature_values):.4f}")
    print(f"Std feature value: {np.std(feature_values):.4f}")
    print(f"Min feature value: {np.min(feature_values):.4f}")
    print(f"Max feature value: {np.max(feature_values):.4f}")

    print("\n" + "="*80)
    print("[SUCCESS] Feature extraction completed!")
    print("="*80 + "\n")

    # Show sample utterances
    print("Sample Child Utterances (first 3):")
    for i, utt in enumerate(transcript.child_utterances[:3]):
        print(f"  {i+1}. [{utt.speaker}]: {utt.text}")
    print()


if __name__ == "__main__":
    main()
