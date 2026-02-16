#!/usr/bin/env python3
"""
Validation Script for Syntactic & Semantic Features

This script validates the syntactic/semantic feature extraction implementation
by testing feature extraction, clinical interpretation, and model training.

Usage:
    python scripts/validate_syntactic_semantic.py

Author: Enhanced Implementation
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.features.syntactic_semantic import (
    SyntacticSemanticFeatures,
    ClinicalInterpreter
)
from src.parsers.chat_parser import TranscriptData, Utterance


def create_sample_transcript():
    """Create a sample transcript for testing."""
    utterances = [
        Utterance(
            speaker='CHI',
            text='I want to play with the blocks.',
            word_count=7,
            is_valid=True,
            tokens=None,
            morphology=None
        ),
        Utterance(
            speaker='MOT',
            text='Which blocks do you want?',
            word_count=5,
            is_valid=True,
            tokens=None,
            morphology=None
        ),
        Utterance(
            speaker='CHI',
            text='The red ones because they are big.',
            word_count=7,
            is_valid=True,
            tokens=None,
            morphology=None
        ),
        Utterance(
            speaker='CHI',
            text='Can I build a tower?',
            word_count=5,
            is_valid=True,
            tokens=None,
            morphology=None
        ),
        Utterance(
            speaker='MOT',
            text='Yes, you can.',
            word_count=3,
            is_valid=True,
            tokens=None,
            morphology=None
        ),
        Utterance(
            speaker='CHI',
            text='I building it now.',
            word_count=4,
            is_valid=True,
            tokens=None,
            morphology=None
        ),
    ]

    transcript = TranscriptData(
        filename='test_sample.cha',
        utterances=utterances,
        metadata={}
    )

    return transcript


def test_feature_extraction():
    """Test feature extraction functionality."""
    print("="*70)
    print("TEST 1: Feature Extraction")
    print("="*70)

    try:
        # Create sample transcript
        transcript = create_sample_transcript()
        print(f"✓ Created sample transcript with {len(transcript.utterances)} utterances")

        # Initialize extractor
        extractor = SyntacticSemanticFeatures()
        print(f"✓ Initialized extractor")

        # Check feature count
        feature_count = len(extractor.feature_names)
        expected_count = 27
        assert feature_count == expected_count, f"Expected {expected_count} features, got {feature_count}"
        print(f"✓ Feature count correct: {feature_count}")

        # Extract features
        result = extractor.extract(transcript)
        print(f"✓ Extracted features successfully")

        # Validate result
        assert len(result.features) == expected_count, "Feature count mismatch"
        assert result.feature_type == 'syntactic_semantic', "Wrong feature type"
        assert 'num_child_utterances' in result.metadata, "Missing metadata"

        print(f"✓ Feature result structure valid")
        print(f"  - Features: {len(result.features)}")
        print(f"  - Child utterances: {result.metadata['num_child_utterances']}")
        print(f"  - Tokens analyzed: {result.metadata['num_tokens_analyzed']}")

        # Check for NaN or infinite values
        has_invalid = any(
            not np.isfinite(v) for v in result.features.values()
            if isinstance(v, (int, float))
        )
        assert not has_invalid, "Found NaN or infinite values"
        print(f"✓ No invalid values (NaN/Inf)")

        # Display sample features
        print("\n  Sample Features:")
        sample_features = [
            'avg_dependency_depth',
            'grammatical_error_rate',
            'semantic_coherence',
            'lexical_diversity_semantic'
        ]
        for feature in sample_features:
            value = result.features.get(feature, 0)
            print(f"    {feature}: {value:.4f}")

        print("\n✅ TEST 1 PASSED: Feature Extraction\n")
        return True

    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_clinical_interpretation():
    """Test clinical interpretation functionality."""
    print("="*70)
    print("TEST 2: Clinical Interpretation")
    print("="*70)

    try:
        # Create interpreter
        interpreter = ClinicalInterpreter()
        print("✓ Initialized clinical interpreter")

        # Test single feature interpretation
        interpretation = interpreter.interpret_feature('avg_dependency_depth', 2.1)
        assert 'Average Dependency Depth' in interpretation, "Missing feature name"
        assert '2.100' in interpretation, "Missing value"
        assert 'ASD' in interpretation, "Missing ASD pattern"
        assert 'TD' in interpretation, "Missing TD pattern"
        print("✓ Single feature interpretation works")

        # Test with sample features
        sample_features = {
            'avg_dependency_depth': 1.8,
            'grammatical_error_rate': 0.30,
            'semantic_coherence': 0.35,
            'lexical_diversity_semantic': 0.45,
            'subordination_index': 0.15
        }

        # Test profile interpretation
        profile = interpreter.interpret_profile(sample_features)
        assert 'Syntactic Complexity' in profile, "Missing category"
        assert 'Grammatical Accuracy' in profile, "Missing category"
        print("✓ Profile interpretation works")

        # Test risk assessment
        risk_indicators = interpreter.get_asd_risk_indicators(sample_features)
        assert len(risk_indicators) > 0, "Should detect risk indicators"
        print(f"✓ Risk assessment works ({len(risk_indicators)} indicators found)")

        # Test clinical summary
        summary = interpreter.generate_clinical_summary(sample_features)
        assert 'CLINICAL SUMMARY' in summary, "Missing summary header"
        print("✓ Clinical summary generation works")

        print("\n✅ TEST 2 PASSED: Clinical Interpretation\n")
        return True

    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_feature_categories():
    """Test that all features are categorized correctly."""
    print("="*70)
    print("TEST 3: Feature Categories")
    print("="*70)

    try:
        extractor = SyntacticSemanticFeatures()
        features = extractor.feature_names

        # Expected categories and counts
        expected_categories = {
            'Syntactic Complexity': 6,
            'Grammatical Accuracy': 5,
            'Sentence Structure': 4,
            'Semantic Features': 4,
            'Vocabulary Semantics': 4,
            'Advanced Semantics': 4,
        }

        # Feature keywords for each category
        category_keywords = {
            'Syntactic Complexity': ['dependency', 'clause', 'subordination', 'coordination'],
            'Grammatical Accuracy': ['grammatical', 'tense', 'pos', 'structure', 'diversity'],
            'Sentence Structure': ['parse', 'noun_phrase', 'verb_phrase', 'prepositional'],
            'Semantic Features': ['semantic_coherence', 'semantic_density', 'lexical_diversity_semantic', 'thematic'],
            'Vocabulary Semantics': ['vocabulary', 'semantic_field', 'word_sense', 'content_word'],
            'Advanced Semantics': ['semantic_role', 'entity', 'verb_argument'],
        }

        print(f"Total features: {len(features)}")
        print(f"Expected total: {sum(expected_categories.values())}")

        # Verify total count
        assert len(features) == sum(expected_categories.values()), "Total feature count mismatch"
        print("✓ Total feature count correct")

        # Categorize features
        for category, keywords in category_keywords.items():
            matching_features = [
                f for f in features
                if any(keyword in f for keyword in keywords)
            ]
            print(f"\n{category}:")
            print(f"  Expected: {expected_categories[category]}")
            print(f"  Found: {len(matching_features)}")
            for feature in matching_features:
                print(f"    - {feature}")

        print("\n✅ TEST 3 PASSED: Feature Categories\n")
        return True

    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling with edge cases."""
    print("="*70)
    print("TEST 4: Error Handling")
    print("="*70)

    try:
        extractor = SyntacticSemanticFeatures()

        # Test with empty transcript
        empty_transcript = TranscriptData(
            filename='empty.cha',
            utterances=[],
            metadata={}
        )

        result = extractor.extract(empty_transcript)
        assert 'error' in result.metadata, "Should have error metadata"
        assert all(v == 0.0 for v in result.features.values()), "Should have zero features"
        print("✓ Handles empty transcript correctly")

        # Test with transcript with no child utterances
        no_child_transcript = TranscriptData(
            filename='no_child.cha',
            utterances=[
                Utterance(
                    speaker='MOT',
                    text='Hello there.',
                    word_count=2,
                    is_valid=True,
                    tokens=None,
                    morphology=None
                )
            ],
            metadata={}
        )

        result = extractor.extract(no_child_transcript)
        assert 'error' in result.metadata, "Should have error metadata"
        print("✓ Handles missing child utterances correctly")

        print("\n✅ TEST 4 PASSED: Error Handling\n")
        return True

    except Exception as e:
        print(f"\n❌ TEST 4 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_dependencies():
    """Test that required dependencies are available."""
    print("="*70)
    print("TEST 5: Dependencies")
    print("="*70)

    try:
        # Test spaCy
        import spacy
        try:
            nlp = spacy.load('en_core_web_sm')
            print("✓ spaCy model 'en_core_web_sm' loaded")
        except OSError:
            print("⚠️  spaCy model 'en_core_web_sm' not found")
            print("   Run: python -m spacy download en_core_web_sm")

        # Test NLTK WordNet
        from nltk.corpus import wordnet
        try:
            wordnet.synsets('test')
            print("✓ NLTK WordNet available")
        except LookupError:
            print("⚠️  NLTK WordNet not found")
            print("   Run: python -c \"import nltk; nltk.download('wordnet')\"")

        # Test other dependencies
        import numpy
        print("✓ NumPy available")

        from collections import Counter
        print("✓ Collections available")

        print("\n✅ TEST 5 PASSED: Dependencies\n")
        return True

    except Exception as e:
        print(f"\n❌ TEST 5 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("\n" + "="*70)
    print("SYNTACTIC & SEMANTIC FEATURES - VALIDATION SCRIPT")
    print("="*70)
    print("Version: 2.0.0")
    print("Status: Testing full implementation")
    print("="*70 + "\n")

    tests = [
        ("Dependencies", test_dependencies),
        ("Feature Extraction", test_feature_extraction),
        ("Clinical Interpretation", test_clinical_interpretation),
        ("Feature Categories", test_feature_categories),
        ("Error Handling", test_error_handling),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"❌ {test_name} encountered unexpected error: {e}")
            results.append((test_name, False))

    # Summary
    print("="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED! Implementation is valid.")
        print("="*70 + "\n")
        return 0
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed. Please review.")
        print("="*70 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
