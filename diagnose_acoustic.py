"""
Quick diagnostic test to identify where acoustic processing is getting stuck.
"""

from pathlib import Path
from src.pipeline.acoustic_dataset_preparation import prepare_acoustic_training_data
from src.features.acoustic_prosodic.acoustic_extractor import AcousticFeatureExtractor
import time
import signal
import sys


def timeout_handler(signum, frame):
    print("\n⏰ TIMEOUT: Process took too long, likely stuck")
    sys.exit(1)


def diagnose_acoustic_processing():
    """Run diagnostic test with timeout protection."""
    print("="*70)
    print("ACOUSTIC PROCESSING DIAGNOSTIC TEST")
    print("="*70)

    # Set timeout (5 minutes max) - Only works on Unix
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(300)  # 5 minute timeout
    else:
        print("ℹ️  Note: Timeout protection not available on Windows, skipping signal setup.")

    try:
        print("Step 1: Testing preparation phase...")
        start_time = time.time()

        td_path = Path('data/td')
        if not td_path.exists():
            print("❌ TD dataset not found")
            return

        # Test with minimal data first
        prepared = prepare_acoustic_training_data(
            [td_path],
            n_per_group=2,  # Very small groups
            max_merged_per_diagnosis=2,  # Just 2 groups
            random_state=42
        )

        prep_time = time.time() - start_time
        print(f"✅ Preparation: {prep_time:.1f}s - {len(prepared)} groups created")

        if not prepared:
            print("❌ No groups prepared")
            return

        print("\nStep 2: Testing feature extraction...")
        print("This is where it might get stuck...")

        extractor = AcousticFeatureExtractor()

        # Test with just 1 group first
        print("Testing with 1 group only...")
        test_group = [prepared[0]]

        start_time = time.time()
        try:
            features_df = extractor.extract_from_prepared_groups(test_group)
            extract_time = time.time() - start_time
            print(f"✅ Single group extraction: {extract_time:.1f}s")
            print(f"📊 Features: {len(features_df)} rows, {len(features_df.columns)} columns")

            # If that works, try all groups
            if len(prepared) > 1:
                print(f"\nTesting all {len(prepared)} groups...")
                start_time = time.time()
                features_df = extractor.extract_from_prepared_groups(prepared)
                extract_time = time.time() - start_time
                print(f"✅ Full extraction: {extract_time:.1f}s")
                print(f"📊 Features: {len(features_df)} rows, {len(features_df.columns)} columns")

        except Exception as e:
            extract_time = time.time() - start_time
            print(f"❌ Extraction failed after {extract_time:.1f}s: {e}")
            print("\nDetailed error:")
            import traceback
            traceback.print_exc()

    except Exception as e:
        print(f"❌ Overall test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if hasattr(signal, 'alarm'):
            signal.alarm(0)  # Cancel timeout

    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)


if __name__ == "__main__":
    try:
        diagnose_acoustic_processing()
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except SystemExit:
        print("\n⏰ Test timed out - process was stuck")
