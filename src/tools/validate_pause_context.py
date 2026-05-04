
"""
Pause Context Validation Tool

A diagnostic script to verify that pause duration statistics differ meaningfully
between two types of child responses in the ASDBank dataset:

  - Backchannel responses (< 2 words): minimal acknowledgments like "yes", "okay"
  - Substantive responses  (>= 2 words): real turns where the child forms an utterance

Motivation: The pause/latency feature extractor uses a single response-latency
threshold, but a very short child response may have a shorter natural latency than
a full utterance. Running this script on the corpus confirms (or refutes) whether
the 2-word boundary is a useful split before deciding whether to apply separate
thresholds in the live extractor.

Usage:
    python -m src.tools.validate_pause_context
"""

import sys
import numpy as np
from pathlib import Path
import pylangacq

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.parsers.chat_parser import CHATParser

def validate_context():
    """
    Iterate over all ASDBank transcripts and split inter-turn gaps by response type.

    For every child utterance that follows any other utterance, the inter-turn gap
    (curr.timing - prev.end_timing) is collected. Gaps outside [0, 10s) are
    discarded as likely annotation artefacts or session boundaries.

    Backchannel vs substantive classification is based on word count — this is a
    deliberate simplification: the goal is a quick distributional sanity check, not
    a linguistically precise categorisation.
    """
    data_dir = project_root / "data/asdbank_aac"
    parser = CHATParser()
    try:
        transcripts = parser.parse_directory(data_dir, recursive=True)
    except Exception:
        return

    backchannel_pauses = []  # Child turns with < 2 words (minimal response)
    substantive_pauses = []  # Child turns with >= 2 words (full utterance)
    
    for t in transcripts:
        utterances = t.utterances
        if not utterances:
            continue
        
        for i in range(1, len(utterances)):
            curr = utterances[i]
            prev = utterances[i-1]
            
            # Only consider child utterances where both timing markers are available
            if curr.timing is not None and prev.end_timing is not None:
                diff = curr.timing - prev.end_timing

                # Ignore negative gaps (alignment errors) and very long silences
                if 0 <= diff < 10.0:
                    if curr.speaker == 'CHI':
                        n_words = len(curr.text.split())
                        if n_words < 2:
                            backchannel_pauses.append(diff)
                        else:
                            substantive_pauses.append(diff)

    print(f"Backchannels (n={len(backchannel_pauses)}): Mean={np.mean(backchannel_pauses):.2f}s, Median={np.median(backchannel_pauses):.2f}s")
    print(f"Substantive (n={len(substantive_pauses)}): Mean={np.mean(substantive_pauses):.2f}s, Median={np.median(substantive_pauses):.2f}s")

if __name__ == "__main__":
    validate_context()
