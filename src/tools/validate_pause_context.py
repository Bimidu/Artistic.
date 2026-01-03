
import sys
import numpy as np
from pathlib import Path
import pylangacq

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.parsers.chat_parser import CHATParser

def validate_context():
    data_dir = project_root / "data/asdbank_aac"
    parser = CHATParser()
    try:
        transcripts = parser.parse_directory(data_dir, recursive=True)
    except Exception:
        return

    backchannel_pauses = [] # < 2 words
    substantive_pauses = [] # >= 2 words
    
    for t in transcripts:
        utterances = t.utterances
        if not utterances: continue
        
        for i in range(1, len(utterances)):
            curr = utterances[i]
            prev = utterances[i-1]
            
            if curr.timing is not None and prev.end_timing is not None:
                diff = curr.timing - prev.end_timing
                if 0 <= diff < 10.0:
                    if curr.speaker == 'CHI':
                        # Count words roughly
                        n_words = len(curr.text.split())
                        if n_words < 2:
                            backchannel_pauses.append(diff)
                        else:
                            substantive_pauses.append(diff)

    print(f"Backchannels (n={len(backchannel_pauses)}): Mean={np.mean(backchannel_pauses):.2f}s, Median={np.median(backchannel_pauses):.2f}s")
    print(f"Substantive (n={len(substantive_pauses)}): Mean={np.mean(substantive_pauses):.2f}s, Median={np.median(substantive_pauses):.2f}s")

if __name__ == "__main__":
    validate_context()
