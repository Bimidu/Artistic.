
import pylangacq
from pathlib import Path

# Check a file we know exists and probably has diagnosis
path = "data/asdbank_aac/AAC/01_T1_1.cha"
try:
    reader = pylangacq.read_chat(path)
    participants = reader.participants()
    print(f"Participants raw: {participants}")
    print(f"Headers raw: {reader.headers()}")
    
    # Check what my parser does
    import sys
    sys.path.append(".")
    from src.parsers.chat_parser import CHATParser
    p = CHATParser()
    t = p.parse_file(path)
    print(f"Parsed Diagnosis: {t.diagnosis}")
    print(f"Parsed Metadata: {t.metadata}")
    print(f"Parsed Speakers: {t.speakers}")

except Exception as e:
    print(e)
