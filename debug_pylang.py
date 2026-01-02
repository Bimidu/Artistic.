
import pylangacq
from pathlib import Path

path = "data/asdbank_aac/AAC/01_T1_1.cha"
try:
    reader = pylangacq.read_chat(path)
    u = reader.utterances()[0]
    print(f"Type: {type(u)}")
    print(f"Segments: {dir(u)}")
    if hasattr(u, "time_marks"):
        print(f"Time marks: {u.time_marks}")
    else:
        print("No time_marks attribute")
        
    # Check tiers 
    print(f"Tiers: {u.tiers}")
except Exception as e:
    print(e)
