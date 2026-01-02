
import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))
from src.parsers.chat_parser import CHATParser

if __name__ == "__main__":
    p = CHATParser()
    # Parse a subfolder to be faster
    # asdbank_aac/AAC looks like it has diagnoses based on filename structure?
    # Let's try to parse just 5 files and print everything
    ts = p.parse_directory("data/asdbank_aac/AAC", recursive=False)
    for i, t in enumerate(ts):
        if i > 5: break
        print(f"File: {t.file_path.name}")
        print(f"Diagnosis: {t.diagnosis}")
        print(f"Speakers CHI: {t.speakers.get('CHI', {})}")
        print("---")
