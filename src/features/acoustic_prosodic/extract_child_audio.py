import re
from pathlib import Path
from pydub import AudioSegment

base_path = Path("data/asdbank_aac/AAC")
output_dir = base_path / "child_only"
output_dir.mkdir(exist_ok=True)

def extract_child_audio(cha_path, wav_path, out_path):
    """Extract *CHI: speech segments from .wav using timestamps in .cha"""
    text = cha_path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"\*CHI:.*?(\d+)_(\d+)", text)
    if not matches:
        print(f"No *CHI: timestamps in {cha_path.name}")
        return

    audio = AudioSegment.from_wav(wav_path)
    child_audio = AudioSegment.empty()
    total_dur = 0

    for start, end in matches:
        start, end = int(start), int(end)
        segment = audio[start:end]
        child_audio += segment
        total_dur += (end - start)

    if len(child_audio) == 0:
        print(f"No audio extracted for {cha_path.name}")
        return

    child_audio.export(out_path, format="wav")
    print(f" {cha_path.stem} → {out_path.name} ({total_dur/1000:.2f}s total child speech)")

# Loop through all files
for cha_file in base_path.glob("*.cha"):
    wav_file = cha_file.with_suffix(".wav")
    if wav_file.exists():
        out_file = output_dir / f"{cha_file.stem}_child.wav"
        extract_child_audio(cha_file, wav_file, out_file)
    else:
        print(f"Skipped {cha_file.name} (no .wav found)")
