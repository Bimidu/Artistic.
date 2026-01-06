# CHAT Converter - Usage Examples

This document provides practical examples of using the CHAT file converter.

## Table of Contents
- [Quick Start](#quick-start)
- [Single File Conversion](#single-file-conversion)
- [Batch Conversion](#batch-conversion)
- [Programmatic Usage](#programmatic-usage)
- [Common Use Cases](#common-use-cases)

## Quick Start

### Convert One File (Simplest)

```bash
# This creates "1082.txt" in the same directory as the .cha file
python scripts/convert_chat_files.py data/asdbank_eigsti/Eigsti/ASD/1082.cha
```

### Convert to Markdown with Custom Output

```bash
python scripts/convert_chat_files.py \
    data/asdbank_eigsti/Eigsti/ASD/1082.cha \
    -o my_transcript.md \
    -f md
```

## Single File Conversion

### Text Format (Default)

```bash
# Basic text conversion
python scripts/convert_chat_files.py data/file.cha

# Text with timing information
python scripts/convert_chat_files.py data/file.cha --include-timing

# Text without metadata
python scripts/convert_chat_files.py data/file.cha --no-metadata

# Just the conversation (no metadata, no speaker labels)
python scripts/convert_chat_files.py data/file.cha --no-metadata --no-speaker-labels
```

### Markdown Format

```bash
# Basic markdown
python scripts/convert_chat_files.py data/file.cha -f md

# Markdown with timing
python scripts/convert_chat_files.py data/file.cha -f md --include-timing

# Markdown to specific file
python scripts/convert_chat_files.py data/file.cha -o output/transcript.md -f md
```

### HTML Format

```bash
# Create an HTML page
python scripts/convert_chat_files.py data/file.cha -f html

# HTML with all options
python scripts/convert_chat_files.py data/file.cha \
    -f html \
    -o transcript.html \
    --include-timing
```

### JSON Format

```bash
# Convert to JSON
python scripts/convert_chat_files.py data/file.cha -f json

# JSON with full metadata
python scripts/convert_chat_files.py data/file.cha \
    -o data.json \
    -f json
```

## Batch Conversion

### Convert Entire Dataset

```bash
# Convert all files in eigsti dataset to markdown
python scripts/convert_chat_files.py \
    data/asdbank_eigsti \
    -o converted_eigsti \
    -f md

# Convert with timing information
python scripts/convert_chat_files.py \
    data/asdbank_eigsti \
    -o converted_eigsti_with_timing \
    -f md \
    --include-timing
```

### Convert All Datasets

```bash
# Convert all datasets to text
bash scripts/convert_all_datasets.sh txt

# Convert all datasets to markdown
bash scripts/convert_all_datasets.sh md

# Convert all datasets to HTML
bash scripts/convert_all_datasets.sh html

# Output will be in: data_converted_[format]/
```

### Convert Only Top-Level Files (No Subdirectories)

```bash
python scripts/convert_chat_files.py \
    data/asdbank_eigsti \
    -o output \
    -f md \
    --no-recursive
```

## Programmatic Usage

### Basic Python Usage

```python
from src.preprocessing.chat_converter import CHATConverter

# Initialize converter
converter = CHATConverter()

# Convert single file
output_path = converter.convert_file(
    'data/asdbank_eigsti/Eigsti/ASD/1082.cha',
    'output.txt'
)
print(f"Saved to: {output_path}")
```

### Advanced Options

```python
from src.preprocessing.chat_converter import CHATConverter

converter = CHATConverter()

# Convert with custom options
output_path = converter.convert_file(
    input_path='data/file.cha',
    output_path='output.md',
    format='md',
    include_metadata=True,
    include_speaker_labels=True,
    include_timing=True
)
```

### Batch Processing in Python

```python
from src.preprocessing.chat_converter import CHATConverter
from pathlib import Path

converter = CHATConverter()

# Convert all files in directory
converted_files = converter.convert_directory(
    input_dir='data/asdbank_eigsti',
    output_dir='converted_output',
    format='html',
    recursive=True,
    include_timing=True
)

print(f"Converted {len(converted_files)} files")
for file in converted_files:
    print(f"  - {file}")
```

### Process Specific Groups

```python
from src.preprocessing.chat_converter import CHATConverter
from pathlib import Path

converter = CHATConverter()

# Convert only ASD files
asd_dir = Path('data/asdbank_eigsti/Eigsti/ASD')
asd_files = converter.convert_directory(
    asd_dir,
    'converted_asd',
    format='md'
)

# Convert only TD files
td_dir = Path('data/asdbank_eigsti/Eigsti/TD')
td_files = converter.convert_directory(
    td_dir,
    'converted_td',
    format='md'
)

print(f"ASD files: {len(asd_files)}")
print(f"TD files: {len(td_files)}")
```

### Using Convenience Function

```python
from src.preprocessing.chat_converter import convert_chat_files

# Single file
convert_chat_files('data/file.cha', 'output.txt', format='txt')

# Directory
convert_chat_files(
    'data/asdbank_eigsti',
    'output_md',
    format='md',
    include_timing=True
)
```

## Common Use Cases

### 1. Quick Preview of Dataset

Convert a few sample files to quickly understand the dataset:

```bash
# Convert first ASD file
python scripts/convert_chat_files.py \
    data/asdbank_eigsti/Eigsti/ASD/1082.cha \
    -o preview_asd.md -f md

# Convert first TD file
python scripts/convert_chat_files.py \
    data/asdbank_eigsti/Eigsti/TD/1068.cha \
    -o preview_td.md -f md
```

### 2. Create Documentation

Generate HTML versions for documentation or sharing:

```bash
# Convert dataset to HTML
python scripts/convert_chat_files.py \
    data/asdbank_eigsti \
    -o docs/transcripts \
    -f html
```

### 3. Extract Structured Data

Convert to JSON for data analysis:

```bash
# Convert to JSON
python scripts/convert_chat_files.py \
    data/asdbank_eigsti \
    -o data_json \
    -f json
```

Then process in Python:

```python
import json
from pathlib import Path

# Load and analyze JSON transcripts
json_files = Path('data_json').glob('**/*.json')

for json_file in json_files:
    with open(json_file) as f:
        data = json.load(f)

    # Access metadata
    diagnosis = data['metadata']['diagnosis']
    age = data['metadata']['age_months']

    # Access utterances
    total_utterances = len(data['utterances'])
    child_utterances = [u for u in data['utterances'] if u['speaker'] == 'CHI']

    print(f"{json_file.name}: {diagnosis}, Age {age}, "
          f"{len(child_utterances)}/{total_utterances} child utterances")
```

### 4. Compare ASD vs TD Transcripts

```python
from src.preprocessing.chat_converter import CHATConverter

converter = CHATConverter()

# Convert both groups
asd_files = converter.convert_directory(
    'data/asdbank_eigsti/Eigsti/ASD',
    'comparison/asd',
    format='md'
)

td_files = converter.convert_directory(
    'data/asdbank_eigsti/Eigsti/TD',
    'comparison/td',
    format='md'
)

print(f"Created {len(asd_files)} ASD transcripts")
print(f"Created {len(td_files)} TD transcripts")
print("Review files in comparison/ directory")
```

### 5. Annotate Transcripts

Convert to markdown for easy annotation:

```bash
# Convert to markdown
python scripts/convert_chat_files.py \
    data/asdbank_eigsti \
    -o transcripts_for_annotation \
    -f md

# Now you can open the .md files in any text editor
# and add annotations using markdown syntax
```

### 6. Extract Timing Information

```bash
# Get transcripts with timestamps
python scripts/convert_chat_files.py \
    data/asdbank_eigsti \
    -o transcripts_timed \
    -f txt \
    --include-timing
```

## Tips and Best Practices

1. **Start Small**: Test with one file before batch converting
   ```bash
   python scripts/convert_chat_files.py data/single_file.cha -f md
   ```

2. **Check Output**: Always verify the first converted file
   ```bash
   cat output.txt  # or open in your editor
   ```

3. **Use Markdown for GitHub**: Markdown files render nicely on GitHub
   ```bash
   python scripts/convert_chat_files.py data/ -o docs/ -f md
   git add docs/
   git commit -m "Add readable transcripts"
   ```

4. **Organize Output**: Create organized output directories
   ```bash
   mkdir -p converted/{asd,td}
   python scripts/convert_chat_files.py data/ASD -o converted/asd -f md
   python scripts/convert_chat_files.py data/TD -o converted/td -f md
   ```

5. **Preserve Structure**: Output maintains directory structure
   ```bash
   # Input:  data/dataset1/group1/file.cha
   # Output: output/dataset1/group1/file.md
   ```

## Sample Output Locations

After running the converter, you'll find files in these locations:

```
project/
├── data/                           # Original .cha files
│   └── asdbank_eigsti/
│       └── Eigsti/
│           ├── ASD/
│           │   └── 1082.cha
│           └── TD/
│               └── 1068.cha
├── samples_converted/              # Sample conversions
│   ├── asd_sample.md
│   └── td_sample.md
├── data_converted_md/              # Full batch conversion
│   └── asdbank_eigsti/
│       └── Eigsti/
│           ├── ASD/
│           │   └── 1082.md
│           └── TD/
│               └── 1068.md
└── scripts/
    ├── convert_chat_files.py       # Main converter script
    └── convert_all_datasets.sh     # Batch conversion script
```

## Troubleshooting

### Problem: "File not found"
```bash
# Check file exists
ls data/asdbank_eigsti/Eigsti/ASD/1082.cha

# Use absolute path if needed
python scripts/convert_chat_files.py $(pwd)/data/file.cha
```

### Problem: "No files converted"
```bash
# Check for .cha files in directory
find data/asdbank_eigsti -name "*.cha" | head

# Try with absolute path
python scripts/convert_chat_files.py $(pwd)/data/asdbank_eigsti
```

### Problem: "Permission denied"
```bash
# Make scripts executable
chmod +x scripts/convert_chat_files.py
chmod +x scripts/convert_all_datasets.sh
```

## See Also

- [CHAT_CONVERTER_README.md](CHAT_CONVERTER_README.md) - Full documentation
- [src/preprocessing/chat_converter.py](src/preprocessing/chat_converter.py) - Source code
- [src/parsers/chat_parser.py](src/parsers/chat_parser.py) - CHAT parser
