# CHAT File Converter

This tool converts CHAT-formatted transcript files (.cha) from the TalkBank/CHILDES database into human-readable formats.

## Features

- **Multiple output formats**: Text, Markdown, HTML, JSON
- **Metadata extraction**: Participant ID, diagnosis, age, gender, session date
- **Speaker identification**: Clear labeling of who said what
- **Timing information**: Optional timestamp display
- **Batch conversion**: Convert entire directories at once
- **Flexible options**: Customize what information to include

## Installation

No additional installation required if you have the main project set up. The converter uses the existing CHAT parser.

## Quick Start

### Convert a Single File

```bash
# Convert to text (default)
python scripts/convert_chat_files.py data/asdbank_eigsti/Eigsti/ASD/1082.cha

# Convert to markdown
python scripts/convert_chat_files.py data/asdbank_eigsti/Eigsti/ASD/1082.cha -f md

# Convert to HTML
python scripts/convert_chat_files.py data/asdbank_eigsti/Eigsti/ASD/1082.cha -f html

# Convert to JSON
python scripts/convert_chat_files.py data/asdbank_eigsti/Eigsti/ASD/1082.cha -f json
```

### Convert an Entire Directory

```bash
# Convert all files in a directory to markdown
python scripts/convert_chat_files.py data/asdbank_eigsti -o output_md -f md

# Convert with timing information
python scripts/convert_chat_files.py data/asdbank_eigsti -o output_md -f md --include-timing

# Convert without metadata
python scripts/convert_chat_files.py data/asdbank_eigsti -o output_txt --no-metadata
```

### Convert All Datasets

```bash
# Convert all datasets to text format
bash scripts/convert_all_datasets.sh txt

# Convert all datasets to markdown
bash scripts/convert_all_datasets.sh md

# Convert all datasets to HTML
bash scripts/convert_all_datasets.sh html
```

## Command-Line Options

```
usage: convert_chat_files.py [-h] [-o OUTPUT] [-f {txt,md,html,json}]
                             [--no-metadata] [--no-speaker-labels]
                             [--include-timing] [--no-recursive]
                             input

positional arguments:
  input                 Input .cha file or directory containing .cha files

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Output file or directory (auto-generated if not specified)
  -f {txt,md,html,json}, --format {txt,md,html,json}
                        Output format (default: txt)
  --no-metadata         Exclude metadata from output
  --no-speaker-labels   Exclude speaker labels from output
  --include-timing      Include timing information in output
  --no-recursive        Do not search subdirectories (only for directory input)
```

## Output Format Examples

### Text Format (.txt)

```
======================================================================
TRANSCRIPT METADATA
======================================================================
File: 1082.cha
Participant ID: a-00032788-1
Diagnosis: ASD
Age: 60 months
Gender: male
Total Utterances: 350
======================================================================

INV1: here , set it on the floor so we can look inside of it .
INV1: what do you see ?
CHI: nothing .
INV1: nothing , well maybe we gotta do something with it then .
...
```

### Markdown Format (.md)

```markdown
# Transcript: 1082.cha

## Metadata

- **Participant ID**: a-00032788-1
- **Diagnosis**: ASD
- **Age**: 60 months
- **Gender**: male
- **Total Utterances**: 350

## Transcript

**INV1:** here , set it on the floor so we can look inside of it .
**INV1:** what do you see ?
**CHI:** nothing .
**INV1:** nothing , well maybe we gotta do something with it then .
...
```

### HTML Format (.html)

Generates a styled HTML page with:
- Formatted metadata section
- Speaker labels in blue
- Utterances with left border
- Responsive design
- Optional timing information in gray

### JSON Format (.json)

```json
{
  "metadata": {
    "file_path": "data/asdbank_eigsti/Eigsti/ASD/1082.cha",
    "participant_id": "a-00032788-1",
    "diagnosis": "ASD",
    "age_months": 60,
    "gender": "male",
    "total_utterances": 350
  },
  "utterances": [
    {
      "speaker": "INV1",
      "text": "here , set it on the floor...",
      "word_count": 15
    },
    ...
  ]
}
```

## Use Cases

### Research Analysis
Convert transcripts to markdown or HTML for easy reading and annotation during analysis.

### Data Exploration
Convert a few sample files to text to quickly understand the content of a dataset.

### Documentation
Generate HTML versions of transcripts for sharing with collaborators or including in reports.

### Data Processing
Convert to JSON for custom data processing pipelines or integration with other tools.

### Quality Checking
Quickly review converted files to verify parser accuracy and data quality.

## Programmatic Usage

You can also use the converter in Python scripts:

```python
from src.preprocessing.chat_converter import CHATConverter

# Initialize converter
converter = CHATConverter()

# Convert single file
output_path = converter.convert_file(
    'data/sample.cha',
    'output.md',
    format='md',
    include_metadata=True,
    include_timing=True
)

# Convert directory
converted_files = converter.convert_directory(
    'data/asdbank_eigsti',
    'output_html',
    format='html',
    recursive=True
)

print(f"Converted {len(converted_files)} files")
```

## Speaker Codes

Common speaker codes in CHAT files:
- **CHI**: Target child (the main participant)
- **INV**: Investigator/Interviewer
- **INV1, INV2**: Multiple investigators
- **MOT**: Mother
- **FAT**: Father

## Tips

1. **Start small**: Convert a single file first to verify the output meets your needs
2. **Use markdown for documentation**: Markdown is great for viewing on GitHub or in text editors
3. **Use HTML for presentations**: HTML provides the nicest formatting for sharing
4. **Use JSON for data processing**: JSON is ideal for programmatic access
5. **Include timing for analysis**: Timing information can be useful for conversation flow analysis

## Troubleshooting

### Error: "No files in reader"
Make sure you're pointing to a valid .cha file with correct format.

### Missing metadata
Some older CHAT files may not have all metadata fields. The converter handles missing fields gracefully.

### Large directories taking too long
Use `--no-recursive` to avoid searching subdirectories, or convert specific subdirectories individually.

## Performance

- Single file conversion: ~1-3 seconds per file
- Batch conversion: Processes all files sequentially
- Large datasets (500+ files): May take several minutes

## Related Files

- `src/preprocessing/chat_converter.py` - Main converter module
- `scripts/convert_chat_files.py` - Command-line script
- `scripts/convert_all_datasets.sh` - Batch conversion script
- `src/parsers/chat_parser.py` - CHAT file parser
