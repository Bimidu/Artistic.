#!/bin/bash
# Convert all CHAT datasets to human-readable formats
# Usage: bash scripts/convert_all_datasets.sh [format]
# Format options: txt, md, html, json (default: txt)

FORMAT=${1:-txt}

echo "========================================================================"
echo "Converting all CHAT datasets to ${FORMAT} format"
echo "========================================================================"
echo ""

# Create output directory
OUTPUT_DIR="data_converted_${FORMAT}"
mkdir -p "$OUTPUT_DIR"

# Convert each dataset
for dataset_dir in data/asdbank_*; do
    if [ -d "$dataset_dir" ]; then
        dataset_name=$(basename "$dataset_dir")
        echo "Converting: $dataset_name"

        python3 scripts/convert_chat_files.py \
            "$dataset_dir" \
            -o "$OUTPUT_DIR/$dataset_name" \
            -f "$FORMAT" \
            --include-timing

        echo ""
    fi
done

echo "========================================================================"
echo "✓ Conversion complete!"
echo "  Output directory: $OUTPUT_DIR"
echo "========================================================================"
