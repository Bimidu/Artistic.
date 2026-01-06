#!/usr/bin/env python3
"""
Command-line script to convert CHAT (.cha) files to human-readable formats.

Usage:
    # Convert single file to text
    python scripts/convert_chat_files.py data/file.cha -o output.txt

    # Convert single file to markdown
    python scripts/convert_chat_files.py data/file.cha -f md

    # Convert entire directory to HTML
    python scripts/convert_chat_files.py data/asdbank_eigsti -o output_html -f html

    # Convert with options
    python scripts/convert_chat_files.py data/file.cha -f md --no-metadata --include-timing

Author: Bimidu Gunathilake
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.chat_converter import CHATConverter


def main():
    parser = argparse.ArgumentParser(
        description='Convert CHAT (.cha) files to human-readable formats',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file to text
  python scripts/convert_chat_files.py data/sample.cha

  # Convert to markdown
  python scripts/convert_chat_files.py data/sample.cha -f md -o output.md

  # Convert entire directory to HTML
  python scripts/convert_chat_files.py data/asdbank_eigsti -o output_html -f html

  # Convert with timing information
  python scripts/convert_chat_files.py data/sample.cha -f md --include-timing

  # Convert without metadata
  python scripts/convert_chat_files.py data/sample.cha --no-metadata
        """
    )

    parser.add_argument(
        'input',
        type=str,
        help='Input .cha file or directory containing .cha files'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output file or directory (auto-generated if not specified)'
    )

    parser.add_argument(
        '-f', '--format',
        type=str,
        choices=['txt', 'md', 'html', 'json'],
        default='txt',
        help='Output format (default: txt)'
    )

    parser.add_argument(
        '--no-metadata',
        action='store_true',
        help='Exclude metadata from output'
    )

    parser.add_argument(
        '--no-speaker-labels',
        action='store_true',
        help='Exclude speaker labels from output'
    )

    parser.add_argument(
        '--include-timing',
        action='store_true',
        help='Include timing information in output'
    )

    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='Do not search subdirectories (only for directory input)'
    )

    args = parser.parse_args()

    # Validate input path
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Initialize converter
    converter = CHATConverter()

    # Prepare conversion options
    options = {
        'include_metadata': not args.no_metadata,
        'include_speaker_labels': not args.no_speaker_labels,
        'include_timing': args.include_timing,
    }

    try:
        if input_path.is_file():
            # Convert single file
            if not input_path.suffix == '.cha':
                print(f"Error: Input file must have .cha extension", file=sys.stderr)
                sys.exit(1)

            output_path = converter.convert_file(
                input_path,
                args.output,
                format=args.format,
                **options
            )
            print(f"✓ Converted: {input_path.name}")
            print(f"  Output: {output_path}")

        elif input_path.is_dir():
            # Convert directory
            if args.output is None:
                output_dir = input_path.parent / f"{input_path.name}_converted_{args.format}"
            else:
                output_dir = Path(args.output)

            options['recursive'] = not args.no_recursive

            converted_files = converter.convert_directory(
                input_path,
                output_dir,
                format=args.format,
                **options
            )

            print(f"✓ Converted {len(converted_files)} files")
            print(f"  Output directory: {output_dir}")

        else:
            print(f"Error: Invalid input path: {input_path}", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"Error during conversion: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
