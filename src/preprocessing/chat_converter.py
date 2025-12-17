"""
CHAT File Converter Module

This module provides functionality to convert CHAT-formatted transcript files
(.cha) into human-readable formats (text, markdown, HTML, JSON).

Key Features:
- Convert .cha files to plain text
- Convert .cha files to markdown
- Convert .cha files to HTML
- Convert .cha files to JSON
- Batch conversion of multiple files
- Preserve metadata and speaker information

Author: Bimidu Gunathilake
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Literal
from dataclasses import asdict
import html as html_module

from src.parsers.chat_parser import CHATParser, TranscriptData, Utterance
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CHATConverter:
    """
    Convert CHAT transcript files to human-readable formats.

    Supported output formats:
    - txt: Plain text format
    - md: Markdown format
    - html: HTML format
    - json: JSON format
    """

    def __init__(self):
        """Initialize the CHAT converter."""
        self.parser = CHATParser()
        logger.info("CHATConverter initialized")

    def convert_file(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        format: Literal['txt', 'md', 'html', 'json'] = 'txt',
        include_metadata: bool = True,
        include_speaker_labels: bool = True,
        include_timing: bool = False,
    ) -> Path:
        """
        Convert a single CHAT file to the specified format.

        Args:
            input_path: Path to input .cha file
            output_path: Path to output file (auto-generated if None)
            format: Output format ('txt', 'md', 'html', 'json')
            include_metadata: Include file metadata in output
            include_speaker_labels: Include speaker labels in output
            include_timing: Include timing information in output

        Returns:
            Path to the converted file
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        logger.info(f"Converting {input_path.name} to {format.upper()} format")

        # Parse the CHAT file
        transcript = self.parser.parse_file(input_path)

        # Generate output path if not provided
        if output_path is None:
            output_path = input_path.with_suffix(f'.{format}')
        else:
            output_path = Path(output_path)

        # Convert based on format
        if format == 'txt':
            content = self._to_text(
                transcript,
                include_metadata,
                include_speaker_labels,
                include_timing
            )
        elif format == 'md':
            content = self._to_markdown(
                transcript,
                include_metadata,
                include_speaker_labels,
                include_timing
            )
        elif format == 'html':
            content = self._to_html(
                transcript,
                include_metadata,
                include_speaker_labels,
                include_timing
            )
        elif format == 'json':
            content = self._to_json(transcript, include_metadata)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding='utf-8')

        logger.info(f"Converted file saved to: {output_path}")
        return output_path

    def convert_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        format: Literal['txt', 'md', 'html', 'json'] = 'txt',
        recursive: bool = True,
        **kwargs
    ) -> List[Path]:
        """
        Convert all CHAT files in a directory.

        Args:
            input_dir: Directory containing .cha files
            output_dir: Directory for converted files
            format: Output format
            recursive: Search subdirectories recursively
            **kwargs: Additional arguments passed to convert_file()

        Returns:
            List of paths to converted files
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        # Find all .cha files
        pattern = "**/*.cha" if recursive else "*.cha"
        cha_files = list(input_dir.glob(pattern))

        logger.info(f"Found {len(cha_files)} .cha files in {input_dir}")

        converted_files = []
        for cha_file in cha_files:
            try:
                # Preserve directory structure
                relative_path = cha_file.relative_to(input_dir)
                output_path = output_dir / relative_path.with_suffix(f'.{format}')

                converted_path = self.convert_file(
                    cha_file,
                    output_path,
                    format=format,
                    **kwargs
                )
                converted_files.append(converted_path)

            except Exception as e:
                logger.error(f"Error converting {cha_file}: {e}")

        logger.info(f"Converted {len(converted_files)} files to {output_dir}")
        return converted_files

    def _to_text(
        self,
        transcript: TranscriptData,
        include_metadata: bool,
        include_speaker_labels: bool,
        include_timing: bool
    ) -> str:
        """Convert transcript to plain text format."""
        lines = []

        if include_metadata:
            lines.append("=" * 70)
            lines.append("TRANSCRIPT METADATA")
            lines.append("=" * 70)
            lines.append(f"File: {transcript.file_path.name}")
            lines.append(f"Participant ID: {transcript.participant_id}")
            if transcript.diagnosis:
                lines.append(f"Diagnosis: {transcript.diagnosis}")
            if transcript.age_months:
                lines.append(f"Age: {transcript.age_months} months")
            if transcript.gender:
                lines.append(f"Gender: {transcript.gender}")
            if transcript.session_date:
                lines.append(f"Date: {transcript.session_date.strftime('%Y-%m-%d')}")
            lines.append(f"Total Utterances: {transcript.total_utterances}")
            lines.append("=" * 70)
            lines.append("")

        # Add utterances
        for i, utt in enumerate(transcript.utterances, 1):
            parts = []

            if include_timing and utt.timing is not None:
                parts.append(f"[{utt.timing:.2f}s]")

            if include_speaker_labels:
                parts.append(f"{utt.speaker}:")

            parts.append(utt.text)

            lines.append(" ".join(parts))

        return "\n".join(lines)

    def _to_markdown(
        self,
        transcript: TranscriptData,
        include_metadata: bool,
        include_speaker_labels: bool,
        include_timing: bool
    ) -> str:
        """Convert transcript to Markdown format."""
        lines = []

        if include_metadata:
            lines.append(f"# Transcript: {transcript.file_path.name}")
            lines.append("")
            lines.append("## Metadata")
            lines.append("")
            lines.append(f"- **Participant ID**: {transcript.participant_id}")
            if transcript.diagnosis:
                lines.append(f"- **Diagnosis**: {transcript.diagnosis}")
            if transcript.age_months:
                lines.append(f"- **Age**: {transcript.age_months} months")
            if transcript.gender:
                lines.append(f"- **Gender**: {transcript.gender}")
            if transcript.session_date:
                lines.append(f"- **Date**: {transcript.session_date.strftime('%Y-%m-%d')}")
            lines.append(f"- **Total Utterances**: {transcript.total_utterances}")
            lines.append("")
            lines.append("## Transcript")
            lines.append("")

        # Add utterances
        for i, utt in enumerate(transcript.utterances, 1):
            parts = []

            if include_timing and utt.timing is not None:
                parts.append(f"*[{utt.timing:.2f}s]*")

            if include_speaker_labels:
                parts.append(f"**{utt.speaker}:**")

            parts.append(utt.text)

            lines.append(" ".join(parts) + "  ")

        return "\n".join(lines)

    def _to_html(
        self,
        transcript: TranscriptData,
        include_metadata: bool,
        include_speaker_labels: bool,
        include_timing: bool
    ) -> str:
        """Convert transcript to HTML format."""
        lines = []

        # HTML header
        lines.append("<!DOCTYPE html>")
        lines.append("<html lang='en'>")
        lines.append("<head>")
        lines.append("    <meta charset='UTF-8'>")
        lines.append("    <meta name='viewport' content='width=device-width, initial-scale=1.0'>")
        lines.append(f"    <title>Transcript: {html_module.escape(transcript.file_path.name)}</title>")
        lines.append("    <style>")
        lines.append("        * { margin: 0; padding: 0; box-sizing: border-box; }")
        lines.append("        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 40px 20px; background-color: #ffffff; color: #333; line-height: 1.6; }")
        lines.append("        h1 { font-size: 28px; margin-bottom: 30px; color: #1a1a1a; border-bottom: 2px solid #e0e0e0; padding-bottom: 10px; }")
        lines.append("        .metadata { background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px; padding: 20px; margin-bottom: 30px; }")
        lines.append("        .metadata h2 { font-size: 18px; margin-bottom: 15px; color: #495057; }")
        lines.append("        .metadata-grid { display: grid; grid-template-columns: auto 1fr; gap: 8px 20px; }")
        lines.append("        .metadata-label { font-weight: 600; color: #6c757d; }")
        lines.append("        .metadata-value { color: #212529; }")
        lines.append("        .transcript-section { margin-top: 30px; }")
        lines.append("        .transcript-section h2 { font-size: 20px; margin-bottom: 20px; color: #495057; }")
        lines.append("        .utterance { margin-bottom: 12px; padding: 12px 16px; background-color: #ffffff; border-left: 3px solid #0066cc; }")
        lines.append("        .utterance:nth-child(even) { background-color: #f8f9fa; }")
        lines.append("        .speaker { font-weight: 600; color: #0066cc; margin-right: 8px; }")
        lines.append("        .timing { color: #6c757d; font-size: 13px; margin-right: 8px; font-family: 'Courier New', monospace; }")
        lines.append("        .text { color: #212529; }")
        lines.append("        @media print { body { max-width: 100%; padding: 20px; } .utterance { page-break-inside: avoid; } }")
        lines.append("        @media (max-width: 600px) { body { padding: 20px 15px; } .metadata-grid { grid-template-columns: 1fr; gap: 4px; } }")
        lines.append("    </style>")
        lines.append("</head>")
        lines.append("<body>")
        lines.append(f"    <h1>Transcript: {html_module.escape(transcript.file_path.name)}</h1>")

        if include_metadata:
            lines.append("    <div class='metadata'>")
            lines.append("        <h2>Metadata</h2>")
            lines.append("        <div class='metadata-grid'>")
            lines.append(f"            <span class='metadata-label'>File:</span>")
            lines.append(f"            <span class='metadata-value'>{html_module.escape(transcript.file_path.name)}</span>")
            lines.append(f"            <span class='metadata-label'>Participant ID:</span>")
            lines.append(f"            <span class='metadata-value'>{html_module.escape(str(transcript.participant_id))}</span>")
            if transcript.diagnosis:
                lines.append(f"            <span class='metadata-label'>Diagnosis:</span>")
                lines.append(f"            <span class='metadata-value'>{html_module.escape(transcript.diagnosis)}</span>")
            if transcript.age_months:
                lines.append(f"            <span class='metadata-label'>Age:</span>")
                lines.append(f"            <span class='metadata-value'>{transcript.age_months} months</span>")
            if transcript.gender:
                lines.append(f"            <span class='metadata-label'>Gender:</span>")
                lines.append(f"            <span class='metadata-value'>{html_module.escape(transcript.gender)}</span>")
            if transcript.session_date:
                lines.append(f"            <span class='metadata-label'>Date:</span>")
                lines.append(f"            <span class='metadata-value'>{transcript.session_date.strftime('%Y-%m-%d')}</span>")
            lines.append(f"            <span class='metadata-label'>Total Utterances:</span>")
            lines.append(f"            <span class='metadata-value'>{transcript.total_utterances}</span>")
            lines.append("        </div>")
            lines.append("    </div>")

        lines.append("    <div class='transcript-section'>")
        lines.append("        <h2>Conversation</h2>")

        # Add utterances
        for utt in transcript.utterances:
            lines.append("        <div class='utterance'>")

            parts = []
            if include_timing and utt.timing is not None:
                parts.append(f"<span class='timing'>[{utt.timing:.2f}s]</span>")

            if include_speaker_labels:
                parts.append(f"<span class='speaker'>{html_module.escape(utt.speaker)}</span>")

            parts.append(f"<span class='text'>{html_module.escape(utt.text)}</span>")

            lines.append("            " + "".join(parts))
            lines.append("        </div>")

        lines.append("    </div>")
        lines.append("</body>")
        lines.append("</html>")

        return "\n".join(lines)

    def _to_json(
        self,
        transcript: TranscriptData,
        include_metadata: bool
    ) -> str:
        """Convert transcript to JSON format."""
        data = {}

        if include_metadata:
            data['metadata'] = {
                'file_path': str(transcript.file_path),
                'participant_id': transcript.participant_id,
                'diagnosis': transcript.diagnosis,
                'age_months': transcript.age_months,
                'gender': transcript.gender,
                'session_date': transcript.session_date.isoformat() if transcript.session_date else None,
                'total_utterances': transcript.total_utterances,
            }

        data['utterances'] = []
        for utt in transcript.utterances:
            utt_dict = {
                'speaker': utt.speaker,
                'text': utt.text,
                'word_count': utt.word_count,
            }

            if utt.timing is not None:
                utt_dict['timing'] = utt.timing

            if utt.morphology:
                utt_dict['morphology'] = utt.morphology

            if utt.grammar:
                utt_dict['grammar'] = utt.grammar

            data['utterances'].append(utt_dict)

        return json.dumps(data, indent=2, ensure_ascii=False)


def convert_chat_files(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    format: Literal['txt', 'md', 'html', 'json'] = 'txt',
    **kwargs
) -> Union[Path, List[Path]]:
    """
    Convenience function to convert CHAT files.

    Args:
        input_path: Path to .cha file or directory
        output_path: Path to output file or directory
        format: Output format
        **kwargs: Additional arguments

    Returns:
        Path or list of paths to converted files
    """
    converter = CHATConverter()
    input_path = Path(input_path)

    if input_path.is_file():
        return converter.convert_file(input_path, output_path, format, **kwargs)
    elif input_path.is_dir():
        return converter.convert_directory(input_path, output_path, format, **kwargs)
    else:
        raise ValueError(f"Invalid input path: {input_path}")
