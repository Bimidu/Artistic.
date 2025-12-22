"""
Annotated Transcript Generator Module

This module generates annotated transcripts that visually mark where
features were extracted from the text. This provides:
1. Transparency in feature extraction
2. Debugging capability for algorithm verification
3. User-facing explanations of predictions

Features are annotated with:
- Color codes for different feature types
- Markers/symbols indicating specific features
- Span highlighting for regions of interest

Author: Bimidu Gunathilake
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import html

from src.utils.logger import get_logger
from src.parsers.chat_parser import TranscriptData, Utterance

logger = get_logger(__name__)


class AnnotationType(Enum):
    """Types of feature annotations."""
    # Turn-taking related
    TURN_START = "turn_start"
    TURN_END = "turn_end"
    OVERLAP = "overlap"
    INTERRUPTION = "interruption"
    LONG_PAUSE = "long_pause"
    RESPONSE_LATENCY = "response_latency"
    
    # Pragmatic markers
    ECHOLALIA = "echolalia"
    PRONOUN_REVERSAL = "pronoun_reversal"
    STEREOTYPED_PHRASE = "stereotyped_phrase"
    SOCIAL_GREETING = "social_greeting"
    QUESTION = "question"
    
    # Conversational features
    TOPIC_SHIFT = "topic_shift"
    TOPIC_MAINTENANCE = "topic_maintenance"
    REPAIR_INITIATION = "repair_initiation"
    REPAIR_COMPLETION = "repair_completion"
    CLARIFICATION_REQUEST = "clarification_request"
    
    # Linguistic features
    COMPLEX_SENTENCE = "complex_sentence"
    SIMPLE_SENTENCE = "simple_sentence"
    FILLED_PAUSE = "filled_pause"
    DISCOURSE_MARKER = "discourse_marker"
    
    # General
    FEATURE_REGION = "feature_region"


@dataclass
class FeatureAnnotation:
    """
    A single feature annotation on the transcript.
    
    Attributes:
        annotation_type: Type of annotation
        start_pos: Start position in text (character index)
        end_pos: End position in text (character index)
        utterance_idx: Index of the utterance containing this annotation
        feature_name: Name of the specific feature
        feature_value: Value of the feature
        confidence: Confidence score (0-1)
        description: Human-readable description
        metadata: Additional metadata
    """
    annotation_type: AnnotationType
    start_pos: int
    end_pos: int
    utterance_idx: int
    feature_name: str
    feature_value: Any = None
    confidence: float = 1.0
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def span_text(self) -> str:
        """Get the text span (requires original text to be passed)."""
        return self.metadata.get('span_text', '')
    
    @property
    def color_code(self) -> str:
        """Get the color code for this annotation type."""
        return ANNOTATION_COLORS.get(self.annotation_type, "#808080")


# Color mapping for different annotation types
ANNOTATION_COLORS = {
    # Turn-taking (blues)
    AnnotationType.TURN_START: "#2196F3",
    AnnotationType.TURN_END: "#1976D2",
    AnnotationType.OVERLAP: "#03A9F4",
    AnnotationType.INTERRUPTION: "#00BCD4",
    AnnotationType.LONG_PAUSE: "#0097A7",
    AnnotationType.RESPONSE_LATENCY: "#00838F",
    
    # Pragmatic markers (reds/oranges)
    AnnotationType.ECHOLALIA: "#F44336",
    AnnotationType.PRONOUN_REVERSAL: "#E91E63",
    AnnotationType.STEREOTYPED_PHRASE: "#FF5722",
    AnnotationType.SOCIAL_GREETING: "#FF9800",
    AnnotationType.QUESTION: "#FFC107",
    
    # Conversational (greens)
    AnnotationType.TOPIC_SHIFT: "#4CAF50",
    AnnotationType.TOPIC_MAINTENANCE: "#8BC34A",
    AnnotationType.REPAIR_INITIATION: "#CDDC39",
    AnnotationType.REPAIR_COMPLETION: "#009688",
    AnnotationType.CLARIFICATION_REQUEST: "#00BFA5",
    
    # Linguistic (purples)
    AnnotationType.COMPLEX_SENTENCE: "#9C27B0",
    AnnotationType.SIMPLE_SENTENCE: "#E1BEE7",
    AnnotationType.FILLED_PAUSE: "#7B1FA2",
    AnnotationType.DISCOURSE_MARKER: "#AB47BC",
    
    # General
    AnnotationType.FEATURE_REGION: "#607D8B",
}

# Symbol mapping for text-based display
ANNOTATION_SYMBOLS = {
    AnnotationType.TURN_START: "[>",
    AnnotationType.TURN_END: "<]",
    AnnotationType.OVERLAP: "[OVR]",
    AnnotationType.INTERRUPTION: "[INT]",
    AnnotationType.LONG_PAUSE: "[...]",
    AnnotationType.RESPONSE_LATENCY: "[LAT]",
    
    AnnotationType.ECHOLALIA: "[ECH]",
    AnnotationType.PRONOUN_REVERSAL: "[PR]",
    AnnotationType.STEREOTYPED_PHRASE: "[STP]",
    AnnotationType.SOCIAL_GREETING: "[SOC]",
    AnnotationType.QUESTION: "[Q]",
    
    AnnotationType.TOPIC_SHIFT: "[TS]",
    AnnotationType.TOPIC_MAINTENANCE: "[TM]",
    AnnotationType.REPAIR_INITIATION: "[RI]",
    AnnotationType.REPAIR_COMPLETION: "[RC]",
    AnnotationType.CLARIFICATION_REQUEST: "[CR]",
    
    AnnotationType.COMPLEX_SENTENCE: "[CX]",
    AnnotationType.SIMPLE_SENTENCE: "[SP]",
    AnnotationType.FILLED_PAUSE: "[FP]",
    AnnotationType.DISCOURSE_MARKER: "[DM]",
    
    AnnotationType.FEATURE_REGION: "[*]",
}


@dataclass
class AnnotatedTranscript:
    """
    A transcript with feature annotations.
    
    Provides multiple output formats:
    - Plain text with markers
    - HTML with color-coded highlighting
    - JSON for programmatic access
    
    Attributes:
        transcript: Original transcript data
        annotations: List of feature annotations
        component: Which component produced these annotations
        metadata: Additional metadata
    """
    transcript: TranscriptData
    annotations: List[FeatureAnnotation] = field(default_factory=list)
    component: str = "pragmatic_conversational"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_annotation(self, annotation: FeatureAnnotation):
        """Add an annotation to the transcript."""
        self.annotations.append(annotation)
    
    def add_annotations(self, annotations: List[FeatureAnnotation]):
        """Add multiple annotations."""
        self.annotations.extend(annotations)
    
    def get_annotations_by_type(
        self,
        annotation_type: AnnotationType
    ) -> List[FeatureAnnotation]:
        """Get all annotations of a specific type."""
        return [a for a in self.annotations if a.annotation_type == annotation_type]
    
    def get_annotations_for_utterance(
        self,
        utterance_idx: int
    ) -> List[FeatureAnnotation]:
        """Get all annotations for a specific utterance."""
        return [a for a in self.annotations if a.utterance_idx == utterance_idx]
    
    def to_plain_text(self, include_legend: bool = True) -> str:
        """
        Generate plain text with symbol markers.
        
        Returns:
            Annotated transcript as plain text
        """
        lines = []
        
        if include_legend:
            lines.append("=" * 60)
            lines.append("ANNOTATED TRANSCRIPT")
            lines.append(f"Component: {self.component}")
            lines.append(f"Total Annotations: {len(self.annotations)}")
            lines.append("=" * 60)
            lines.append("")
            lines.append("LEGEND:")
            for ann_type in set(a.annotation_type for a in self.annotations):
                symbol = ANNOTATION_SYMBOLS.get(ann_type, "[*]")
                lines.append(f"  {symbol} = {ann_type.value}")
            lines.append("")
            lines.append("-" * 60)
            lines.append("")
        
        # Process each utterance
        for idx, utterance in enumerate(self.transcript.utterances):
            utterance_annotations = self.get_annotations_for_utterance(idx)
            
            # Build annotated text
            annotated_text = self._annotate_text_with_symbols(
                utterance.text,
                utterance_annotations
            )
            
            # Format utterance line
            speaker = utterance.speaker
            lines.append(f"*{speaker}: {annotated_text}")
            
            # Add annotation details below if any
            if utterance_annotations:
                for ann in utterance_annotations:
                    symbol = ANNOTATION_SYMBOLS.get(ann.annotation_type, "[*]")
                    lines.append(f"    {symbol} {ann.feature_name}: {ann.description}")
        
        return "\n".join(lines)
    
    def to_html(
        self,
        include_legend: bool = True,
        include_styles: bool = True
    ) -> str:
        """
        Generate HTML with color-coded highlighting.
        
        Returns:
            Annotated transcript as HTML
        """
        html_parts = []
        
        if include_styles:
            html_parts.append(self._get_html_styles())
        
        html_parts.append('<div class="annotated-transcript">')
        
        if include_legend:
            html_parts.append(self._generate_html_legend())
        
        html_parts.append('<div class="transcript-content">')
        
        # Process each utterance
        for idx, utterance in enumerate(self.transcript.utterances):
            utterance_annotations = self.get_annotations_for_utterance(idx)
            
            # Build annotated HTML
            annotated_html = self._annotate_text_with_html(
                utterance.text,
                utterance_annotations
            )
            
            speaker = html.escape(utterance.speaker)
            html_parts.append(
                f'<div class="utterance">'
                f'<span class="speaker">*{speaker}:</span> '
                f'<span class="text">{annotated_html}</span>'
                f'</div>'
            )
        
        html_parts.append('</div>')  # transcript-content
        html_parts.append('</div>')  # annotated-transcript
        
        return "\n".join(html_parts)
    
    def to_json(self) -> Dict[str, Any]:
        """
        Generate JSON representation.
        
        Returns:
            Dictionary with all annotation data
        """
        return {
            'participant_id': self.transcript.participant_id,
            'diagnosis': self.transcript.diagnosis,
            'component': self.component,
            'total_annotations': len(self.annotations),
            'utterances': [
                {
                    'idx': idx,
                    'speaker': u.speaker,
                    'text': u.text,
                    'annotations': [
                        {
                            'type': a.annotation_type.value,
                            'feature_name': a.feature_name,
                            'feature_value': a.feature_value,
                            'description': a.description,
                            'start_pos': a.start_pos,
                            'end_pos': a.end_pos,
                            'color': a.color_code,
                        }
                        for a in self.get_annotations_for_utterance(idx)
                    ]
                }
                for idx, u in enumerate(self.transcript.utterances)
            ],
            'annotation_summary': self._get_annotation_summary(),
        }
    
    def _annotate_text_with_symbols(
        self,
        text: str,
        annotations: List[FeatureAnnotation]
    ) -> str:
        """Insert symbol markers into text."""
        if not annotations:
            return text
        
        # Sort annotations by position (reverse order for safe insertion)
        sorted_anns = sorted(annotations, key=lambda a: a.start_pos, reverse=True)
        
        result = text
        for ann in sorted_anns:
            symbol = ANNOTATION_SYMBOLS.get(ann.annotation_type, "[*]")
            
            # Insert markers around the annotated region
            if ann.end_pos <= len(result):
                result = (
                    result[:ann.start_pos] +
                    symbol +
                    result[ann.start_pos:ann.end_pos] +
                    symbol +
                    result[ann.end_pos:]
                )
        
        return result
    
    def _annotate_text_with_html(
        self,
        text: str,
        annotations: List[FeatureAnnotation]
    ) -> str:
        """Generate HTML with highlighted spans."""
        if not annotations:
            return html.escape(text)
        
        # Sort annotations by position
        sorted_anns = sorted(annotations, key=lambda a: a.start_pos)
        
        result_parts = []
        last_pos = 0
        
        for ann in sorted_anns:
            # Add text before this annotation
            if ann.start_pos > last_pos:
                result_parts.append(html.escape(text[last_pos:ann.start_pos]))
            
            # Add annotated span
            color = ann.color_code
            span_text = html.escape(text[ann.start_pos:ann.end_pos])
            tooltip = html.escape(f"{ann.feature_name}: {ann.description}")
            
            result_parts.append(
                f'<span class="annotation" '
                f'style="background-color: {color}20; border-bottom: 2px solid {color};" '
                f'title="{tooltip}" '
                f'data-type="{ann.annotation_type.value}">'
                f'{span_text}</span>'
            )
            
            last_pos = ann.end_pos
        
        # Add remaining text
        if last_pos < len(text):
            result_parts.append(html.escape(text[last_pos:]))
        
        return "".join(result_parts)
    
    def _generate_html_legend(self) -> str:
        """Generate HTML legend for annotation types."""
        unique_types = set(a.annotation_type for a in self.annotations)
        
        legend_items = []
        for ann_type in unique_types:
            color = ANNOTATION_COLORS.get(ann_type, "#808080")
            legend_items.append(
                f'<span class="legend-item" style="border-left: 4px solid {color}; padding-left: 8px;">'
                f'{ann_type.value}</span>'
            )
        
        return (
            '<div class="legend">'
            '<h4>Annotation Legend</h4>'
            '<div class="legend-items">' +
            " ".join(legend_items) +
            '</div></div>'
        )
    
    def _get_html_styles(self) -> str:
        """Get CSS styles for HTML output."""
        return """
        <style>
        .annotated-transcript {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 900px;
            margin: 20px auto;
            padding: 20px;
            background: #fafafa;
            border-radius: 8px;
        }
        .legend {
            background: #fff;
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 20px;
            border: 1px solid #e0e0e0;
        }
        .legend h4 {
            margin: 0 0 10px 0;
            color: #333;
        }
        .legend-items {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        .legend-item {
            font-size: 12px;
            padding: 4px 8px;
            background: #f5f5f5;
            border-radius: 4px;
        }
        .transcript-content {
            background: #fff;
            padding: 20px;
            border-radius: 6px;
            border: 1px solid #e0e0e0;
        }
        .utterance {
            padding: 8px 0;
            border-bottom: 1px solid #f0f0f0;
            line-height: 1.6;
        }
        .utterance:last-child {
            border-bottom: none;
        }
        .speaker {
            font-weight: 600;
            color: #1976D2;
        }
        .annotation {
            padding: 2px 4px;
            border-radius: 3px;
            cursor: help;
            transition: all 0.2s ease;
        }
        .annotation:hover {
            filter: brightness(0.95);
        }
        </style>
        """
    
    def _get_annotation_summary(self) -> Dict[str, int]:
        """Get summary count of annotation types."""
        summary = {}
        for ann in self.annotations:
            type_name = ann.annotation_type.value
            summary[type_name] = summary.get(type_name, 0) + 1
        return summary


class TranscriptAnnotator:
    """
    Generates annotations from feature extraction results.
    
    This class takes feature extraction output and maps it back to
    specific locations in the transcript for annotation.
    
    Example:
        >>> annotator = TranscriptAnnotator()
        >>> annotated = annotator.annotate(transcript, features)
        >>> print(annotated.to_html())
    """
    
    # Patterns for detecting features in text
    ECHOLALIA_PATTERNS = [
        r'\b(\w+(?:\s+\w+)?)\s+\1\b',  # Immediate repetition
    ]
    
    PRONOUN_PATTERNS = {
        'reversal': [
            (r'\byou\b.*\bwant\b', 'you/I reversal context'),
            (r'\bme\b.*\bgive\s+you\b', 'me/you reversal context'),
        ]
    }
    
    QUESTION_PATTERNS = [
        r'\?',
        r'\b(what|where|when|why|how|who|which)\b',
    ]
    
    FILLED_PAUSE_PATTERNS = [
        r'\b(um|uh|er|ah|hmm|erm)\b',
    ]
    
    DISCOURSE_MARKER_PATTERNS = [
        r'\b(well|so|okay|right|you know|I mean|like)\b',
    ]
    
    REPAIR_PATTERNS = [
        r'\b(I mean|no wait|sorry|actually|I meant)\b',
        r'\.\.\.',  # Hesitation
        r'\[/\]',   # CHAT repair marker
        r'\[//\]',  # CHAT retracing marker
    ]
    
    def __init__(self, component: str = "pragmatic_conversational"):
        """
        Initialize the annotator.
        
        Args:
            component: Component name for annotations
        """
        self.component = component
        logger.info(f"TranscriptAnnotator initialized for {component}")
    
    def annotate(
        self,
        transcript: TranscriptData,
        features: Optional[Dict[str, Any]] = None,
        include_patterns: bool = True
    ) -> AnnotatedTranscript:
        """
        Generate annotations for a transcript.
        
        Args:
            transcript: Transcript to annotate
            features: Optional extracted features for context
            include_patterns: Whether to detect patterns in text
            
        Returns:
            AnnotatedTranscript with all annotations
        """
        annotated = AnnotatedTranscript(
            transcript=transcript,
            component=self.component,
            metadata={'features': features}
        )
        
        if include_patterns:
            # Detect patterns in each utterance
            for idx, utterance in enumerate(transcript.utterances):
                annotations = self._detect_patterns(utterance, idx)
                annotated.add_annotations(annotations)
        
        # Add feature-based annotations if features provided
        if features:
            feature_annotations = self._annotations_from_features(
                transcript, features
            )
            annotated.add_annotations(feature_annotations)
        
        logger.info(f"Generated {len(annotated.annotations)} annotations")
        
        return annotated
    
    def _detect_patterns(
        self,
        utterance: Utterance,
        utterance_idx: int
    ) -> List[FeatureAnnotation]:
        """Detect annotation-worthy patterns in an utterance."""
        annotations = []
        text = utterance.text
        
        # Detect echolalia patterns
        for pattern in self.ECHOLALIA_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                annotations.append(FeatureAnnotation(
                    annotation_type=AnnotationType.ECHOLALIA,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    utterance_idx=utterance_idx,
                    feature_name="echolalia_repetition",
                    description=f"Repeated phrase: '{match.group()}'",
                    metadata={'span_text': match.group()}
                ))
        
        # Detect questions
        for pattern in self.QUESTION_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                annotations.append(FeatureAnnotation(
                    annotation_type=AnnotationType.QUESTION,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    utterance_idx=utterance_idx,
                    feature_name="question_marker",
                    description="Question detected",
                    metadata={'span_text': match.group()}
                ))
        
        # Detect filled pauses
        for pattern in self.FILLED_PAUSE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                annotations.append(FeatureAnnotation(
                    annotation_type=AnnotationType.FILLED_PAUSE,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    utterance_idx=utterance_idx,
                    feature_name="filled_pause",
                    description=f"Filled pause: '{match.group()}'",
                    metadata={'span_text': match.group()}
                ))
        
        # Detect discourse markers
        for pattern in self.DISCOURSE_MARKER_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                annotations.append(FeatureAnnotation(
                    annotation_type=AnnotationType.DISCOURSE_MARKER,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    utterance_idx=utterance_idx,
                    feature_name="discourse_marker",
                    description=f"Discourse marker: '{match.group()}'",
                    metadata={'span_text': match.group()}
                ))
        
        # Detect repairs
        for pattern in self.REPAIR_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                annotations.append(FeatureAnnotation(
                    annotation_type=AnnotationType.REPAIR_INITIATION,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    utterance_idx=utterance_idx,
                    feature_name="repair_marker",
                    description=f"Repair detected: '{match.group()}'",
                    metadata={'span_text': match.group()}
                ))
        
        # Detect social greetings
        greeting_pattern = r'\b(hi|hello|bye|goodbye|thank you|thanks|please|sorry)\b'
        for match in re.finditer(greeting_pattern, text, re.IGNORECASE):
            annotations.append(FeatureAnnotation(
                annotation_type=AnnotationType.SOCIAL_GREETING,
                start_pos=match.start(),
                end_pos=match.end(),
                utterance_idx=utterance_idx,
                feature_name="social_language",
                description=f"Social phrase: '{match.group()}'",
                metadata={'span_text': match.group()}
            ))
        
        return annotations
    
    def _annotations_from_features(
        self,
        transcript: TranscriptData,
        features: Dict[str, Any]
    ) -> List[FeatureAnnotation]:
        """Generate annotations from extracted feature values."""
        annotations = []
        
        # Add long pause annotations if detected
        if features.get('long_pause_count', 0) > 0:
            # Mark utterances that might have long pauses before them
            for idx in range(1, len(transcript.utterances)):
                prev_utt = transcript.utterances[idx - 1]
                curr_utt = transcript.utterances[idx]
                
                # Check timing gap if available
                if prev_utt.end_timing and curr_utt.timing:
                    gap = curr_utt.timing - prev_utt.end_timing
                    if gap > 1.0:  # Long pause threshold
                        annotations.append(FeatureAnnotation(
                            annotation_type=AnnotationType.LONG_PAUSE,
                            start_pos=0,
                            end_pos=0,
                            utterance_idx=idx,
                            feature_name="long_pause",
                            feature_value=gap,
                            description=f"Long pause before utterance: {gap:.2f}s",
                        ))
        
        # Add topic shift annotations if high topic_shift_ratio
        if features.get('topic_shift_ratio', 0) > 0.3:
            # This is a simplified approach - real implementation would
            # need the actual topic shift detection results
            annotations.append(FeatureAnnotation(
                annotation_type=AnnotationType.TOPIC_SHIFT,
                start_pos=0,
                end_pos=0,
                utterance_idx=0,
                feature_name="high_topic_shift_ratio",
                feature_value=features.get('topic_shift_ratio'),
                description=f"High topic shift ratio detected: {features.get('topic_shift_ratio', 0):.2f}",
            ))
        
        return annotations


__all__ = [
    "AnnotatedTranscript",
    "FeatureAnnotation",
    "AnnotationType",
    "TranscriptAnnotator",
    "ANNOTATION_COLORS",
    "ANNOTATION_SYMBOLS",
]

