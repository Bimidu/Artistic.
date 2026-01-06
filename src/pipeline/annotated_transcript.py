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

# Import exact patterns and thresholds from feature extractors
from src.features.pragmatic_conversational.repair_detection import (
    RepairDetectionFeatures
)
from src.features.pragmatic_conversational.pragmatic_linguistic import (
    PragmaticLinguisticFeatures
)
from src.features.pragmatic_conversational.pause_latency import (
    PauseLatencyFeatures
)
from src.features.pragmatic_conversational.turn_taking import (
    TurnTakingFeatures
)

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
    # NOTE: STEREOTYPED_PHRASE removed - not an extracted feature
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
    
    # Syntactic Semantic features
    COMPLEX_SYNTAX = "complex_syntax"
    DEEP_DEPENDENCY = "deep_dependency"
    GRAMMATICAL_ERROR = "grammatical_error"
    TENSE_INCONSISTENCY = "tense_inconsistency"
    RICH_VOCABULARY = "rich_vocabulary"
    SEMANTIC_COHERENCE = "semantic_coherence"
    
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
    # STEREOTYPED_PHRASE removed - not an extracted feature
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
    
    # Syntactic Semantic (deep purples/violets)
    AnnotationType.COMPLEX_SYNTAX: "#6A1B9A",
    AnnotationType.DEEP_DEPENDENCY: "#4A148C",
    AnnotationType.GRAMMATICAL_ERROR: "#D32F2F",
    AnnotationType.TENSE_INCONSISTENCY: "#F57C00",
    AnnotationType.RICH_VOCABULARY: "#1976D2",
    AnnotationType.SEMANTIC_COHERENCE: "#388E3C",
    
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
    # STEREOTYPED_PHRASE removed - not an extracted feature
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
    
    # Patterns for detecting features in text (matching extractor patterns)
    ECHOLALIA_PATTERNS = [
        r'\b(\w+(?:\s+\w+)?)\s+\1\b',  # Immediate repetition
    ]
    
    PRONOUN_PATTERNS = {
        'reversal': [
            (r'\byou\b.*\bwant\b', 'you/I reversal - "you want" instead of "I want"'),
            (r'\byou\b.*\blike\b', 'you/I reversal - "you like" instead of "I like"'),
            (r'\byou\b.*\bneed\b', 'you/I reversal - "you need" instead of "I need"'),
            (r'\byou\b.*\bhave\b', 'you/I reversal - "you have" instead of "I have"'),
            (r'\bme\b.*\bgive\s+you\b', 'me/you reversal context'),
        ]
    }
    
    QUESTION_PATTERNS = [
        r'\?',  # Any question mark
    ]
    
    YES_NO_QUESTION_PATTERNS = [
        r'\b(is|are|do|does|did|can|will|would|have|has)\b.*\?',
    ]
    
    WH_QUESTION_PATTERNS = [
        r'\b(what|where|when|why|how|who|which)\b.*\?',
    ]
    
    FILLED_PAUSE_PATTERNS = [
        r'\bum+\b', r'\buh+\b', r'\ber\b', r'\bah\b', r'\behm\b',
        r'\bhmm\b', r'\bmm\b', r'\buhm\b', r'\bumm\b',
        r'&-um', r'&-uh', r'&-er', r'&-ah',  # CHAT format
    ]
    
    DISCOURSE_MARKER_PATTERNS = [
        r'\b(well|so|okay|right|you know|I mean|like)\b',
    ]
    
    CONTINUATION_MARKER_PATTERNS = [
        r'\b(and|then|so|also|plus|next)\b',
    ]
    
    ACKNOWLEDGMENT_PATTERNS = [
        r'\b(oh|I see|okay|yes|oh okay|ah|go on|I got it|uh huh|mm hmm)\b',
    ]
    
    REPAIR_PATTERNS = [
        r'\bi mean\b',
        r'\bno wait\b',
        r'\bsorry\b',
        r'\bactually\b',
        r'\bno\s+i\s+mean\b',
        r'\bwell\s+not\b',
        r'\bor\s+rather\b',
        r'\blet me\s+rephrase\b',
        r'\[/\]',   # CHAT retrace marker
        r'\[//\]',  # CHAT retrace with correction
        r'\[///\]', # CHAT reformulation
    ]
    
    CLARIFICATION_PATTERNS = [
        r'\bwhat\?',
        r'\bhuh\?',
        r'\bpardon\?',
        r'\bexcuse me\?',
        r'\bsay again\b',
        r'\bwhat did you\b',
        r'\bcan you repeat\b',
        r'\bi don\'?t understand\b',
        r'\bwhat do you mean\b',
        r'\bsorry\?',
    ]
    
    CONFIRMATION_PATTERNS = [
        r'\bdo you mean\b',
        r'\bso you\b',
        r'\blike\s+a\b',
        r'\byou mean\b',
        r'\bis that\b',
        r'\bright\?',
        r'\bokay\?',
    ]
    
    POLITENESS_PATTERNS = [
        r'\b(please|thank you|thanks|sorry|excuse me)\b',
    ]
    
    SOCIAL_PHRASES = [
        'hello', 'hi', 'hey', 'bye', 'goodbye', 'good morning', 'good night',
        'thank you', 'thanks', 'please', 'sorry', 'excuse me', 'you\'re welcome',
        'nice to meet you', 'how are you', 'see you later'
    ]
    
    FALSE_START_PATTERNS = [
        r'^[a-z]+\s+[a-z]+\s+[a-z]+\s*\.\.\.',  # Word word word ...
        r'^[a-z]+\s*\[/\]',  # Word followed by retrace
    ]
    
    WORD_REPETITION_PATTERNS = [
        r'\b(\w+)\s+\1\s+\1\b',  # Same word 3+ times
    ]
    
    def __init__(self, component: str = "pragmatic_conversational"):
        """
        Initialize the annotator.
        
        Args:
            component: Component name for annotations
        """
        self.component = component
        
        # Initialize feature extractors to get exact patterns and thresholds
        self.repair_extractor = RepairDetectionFeatures()
        self.linguistic_extractor = PragmaticLinguisticFeatures()
        self.pause_extractor = PauseLatencyFeatures()
        self.turn_extractor = TurnTakingFeatures()
        
        # Use exact patterns from extractors
        self.REPAIR_PATTERNS = self.repair_extractor.SELF_REPAIR_PATTERNS + \
                               [re.escape(m) for m in self.repair_extractor.CHAT_RETRACE_MARKERS]
        self.CLARIFICATION_PATTERNS = self.repair_extractor.CLARIFICATION_PATTERNS
        self.CONFIRMATION_PATTERNS = self.repair_extractor.CONFIRMATION_PATTERNS
        self.ACKNOWLEDGMENT_PATTERNS = self.repair_extractor.ACKNOWLEDGMENT_PATTERNS
        self.FILLED_PAUSE_PATTERNS = self.pause_extractor.FILLED_PAUSE_PATTERNS
        
        # Use exact thresholds from extractors
        self.OVERLAP_THRESHOLD = self.turn_extractor.OVERLAP_THRESHOLD_MS / 1000.0  # Convert to seconds
        self.INTERRUPTION_THRESHOLD = self.turn_extractor.INTERRUPTION_THRESHOLD_MS / 1000.0
        self.LONG_PAUSE_THRESHOLD = self.turn_extractor.LONG_PAUSE_THRESHOLD_SEC
        self.VERY_LONG_PAUSE_THRESHOLD = self.pause_extractor.VERY_LONG_PAUSE_THRESHOLD
        
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
            
            # Detect delayed echolalia (requires looking across utterances)
            delayed_echolalia = self._detect_delayed_echolalia(transcript)
            annotated.add_annotations(delayed_echolalia)
        
        # Add feature-based annotations if features provided
        if features:
            feature_annotations = self._annotations_from_features(
                transcript, features
            )
            annotated.add_annotations(feature_annotations)
        
        logger.info(f"Generated {len(annotated.annotations)} annotations")
        
        return annotated
    
    def _detect_delayed_echolalia(
        self,
        transcript: TranscriptData
    ) -> List[FeatureAnnotation]:
        """Detect delayed echolalia (repetition of earlier utterances)."""
        annotations = []
        utterance_texts = [u.text.lower().strip() for u in transcript.utterances]
        
        for idx, utterance in enumerate(transcript.utterances):
            if utterance.speaker != 'CHI' or idx == 0:
                continue
            
            child_text = utterance.text.lower().strip()
            if len(child_text.split()) < 2:
                continue
            
            # Check for delayed echolalia (look back up to 10 turns)
            for j in range(max(0, idx - 10), idx):
                if utterance_texts[j] == child_text:
                    annotations.append(FeatureAnnotation(
                        annotation_type=AnnotationType.ECHOLALIA,
                        start_pos=0,
                        end_pos=len(utterance.text),
                        utterance_idx=idx,
                        feature_name="delayed_echolalia",
                        description=f"Delayed echolalia (repeats utterance from {idx - j} turns ago)",
                        metadata={'type': 'delayed', 'turns_ago': idx - j, 'original_idx': j}
                    ))
                    break
        
        return annotations
    
    def _detect_patterns(
        self,
        utterance: Utterance,
        utterance_idx: int
    ) -> List[FeatureAnnotation]:
        """Detect annotation-worthy patterns in an utterance."""
        annotations = []
        text = utterance.text
        
        # Detect echolalia patterns (immediate repetition)
        for pattern in self.ECHOLALIA_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                annotations.append(FeatureAnnotation(
                    annotation_type=AnnotationType.ECHOLALIA,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    utterance_idx=utterance_idx,
                    feature_name="immediate_echolalia",
                    description=f"Immediate repetition: '{match.group()}'",
                    metadata={'span_text': match.group(), 'type': 'immediate'}
                ))
        
        # Detect questions (all types)
        if '?' in text:
            # Check for WH questions
            for pattern in self.WH_QUESTION_PATTERNS:
                if re.search(pattern, text, re.IGNORECASE):
                    annotations.append(FeatureAnnotation(
                        annotation_type=AnnotationType.QUESTION,
                        start_pos=0,
                        end_pos=len(text),
                        utterance_idx=utterance_idx,
                        feature_name="wh_question",
                        description="WH-question detected",
                        metadata={'type': 'wh'}
                    ))
                    break
            else:
                # Check for yes/no questions
                for pattern in self.YES_NO_QUESTION_PATTERNS:
                    if re.search(pattern, text, re.IGNORECASE):
                        annotations.append(FeatureAnnotation(
                            annotation_type=AnnotationType.QUESTION,
                            start_pos=0,
                            end_pos=len(text),
                            utterance_idx=utterance_idx,
                            feature_name="yes_no_question",
                            description="Yes/No question detected",
                            metadata={'type': 'yes_no'}
                        ))
                        break
                else:
                    # Generic question
                    annotations.append(FeatureAnnotation(
                        annotation_type=AnnotationType.QUESTION,
                        start_pos=text.rfind('?'),
                        end_pos=text.rfind('?') + 1,
                        utterance_idx=utterance_idx,
                        feature_name="question",
                        description="Question detected",
                        metadata={'type': 'generic'}
                    ))
        
        # Detect filled pauses (with specific types)
        for pattern in self.FILLED_PAUSE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                filler_type = 'um' if 'um' in match.group().lower() else 'uh' if 'uh' in match.group().lower() else 'other'
                annotations.append(FeatureAnnotation(
                    annotation_type=AnnotationType.FILLED_PAUSE,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    utterance_idx=utterance_idx,
                    feature_name="filled_pause",
                    description=f"Filled pause: '{match.group()}'",
                    metadata={'span_text': match.group(), 'type': filler_type}
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
        
        # Detect continuation markers (could be topic maintenance indicator)
        for pattern in self.CONTINUATION_MARKER_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                annotations.append(FeatureAnnotation(
                    annotation_type=AnnotationType.TOPIC_MAINTENANCE,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    utterance_idx=utterance_idx,
                    feature_name="continuation_marker",
                    description=f"Continuation marker: '{match.group()}'",
                    metadata={'span_text': match.group()}
                ))
        
        # Detect acknowledgments
        for pattern in self.ACKNOWLEDGMENT_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                annotations.append(FeatureAnnotation(
                    annotation_type=AnnotationType.REPAIR_COMPLETION,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    utterance_idx=utterance_idx,
                    feature_name="acknowledgment",
                    description=f"Acknowledgment: '{match.group()}'",
                    metadata={'span_text': match.group()}
                ))
        
        # Detect repairs (self-repair initiations)
        for pattern in self.REPAIR_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                annotations.append(FeatureAnnotation(
                    annotation_type=AnnotationType.REPAIR_INITIATION,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    utterance_idx=utterance_idx,
                    feature_name="self_repair",
                    description=f"Self-repair: '{match.group()}'",
                    metadata={'span_text': match.group()}
                ))
        
        # Detect clarification requests
        for pattern in self.CLARIFICATION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                annotations.append(FeatureAnnotation(
                    annotation_type=AnnotationType.CLARIFICATION_REQUEST,
                    start_pos=0,
                    end_pos=len(text),
                    utterance_idx=utterance_idx,
                    feature_name="clarification_request",
                    description="Clarification request",
                ))
                break
        
        # Detect confirmation checks
        for pattern in self.CONFIRMATION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                annotations.append(FeatureAnnotation(
                    annotation_type=AnnotationType.CLARIFICATION_REQUEST,
                    start_pos=0,
                    end_pos=len(text),
                    utterance_idx=utterance_idx,
                    feature_name="confirmation_check",
                    description="Confirmation check",
                ))
                break
        
        # Detect social greetings and politeness
        for phrase in self.SOCIAL_PHRASES:
            pattern = r'\b' + re.escape(phrase) + r'\b'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                is_greeting = phrase in ['hello', 'hi', 'hey', 'bye', 'goodbye', 'good morning', 'good night']
                is_politeness = phrase in ['please', 'thank you', 'thanks', 'sorry', 'excuse me', 'you\'re welcome']
                
                if is_greeting:
                    annotations.append(FeatureAnnotation(
                        annotation_type=AnnotationType.SOCIAL_GREETING,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        utterance_idx=utterance_idx,
                        feature_name="greeting",
                        description=f"Greeting: '{match.group()}'",
                        metadata={'span_text': match.group()}
                    ))
                elif is_politeness:
                    annotations.append(FeatureAnnotation(
                        annotation_type=AnnotationType.SOCIAL_GREETING,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        utterance_idx=utterance_idx,
                        feature_name="politeness_marker",
                        description=f"Politeness marker: '{match.group()}'",
                        metadata={'span_text': match.group()}
                    ))
        
        # Detect false starts
        for pattern in self.FALSE_START_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                annotations.append(FeatureAnnotation(
                    annotation_type=AnnotationType.REPAIR_INITIATION,
                    start_pos=0,
                    end_pos=min(20, len(text)),
                    utterance_idx=utterance_idx,
                    feature_name="false_start",
                    description="False start detected",
                ))
                break
        
        # Detect word repetitions (3+ times)
        for pattern in self.WORD_REPETITION_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                annotations.append(FeatureAnnotation(
                    annotation_type=AnnotationType.ECHOLALIA,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    utterance_idx=utterance_idx,
                    feature_name="word_repetition",
                    description=f"Word repetition: '{match.group()}'",
                    metadata={'span_text': match.group()}
                ))
        
        return annotations
    
    def _annotations_from_features(
        self,
        transcript: TranscriptData,
        features: Dict[str, Any]
    ) -> List[FeatureAnnotation]:
        """
        Generate annotations from extracted feature values.
        
        Uses the extractors' patterns and thresholds to replicate their detection logic,
        but tracks instances instead of just counts. This ensures annotations match
        exactly what the extractors detected.
        """
        annotations = []
        
        all_utterances = transcript.utterances
        has_timing = self.turn_extractor._has_timing_info(all_utterances)
        adult_codes = {'MOT', 'FAT', 'INV', 'INV1', 'INV2', 'EXA', 'EXP'}
        
        # ========================================
        # Turn-Taking Features (Section 3.3.1)
        # ========================================
        
        # Overlaps: Replicate logic from _calculate_overlaps
        if features.get('overlap_count', 0) > 0:
            if has_timing and len(all_utterances) >= 2:
                threshold_sec = self.turn_extractor.OVERLAP_THRESHOLD_MS / 1000.0
                for i in range(1, len(all_utterances)):
                    prev = all_utterances[i - 1]
                    curr = all_utterances[i]
                    if prev.timing is not None and curr.timing is not None:
                        gap = curr.timing - prev.timing
                        if gap < threshold_sec:
                            annotations.append(FeatureAnnotation(
                                annotation_type=AnnotationType.OVERLAP,
                                start_pos=0,
                                end_pos=len(curr.text),
                                utterance_idx=i,
                                feature_name="overlap",
                                feature_value=abs(gap),
                                description=f"Overlap detected: {abs(gap):.2f}s",
                            ))
            else:
                # Text-based overlap detection (from _estimate_overlaps_from_text)
                overlap_markers = ['<', '>', '[>]', '[<]', '[/]', '[//]']
                for i, u in enumerate(all_utterances):
                    if any(marker in u.text for marker in overlap_markers):
                        annotations.append(FeatureAnnotation(
                            annotation_type=AnnotationType.OVERLAP,
                            start_pos=0,
                            end_pos=len(u.text),
                            utterance_idx=i,
                            feature_name="overlap",
                            description="Overlap detected (text marker)",
                        ))
        
        # Long pauses: Replicate logic from _calculate_inter_turn_gaps
        if features.get('long_pause_count', 0) > 0:
            for i in range(1, len(all_utterances)):
                prev = all_utterances[i - 1]
                curr = all_utterances[i]
                if prev.end_timing is not None and curr.timing is not None:
                    gap = curr.timing - prev.end_timing
                    if gap > self.turn_extractor.LONG_PAUSE_THRESHOLD_SEC:
                        annotations.append(FeatureAnnotation(
                            annotation_type=AnnotationType.LONG_PAUSE,
                            start_pos=0,
                            end_pos=0,
                            utterance_idx=i,
                            feature_name="long_pause",
                            feature_value=gap,
                            description=f"Long pause before utterance: {gap:.2f}s",
                        ))
        
        # Interruptions: Replicate logic from _calculate_interruptions
        if features.get('interruption_count', 0) > 0:
            interruption_markers = ['[//]', '+/', '+//', '<', '>', '[<]', '[>]']
            threshold_sec = self.turn_extractor.INTERRUPTION_THRESHOLD_MS / 1000.0
            
            for i in range(1, len(all_utterances)):
                prev = all_utterances[i - 1]
                curr = all_utterances[i]
                
                has_interruption_marker = any(
                    marker in curr.text or marker in prev.text
                    for marker in interruption_markers
                )
                
                timing_interruption = False
                if has_timing and prev.timing is not None and curr.timing is not None:
                    gap = curr.timing - prev.timing
                    if gap < threshold_sec and prev.speaker != curr.speaker:
                        timing_interruption = True
                
                if (has_interruption_marker or timing_interruption) and prev.speaker != curr.speaker:
                    gap = 0.0
                    if prev.end_timing and curr.timing:
                        gap = curr.timing - prev.end_timing
                    annotations.append(FeatureAnnotation(
                        annotation_type=AnnotationType.INTERRUPTION,
                        start_pos=0,
                        end_pos=len(curr.text),
                        utterance_idx=i,
                        feature_name="interruption",
                        feature_value=gap,
                        description=f"Interruption detected: {gap:.2f}s gap" if gap > 0 else "Interruption detected",
                    ))
        
        # Turn starts: Simple speaker change detection
        if len(all_utterances) > 0:
            annotations.append(FeatureAnnotation(
                annotation_type=AnnotationType.TURN_START,
                start_pos=0,
                end_pos=min(10, len(all_utterances[0].text)),
                utterance_idx=0,
                feature_name="turn_start",
                description="Turn start",
            ))
            for i in range(1, len(all_utterances)):
                if all_utterances[i - 1].speaker != all_utterances[i].speaker:
                    annotations.append(FeatureAnnotation(
                        annotation_type=AnnotationType.TURN_START,
                        start_pos=0,
                        end_pos=min(10, len(all_utterances[i].text)),
                        utterance_idx=i,
                        feature_name="turn_start",
                        description="Turn start",
                    ))
        
        # Response latency: Replicate logic from pause extractor
        if features.get('delayed_response_count', 0) > 0 or features.get('response_latency_mean', 0) > 0:
            normal_response_time = self.pause_extractor.NORMAL_RESPONSE_TIME
            for i in range(1, len(all_utterances)):
                prev = all_utterances[i - 1]
                curr = all_utterances[i]
                if prev.end_timing and curr.timing and curr.speaker == 'CHI':
                    gap = curr.timing - prev.end_timing
                    if gap > normal_response_time:
                        annotations.append(FeatureAnnotation(
                            annotation_type=AnnotationType.RESPONSE_LATENCY,
                            start_pos=0,
                            end_pos=len(curr.text),
                            utterance_idx=i,
                            feature_name="delayed_response",
                            feature_value=gap,
                            description=f"Delayed response: {gap:.2f}s",
                        ))
        
        # ========================================
        # Topic Coherence Features (Section 3.3.2)
        # ========================================
        
        # Detect topic shifts and maintenance
        # Only annotate if the feature extractor found topic shifts
        topic_shift_count = int(features.get('topic_shift_count', 0))
        abrupt_shift_count = int(features.get('abrupt_topic_shift_count', 0))
        total_shifts = topic_shift_count + abrupt_shift_count
        
        if total_shifts > 0:
            # Mark utterances where topic might shift (simplified heuristic)
            # Limit to the number of shifts actually detected
            shifts_found = 0
            for idx in range(1, len(transcript.utterances)):
                if shifts_found >= total_shifts:
                    break
                    
                prev_words = set(transcript.utterances[idx - 1].text.lower().split())
                curr_words = set(transcript.utterances[idx].text.lower().split())
                
                # If very few words overlap, likely topic shift
                overlap = prev_words.intersection(curr_words)
                overlap_ratio = len(overlap) / len(prev_words) if prev_words else 0
                
                if overlap_ratio < 0.2 and len(curr_words) > 2:  # Less than 20% overlap
                    annotations.append(FeatureAnnotation(
                        annotation_type=AnnotationType.TOPIC_SHIFT,
                        start_pos=0,
                        end_pos=len(transcript.utterances[idx].text),
                        utterance_idx=idx,
                        feature_name="topic_shift",
                        description=f"Topic shift (low word overlap: {overlap_ratio:.2f})",
                    ))
                    shifts_found += 1
        
        # Topic maintenance - only annotate if we have high overlap utterances
        # This is a derived feature, so we annotate based on lexical_overlap_child feature
        if features.get('lexical_overlap_child', 0) > 0.5:
            # Mark some high-overlap utterances as topic maintenance
            maintenance_count = 0
            max_maintenance = min(10, len(transcript.utterances) // 4)  # Limit annotations
            for idx in range(1, len(transcript.utterances)):
                if maintenance_count >= max_maintenance:
                    break
                    
                prev_words = set(transcript.utterances[idx - 1].text.lower().split())
                curr_words = set(transcript.utterances[idx].text.lower().split())
                overlap_ratio = len(prev_words.intersection(curr_words)) / len(prev_words) if prev_words else 0
                
                if overlap_ratio > 0.5:  # High overlap = topic maintenance
                    annotations.append(FeatureAnnotation(
                        annotation_type=AnnotationType.TOPIC_MAINTENANCE,
                        start_pos=0,
                        end_pos=len(transcript.utterances[idx].text),
                        utterance_idx=idx,
                        feature_name="topic_maintenance",
                        description=f"Topic maintenance (high word overlap: {overlap_ratio:.2f})",
                    ))
                    maintenance_count += 1
        
        # Mark off-topic responses if feature indicates
        if features.get('off_topic_response_count', 0) > 0:
            # Mark child utterances that might be off-topic
            for idx, utterance in enumerate(transcript.utterances):
                if utterance.speaker == 'CHI' and idx > 0:
                    # Simple heuristic: very short responses after long questions
                    prev_utt = transcript.utterances[idx - 1]
                    if len(prev_utt.text.split()) > 5 and len(utterance.text.split()) < 3:
                        if '?' in prev_utt.text:  # Was a question
                            annotations.append(FeatureAnnotation(
                                annotation_type=AnnotationType.TOPIC_SHIFT,
                                start_pos=0,
                                end_pos=len(utterance.text),
                                utterance_idx=idx,
                                feature_name="off_topic_response",
                                description="Potential off-topic response",
                            ))
        
        # ========================================
        # Repair Detection Features (Section 3.3.4)
        # ========================================
        
        # Self-repair: Replicate logic from _calculate_self_repair
        if features.get('self_repair_count', 0) > 0:
            for i, u in enumerate(all_utterances):
                text = u.text.lower()
                repair_found = False
                
                # Check for linguistic self-repair markers (using extractor's patterns)
                for pattern in self.repair_extractor.SELF_REPAIR_PATTERNS:
                    if re.search(pattern, text):
                        repair_found = True
                        break
                
                # Check for CHAT retrace markers (using extractor's patterns)
                if not repair_found:
                    for pattern in self.repair_extractor.CHAT_RETRACE_MARKERS:
                        if re.search(pattern, u.text):
                            repair_found = True
                            break
                
                if repair_found:
                    annotations.append(FeatureAnnotation(
                        annotation_type=AnnotationType.REPAIR_INITIATION,
                        start_pos=0,
                        end_pos=len(u.text),
                        utterance_idx=i,
                        feature_name="self_repair",
                        description="Self-repair detected",
                    ))
        
        # Clarification requests: Replicate logic from _calculate_clarification_requests
        if features.get('clarification_request_count', 0) > 0:
            for i, u in enumerate(all_utterances):
                text = u.text.lower()
                for pattern in self.repair_extractor.CLARIFICATION_PATTERNS:
                    if re.search(pattern, text):
                        annotations.append(FeatureAnnotation(
                            annotation_type=AnnotationType.CLARIFICATION_REQUEST,
                            start_pos=0,
                            end_pos=len(u.text),
                            utterance_idx=i,
                            feature_name="clarification_request",
                            description="Clarification request",
                        ))
                        break
        
        # Repetition repairs: Replicate logic from _calculate_repetition_repairs
        if features.get('repetition_repair_count', 0) > 0 or features.get('exact_repetition_count', 0) > 0:
            for i in range(1, len(all_utterances)):
                prev = all_utterances[i - 1]
                curr = all_utterances[i]
                
                # Skip if same speaker (not a repair situation)
                if prev.speaker == curr.speaker:
                    continue
                
                prev_words = set(prev.text.lower().split())
                curr_words = set(curr.text.lower().split())
                
                if not prev_words or not curr_words:
                    continue
                
                # Exact repetition
                if prev.text.lower().strip() == curr.text.lower().strip():
                    annotations.append(FeatureAnnotation(
                        annotation_type=AnnotationType.REPAIR_COMPLETION,
                        start_pos=0,
                        end_pos=len(curr.text),
                        utterance_idx=i,
                        feature_name="repetition_repair",
                        description="Repetition repair (exact)",
                    ))
                # Partial repetition (significant overlap)
                else:
                    overlap = prev_words & curr_words
                    overlap_ratio = len(overlap) / len(prev_words)
                    if overlap_ratio > 0.5:
                        annotations.append(FeatureAnnotation(
                            annotation_type=AnnotationType.REPAIR_COMPLETION,
                            start_pos=0,
                            end_pos=len(curr.text),
                            utterance_idx=i,
                            feature_name="partial_repetition_repair",
                            description=f"Partial repetition repair ({overlap_ratio:.1%} overlap)",
                        ))
        
        # ========================================
        # Pragmatic Linguistic Features
        # ========================================
        
        # Echolalia: Replicate logic from _calculate_echolalia
        if features.get('immediate_echolalia_count', 0) > 0 or features.get('delayed_echolalia_count', 0) > 0:
            utterance_texts = [u.text.lower().strip() for u in all_utterances]
            
            for i, utterance in enumerate(all_utterances):
                if utterance.speaker != 'CHI':
                    continue
                
                child_text = utterance.text.lower().strip()
                if len(child_text.split()) < 2:
                    continue
                
                # Immediate echolalia (replicate extractor logic)
                if i > 0:
                    prev_text = utterance_texts[i - 1]
                    
                    if child_text == prev_text:
                        annotations.append(FeatureAnnotation(
                            annotation_type=AnnotationType.ECHOLALIA,
                            start_pos=0,
                            end_pos=len(utterance.text),
                            utterance_idx=i,
                            feature_name="immediate_echolalia",
                            description="Immediate echolalia (exact repetition)",
                            metadata={'type': 'immediate'}
                        ))
                        continue
                    
                    # Partial match (replicate extractor's _is_partial_repetition logic)
                    words1 = set(child_text.split())
                    words2 = set(prev_text.split())
                    if words1 and words2:
                        overlap = words1.intersection(words2)
                        overlap_ratio = len(overlap) / len(words2)
                        if overlap_ratio > 0.6:  # Same threshold as extractor
                            annotations.append(FeatureAnnotation(
                                annotation_type=AnnotationType.ECHOLALIA,
                                start_pos=0,
                                end_pos=len(utterance.text),
                                utterance_idx=i,
                                feature_name="partial_repetition",
                                description=f"Partial repetition ({overlap_ratio:.1%} overlap)",
                                metadata={'type': 'partial'}
                            ))
                
                # Delayed echolalia (replicate extractor logic: look back up to 10 turns)
                for j in range(max(0, i - 10), i - 1):
                    if utterance_texts[j] == child_text:
                        annotations.append(FeatureAnnotation(
                            annotation_type=AnnotationType.ECHOLALIA,
                            start_pos=0,
                            end_pos=len(utterance.text),
                            utterance_idx=i,
                            feature_name="delayed_echolalia",
                            description=f"Delayed echolalia (repeats utterance from {i - j} turns ago)",
                            metadata={'type': 'delayed', 'turns_ago': i - j}
                        ))
                        break
        
        # Detect pronoun reversals (using same patterns as extractor)
        if features.get('pronoun_reversal_count', 0) > 0:
            for idx, utterance in enumerate(transcript.utterances):
                if utterance.speaker == 'CHI':
                    text = utterance.text.lower()
                    for pattern, desc in self.PRONOUN_PATTERNS['reversal']:
                        if re.search(pattern, text, re.IGNORECASE):
                            annotations.append(FeatureAnnotation(
                                annotation_type=AnnotationType.PRONOUN_REVERSAL,
                                start_pos=0,
                                end_pos=len(utterance.text),
                                utterance_idx=idx,
                                feature_name="pronoun_reversal",
                                description=desc,
                            ))
                            break
        
        # NOTE: "stereotyped_phrase" is NOT an extracted feature.
        # The pragmatic component does NOT extract this feature.
        # Removed annotation for non-existent feature.
        
        # ========================================
        # Pause and Latency Features (Section 3.3.3)
        # ========================================
        
        # Detect very long pauses (> 4.32s threshold)
        for idx in range(1, len(transcript.utterances)):
            prev_utt = transcript.utterances[idx - 1]
            curr_utt = transcript.utterances[idx]
            
            if prev_utt.end_timing and curr_utt.timing:
                gap = curr_utt.timing - prev_utt.end_timing
                if gap > 4.32:  # Very long pause threshold
                    annotations.append(FeatureAnnotation(
                        annotation_type=AnnotationType.LONG_PAUSE,
                        start_pos=0,
                        end_pos=0,
                        utterance_idx=idx,
                        feature_name="very_long_pause",
                        feature_value=gap,
                        description=f"Very long pause: {gap:.2f}s (disengagement threshold)",
                    ))
        
        # Detect false starts (from pause/latency features)
        if features.get('false_start_count', 0) > 0:
            for idx, utterance in enumerate(transcript.utterances):
                text = utterance.text
                # False start patterns: word word word ... or word [/]
                for pattern in self.FALSE_START_PATTERNS:
                    if re.search(pattern, text, re.IGNORECASE):
                        annotations.append(FeatureAnnotation(
                            annotation_type=AnnotationType.REPAIR_INITIATION,
                            start_pos=0,
                            end_pos=min(30, len(text)),
                            utterance_idx=idx,
                            feature_name="false_start",
                            description="False start detected",
                        ))
                        break
        
        # Detect word repetitions (from pause/latency features)
        if features.get('word_repetition_count', 0) > 0:
            for idx, utterance in enumerate(transcript.utterances):
                text = utterance.text
                # Find word repetitions (same word 3+ times)
                for pattern in self.WORD_REPETITION_PATTERNS:
                    for match in re.finditer(pattern, text, re.IGNORECASE):
                        annotations.append(FeatureAnnotation(
                            annotation_type=AnnotationType.ECHOLALIA,
                            start_pos=match.start(),
                            end_pos=match.end(),
                            utterance_idx=idx,
                            feature_name="word_repetition",
                            description=f"Word repetition: '{match.group()}'",
                            metadata={'span_text': match.group()}
                        ))
        
        # Mark child-initiated turns (from turn-taking features)
        if features.get('child_initiated_turns', 0) > 0:
            for idx, utterance in enumerate(transcript.utterances):
                if utterance.speaker == 'CHI' and idx > 0:
                    prev_speaker = transcript.utterances[idx - 1].speaker
                    # Child-initiated if previous speaker was also child (no adult in between)
                    if prev_speaker == 'CHI':
                        annotations.append(FeatureAnnotation(
                            annotation_type=AnnotationType.TURN_START,
                            start_pos=0,
                            end_pos=min(10, len(utterance.text)),
                            utterance_idx=idx,
                            feature_name="child_initiated_turn",
                            description="Child-initiated turn",
                        ))
        
        # Mark consecutive child turns (monologues)
        if features.get('max_consecutive_child_turns', 0) > 2:
            consecutive_count = 0
            for idx, utterance in enumerate(transcript.utterances):
                if utterance.speaker == 'CHI':
                    consecutive_count += 1
                    if consecutive_count >= 3:  # Mark as part of monologue
                        annotations.append(FeatureAnnotation(
                            annotation_type=AnnotationType.TURN_START,
                            start_pos=0,
                            end_pos=len(utterance.text),
                            utterance_idx=idx,
                            feature_name="child_monologue",
                            description=f"Part of child monologue ({consecutive_count} consecutive turns)",
                            metadata={'consecutive_count': consecutive_count}
                        ))
                else:
                    consecutive_count = 0
        
        # ========================================
        # Linguistic Features
        # ========================================
        
        # Note: Simple/Complex sentence features are NOT extracted by the pragmatic component.
        # The pragmatic component focuses on conversational/pragmatic features only.
        # These would be part of a linguistic/syntactic component, not pragmatic.
        
        return annotations


__all__ = [
    "AnnotatedTranscript",
    "FeatureAnnotation",
    "AnnotationType",
    "TranscriptAnnotator",
    "ANNOTATION_COLORS",
    "ANNOTATION_SYMBOLS",
]

