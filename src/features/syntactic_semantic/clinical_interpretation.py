"""
Clinical Interpretation for Syntactic & Semantic Features

This module provides clinical interpretations for syntactic and semantic features
to help clinicians and researchers understand the significance of feature values
in the context of ASD detection.

Author: Enhanced Implementation (based on Randil Haturusinghe's work)
"""

from typing import Dict, Any, List, Tuple
from dataclasses import dataclass


@dataclass
class FeatureInterpretation:
    """
    Clinical interpretation for a feature.

    Attributes:
        feature_name: Name of the feature
        clinical_meaning: What the feature measures
        asd_pattern: Typical pattern in children with ASD
        td_pattern: Typical pattern in typically developing children
        interpretation_guide: How to interpret values
        clinical_relevance: Why this feature matters clinically
    """
    feature_name: str
    clinical_meaning: str
    asd_pattern: str
    td_pattern: str
    interpretation_guide: str
    clinical_relevance: str


class ClinicalInterpreter:
    """
    Provides clinical interpretations for syntactic and semantic features.

    This class helps translate feature values into clinically meaningful
    insights for ASD detection and language assessment.

    Example:
        >>> interpreter = ClinicalInterpreter()
        >>> interpretation = interpreter.interpret_feature('avg_dependency_depth', 2.1)
        >>> print(interpretation)
    """

    def __init__(self):
        """Initialize clinical interpreter with feature descriptions."""
        self._feature_interpretations = self._initialize_interpretations()

    def _initialize_interpretations(self) -> Dict[str, FeatureInterpretation]:
        """Initialize clinical interpretations for all features."""
        return {
            # Syntactic Complexity Features
            'avg_dependency_depth': FeatureInterpretation(
                feature_name='Average Dependency Depth',
                clinical_meaning='Measures the average structural complexity of sentences through dependency tree depth',
                asd_pattern='Often lower (1.5-2.5) - simpler sentence structures with less embedding',
                td_pattern='Typically higher (2.5-4.0) - more complex, embedded sentence structures',
                interpretation_guide='Lower values suggest simpler syntax. Values <2.0 may indicate reduced syntactic complexity',
                clinical_relevance='Syntactic complexity is a marker of language development and cognitive-linguistic abilities'
            ),

            'max_dependency_depth': FeatureInterpretation(
                feature_name='Maximum Dependency Depth',
                clinical_meaning='Maximum observed structural complexity in any single utterance',
                asd_pattern='Lower maximum depth (3-6) - limited use of highly complex structures',
                td_pattern='Higher maximum depth (6-10+) - occasional use of very complex structures',
                interpretation_guide='Indicates capacity for complex language. Max depth <4 suggests limited syntactic range',
                clinical_relevance='Shows whether child can produce complex structures even if they don\'t use them consistently'
            ),

            'avg_dependency_distance': FeatureInterpretation(
                feature_name='Average Dependency Distance',
                clinical_meaning='Average distance between grammatically related words',
                asd_pattern='Often lower (1.0-1.5) - words that belong together are kept close',
                td_pattern='Higher (1.5-2.5) - comfortable with greater word separation',
                interpretation_guide='Very low values (<1.0) may indicate rigid, simple sentence patterns',
                clinical_relevance='Reflects syntactic planning and working memory capacity'
            ),

            'clause_complexity': FeatureInterpretation(
                feature_name='Clause Complexity',
                clinical_meaning='Average number of clause markers per utterance',
                asd_pattern='Lower (0.1-0.4) - fewer clauses, simpler constructions',
                td_pattern='Higher (0.4-1.0) - more multi-clause utterances',
                interpretation_guide='Values <0.3 suggest primarily simple sentences',
                clinical_relevance='Multi-clause sentences indicate advanced language development'
            ),

            'subordination_index': FeatureInterpretation(
                feature_name='Subordination Index',
                clinical_meaning='Frequency of subordinate clauses (because, when, that, etc.)',
                asd_pattern='Lower (0.1-0.3) - limited use of subordinate clauses',
                td_pattern='Higher (0.3-0.6) - comfortable with complex sentence embedding',
                interpretation_guide='Values <0.2 indicate predominantly simple sentences',
                clinical_relevance='Subordination is a hallmark of mature language use and cognitive complexity'
            ),

            'coordination_index': FeatureInterpretation(
                feature_name='Coordination Index',
                clinical_meaning='Frequency of coordinate clauses (and, but, or)',
                asd_pattern='Variable - may be high with repetitive "and" usage, or low',
                td_pattern='Moderate (0.2-0.5) - balanced use of coordination',
                interpretation_guide='Very high values (>0.6) may indicate repetitive coordination',
                clinical_relevance='Shows ability to connect ideas, but overuse may indicate rigid patterns'
            ),

            # Grammatical Accuracy Features
            'grammatical_error_rate': FeatureInterpretation(
                feature_name='Grammatical Error Rate',
                clinical_meaning='Proportion of utterances with grammatical incompleteness or errors',
                asd_pattern='Often higher (0.15-0.40) - more incomplete or ungrammatical utterances',
                td_pattern='Lower (0.05-0.15) - mostly grammatical speech',
                interpretation_guide='Rates >0.25 suggest significant grammatical difficulties',
                clinical_relevance='Grammatical accuracy reflects language learning and morphosyntactic development'
            ),

            'tense_consistency_score': FeatureInterpretation(
                feature_name='Tense Consistency Score',
                clinical_meaning='Consistency of tense usage across utterances',
                asd_pattern='Variable - may be inconsistent or rigidly consistent',
                td_pattern='Moderately consistent (0.6-0.8) with appropriate tense shifts',
                interpretation_guide='Very high (>0.9) or low (<0.4) scores warrant attention',
                clinical_relevance='Tense consistency reflects temporal reasoning and narrative coherence'
            ),

            'tense_variety': FeatureInterpretation(
                feature_name='Tense Variety',
                clinical_meaning='Diversity of tense forms used',
                asd_pattern='Often lower (0.3-0.5) - limited tense repertoire',
                td_pattern='Higher (0.5-0.8) - uses multiple tense forms appropriately',
                interpretation_guide='Values <0.4 suggest limited tense diversity',
                clinical_relevance='Tense variety indicates morphological development and temporal flexibility'
            ),

            'structure_diversity': FeatureInterpretation(
                feature_name='Structure Diversity',
                clinical_meaning='Variety of sentence structure patterns used',
                asd_pattern='Often lower (0.3-0.6) - more repetitive structures',
                td_pattern='Higher (0.6-0.9) - varied sentence patterns',
                interpretation_guide='Low diversity (<0.4) may indicate rigid language patterns',
                clinical_relevance='Structural flexibility is a marker of linguistic competence'
            ),

            'pos_tag_diversity': FeatureInterpretation(
                feature_name='Part-of-Speech Diversity',
                clinical_meaning='Variety of word types (nouns, verbs, adjectives, etc.) used',
                asd_pattern='May be lower - more restricted vocabulary types',
                td_pattern='Higher - uses full range of word types',
                interpretation_guide='Very low values (<0.05) suggest limited grammatical range',
                clinical_relevance='POS diversity reflects lexical and grammatical sophistication'
            ),

            # Sentence Structure Features
            'avg_parse_tree_height': FeatureInterpretation(
                feature_name='Average Parse Tree Height',
                clinical_meaning='Average hierarchical depth of sentence structure',
                asd_pattern='Lower (2-4) - flatter sentence structures',
                td_pattern='Higher (4-7) - more hierarchical structures',
                interpretation_guide='Heights <3 indicate predominantly simple structures',
                clinical_relevance='Parse tree height correlates with cognitive-linguistic complexity'
            ),

            'noun_phrase_complexity': FeatureInterpretation(
                feature_name='Noun Phrase Complexity',
                clinical_meaning='Average length/complexity of noun phrases',
                asd_pattern='Often lower (1.5-2.5 words) - simpler noun phrases',
                td_pattern='Higher (2.5-4.0 words) - more elaborate descriptions',
                interpretation_guide='Values <2.0 suggest minimal use of modifiers and elaboration',
                clinical_relevance='NP complexity reflects descriptive language abilities'
            ),

            'verb_phrase_complexity': FeatureInterpretation(
                feature_name='Verb Phrase Complexity',
                clinical_meaning='Average number of dependents per verb',
                asd_pattern='Lower (1-2) - simpler verb constructions',
                td_pattern='Higher (2-4) - more elaborate verb phrases',
                interpretation_guide='Values <1.5 indicate minimal verb elaboration',
                clinical_relevance='VP complexity shows action description and argument structure abilities'
            ),

            'prepositional_phrase_ratio': FeatureInterpretation(
                feature_name='Prepositional Phrase Ratio',
                clinical_meaning='Frequency of prepositional phrase usage',
                asd_pattern='Variable - may be lower in children with ASD',
                td_pattern='Moderate usage (0.3-0.8 per utterance)',
                interpretation_guide='Very low (<0.2) or very high (>1.5) ratios warrant attention',
                clinical_relevance='Prepositional phrases add spatial/temporal/relational information'
            ),

            # Semantic Features
            'semantic_coherence': FeatureInterpretation(
                feature_name='Semantic Coherence',
                clinical_meaning='Semantic similarity between consecutive utterances',
                asd_pattern='Often lower (0.2-0.5) - more topic shifts or tangential responses',
                td_pattern='Higher (0.5-0.8) - maintains topic continuity',
                interpretation_guide='Coherence <0.4 suggests difficulty maintaining conversational topic',
                clinical_relevance='Semantic coherence is crucial for conversation and social communication'
            ),

            'semantic_density': FeatureInterpretation(
                feature_name='Semantic Density',
                clinical_meaning='Average number of content words (meaningful words) per utterance',
                asd_pattern='Variable - may be lower or focused on specific topics',
                td_pattern='Moderate (3-6 content words per utterance)',
                interpretation_guide='Very low (<2) suggests limited information content',
                clinical_relevance='Semantic density reflects information-carrying capacity of language'
            ),

            'lexical_diversity_semantic': FeatureInterpretation(
                feature_name='Semantic Lexical Diversity',
                clinical_meaning='Ratio of unique to total content words (reduced repetition)',
                asd_pattern='Often lower (0.4-0.6) - more repetitive language',
                td_pattern='Higher (0.6-0.8) - more varied word use',
                interpretation_guide='Values <0.5 indicate significant word repetition',
                clinical_relevance='Lexical diversity is associated with vocabulary breadth and flexibility'
            ),

            'thematic_consistency': FeatureInterpretation(
                feature_name='Thematic Consistency',
                clinical_meaning='Proportion of content words repeated across utterances (topic maintenance)',
                asd_pattern='May be very high (perseveration) or very low (tangentiality)',
                td_pattern='Moderate (0.3-0.6) - maintains topics while introducing new ideas',
                interpretation_guide='Extreme values (<0.2 or >0.8) may indicate difficulties',
                clinical_relevance='Thematic consistency reflects topic maintenance in conversation'
            ),

            # Vocabulary Semantic Features
            'vocabulary_abstractness': FeatureInterpretation(
                feature_name='Vocabulary Abstractness',
                clinical_meaning='Ratio of abstract to concrete words',
                asd_pattern='Often lower (0.2-0.4) - more concrete vocabulary',
                td_pattern='Balanced (0.4-0.6) - mix of concrete and abstract concepts',
                interpretation_guide='Values <0.3 suggest predominantly concrete language',
                clinical_relevance='Abstract language reflects cognitive development and conceptual thinking'
            ),

            'semantic_field_diversity': FeatureInterpretation(
                feature_name='Semantic Field Diversity',
                clinical_meaning='Variety of semantic domains (topics) discussed',
                asd_pattern='May be lower - restricted to specific topics of interest',
                td_pattern='Higher - discusses varied topics',
                interpretation_guide='Low values suggest restricted range of topics',
                clinical_relevance='Semantic field diversity relates to restricted interests in ASD'
            ),

            'word_sense_diversity': FeatureInterpretation(
                feature_name='Word Sense Diversity',
                clinical_meaning='Average number of meanings (senses) per word used',
                asd_pattern='Variable - may use simpler or more specific words',
                td_pattern='Moderate - uses words with multiple senses appropriately',
                interpretation_guide='Interpret in context with other vocabulary measures',
                clinical_relevance='Indicates vocabulary sophistication and semantic knowledge'
            ),

            'content_word_ratio': FeatureInterpretation(
                feature_name='Content Word Ratio',
                clinical_meaning='Proportion of meaningful words vs. function words',
                asd_pattern='Variable - may be lower in some children with ASD',
                td_pattern='Moderate (0.4-0.6) - balanced content/function words',
                interpretation_guide='Very low (<0.3) may indicate reliance on formulaic phrases',
                clinical_relevance='Content ratio reflects information density and language efficiency'
            ),

            # Advanced Semantic Features
            'semantic_role_diversity': FeatureInterpretation(
                feature_name='Semantic Role Diversity',
                clinical_meaning='Variety of semantic roles (subject, object, etc.) used',
                asd_pattern='May be lower - simpler argument structures',
                td_pattern='Higher - uses varied sentence roles',
                interpretation_guide='Low values suggest limited sentence complexity',
                clinical_relevance='Semantic roles reflect understanding of event structure'
            ),

            'entity_density': FeatureInterpretation(
                feature_name='Entity Density',
                clinical_meaning='Frequency of named entities (people, places, things)',
                asd_pattern='Variable - may be higher for special interests, lower otherwise',
                td_pattern='Moderate - refers to entities appropriately',
                interpretation_guide='Interpret in context - very high may indicate perseveration',
                clinical_relevance='Entity usage reflects referential communication skills'
            ),

            'verb_argument_complexity': FeatureInterpretation(
                feature_name='Verb Argument Complexity',
                clinical_meaning='Average number of arguments (subjects, objects) per verb',
                asd_pattern='Often lower (0.5-1.5) - simpler verb structures',
                td_pattern='Higher (1.5-3.0) - more complex predicate-argument structures',
                interpretation_guide='Values <1.0 suggest very simple sentence patterns',
                clinical_relevance='Argument structure reflects grammatical and semantic sophistication'
            ),
        }

    def interpret_feature(self, feature_name: str, value: float) -> str:
        """
        Generate clinical interpretation for a feature value.

        Args:
            feature_name: Name of the feature
            value: Numerical value of the feature

        Returns:
            Clinical interpretation string
        """
        if feature_name not in self._feature_interpretations:
            return f"No interpretation available for feature: {feature_name}"

        interp = self._feature_interpretations[feature_name]

        interpretation = f"""
Feature: {interp.feature_name}
Value: {value:.3f}

What it measures:
{interp.clinical_meaning}

Typical patterns:
- ASD: {interp.asd_pattern}
- TD: {interp.td_pattern}

Interpretation:
{interp.interpretation_guide}

Clinical relevance:
{interp.clinical_relevance}
        """.strip()

        return interpretation

    def interpret_profile(self, features: Dict[str, float]) -> str:
        """
        Generate comprehensive interpretation for a feature profile.

        Args:
            features: Dictionary of feature names to values

        Returns:
            Comprehensive clinical interpretation
        """
        report_lines = ["=" * 70]
        report_lines.append("SYNTACTIC & SEMANTIC FEATURE PROFILE - CLINICAL INTERPRETATION")
        report_lines.append("=" * 70)
        report_lines.append("")

        # Group features by category
        categories = {
            'Syntactic Complexity': [
                'avg_dependency_depth', 'max_dependency_depth', 'avg_dependency_distance',
                'clause_complexity', 'subordination_index', 'coordination_index'
            ],
            'Grammatical Accuracy': [
                'grammatical_error_rate', 'tense_consistency_score', 'tense_variety',
                'structure_diversity', 'pos_tag_diversity'
            ],
            'Sentence Structure': [
                'avg_parse_tree_height', 'noun_phrase_complexity',
                'verb_phrase_complexity', 'prepositional_phrase_ratio'
            ],
            'Semantic Features': [
                'semantic_coherence', 'semantic_density',
                'lexical_diversity_semantic', 'thematic_consistency'
            ],
            'Vocabulary Semantics': [
                'vocabulary_abstractness', 'semantic_field_diversity',
                'word_sense_diversity', 'content_word_ratio'
            ],
            'Advanced Semantics': [
                'semantic_role_diversity', 'entity_density', 'verb_argument_complexity'
            ]
        }

        for category, feature_list in categories.items():
            report_lines.append(f"\n{category}:")
            report_lines.append("-" * 70)

            for feature in feature_list:
                if feature in features:
                    value = features[feature]
                    if feature in self._feature_interpretations:
                        interp = self._feature_interpretations[feature]
                        report_lines.append(f"\n  {interp.feature_name}: {value:.3f}")
                        report_lines.append(f"    {interp.interpretation_guide}")

        report_lines.append("\n" + "=" * 70)

        return "\n".join(report_lines)

    def get_asd_risk_indicators(self, features: Dict[str, float]) -> List[Tuple[str, str, float]]:
        """
        Identify features that suggest elevated ASD risk.

        Args:
            features: Dictionary of feature names to values

        Returns:
            List of (feature_name, reason, value) tuples for concerning features
        """
        risk_indicators = []

        # Check each feature against ASD patterns
        if features.get('avg_dependency_depth', 0) < 2.0:
            risk_indicators.append((
                'avg_dependency_depth',
                'Very low syntactic complexity',
                features['avg_dependency_depth']
            ))

        if features.get('grammatical_error_rate', 0) > 0.25:
            risk_indicators.append((
                'grammatical_error_rate',
                'High rate of grammatical errors',
                features['grammatical_error_rate']
            ))

        if features.get('semantic_coherence', 1.0) < 0.4:
            risk_indicators.append((
                'semantic_coherence',
                'Low conversational coherence (topic shifts)',
                features['semantic_coherence']
            ))

        if features.get('lexical_diversity_semantic', 1.0) < 0.5:
            risk_indicators.append((
                'lexical_diversity_semantic',
                'Highly repetitive language',
                features['lexical_diversity_semantic']
            ))

        if features.get('subordination_index', 0) < 0.2:
            risk_indicators.append((
                'subordination_index',
                'Very limited use of complex sentences',
                features['subordination_index']
            ))

        return risk_indicators

    def generate_clinical_summary(self, features: Dict[str, float]) -> str:
        """
        Generate a concise clinical summary.

        Args:
            features: Dictionary of feature names to values

        Returns:
            Clinical summary string
        """
        risk_indicators = self.get_asd_risk_indicators(features)

        summary = ["CLINICAL SUMMARY:", ""]

        if risk_indicators:
            summary.append(f"Found {len(risk_indicators)} features suggesting possible ASD patterns:")
            summary.append("")
            for feature, reason, value in risk_indicators:
                summary.append(f"  - {feature}: {value:.3f}")
                summary.append(f"    {reason}")
                summary.append("")
        else:
            summary.append("No strong ASD risk indicators detected in syntactic/semantic features.")
            summary.append("Features appear within typical developmental range.")

        summary.append("\nNOTE: This is a preliminary analysis. Clinical diagnosis requires")
        summary.append("comprehensive evaluation by qualified professionals.")

        return "\n".join(summary)


__all__ = ['ClinicalInterpreter', 'FeatureInterpretation']
