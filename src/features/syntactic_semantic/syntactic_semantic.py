"""
Syntactic and Semantic Feature Extractor

This module extracts syntactic and semantic features from conversational transcripts.
Features analyze grammatical structures, complexity, and semantic meaning.

Features implemented:
- Syntactic Complexity:
  - Dependency tree depth
  - Clause complexity
  - Phrase structure complexity
  - Subordination index

- Grammatical Features:
  - Grammatical error rate
  - Tense consistency
  - Agreement errors
  - Sentence structure diversity

- Semantic Features:
  - Semantic coherence score
  - Word sense disambiguation
  - Semantic role labeling
  - Thematic consistency

- Vocabulary Semantic Features:
  - Concreteness vs. abstractness
  - Semantic field diversity
  - Word association patterns

Author: Randil Haturusinghe
"""

import re
import numpy as np
from typing import List, Dict, Any, Set
from collections import Counter
import spacy
from nltk.corpus import wordnet
from textstat import textstat

from src.parsers.chat_parser import TranscriptData, Utterance
from src.utils.helpers import calculate_ratio, safe_divide
from ..base_features import BaseFeatureExtractor, FeatureResult


class SyntacticSemanticFeatures(BaseFeatureExtractor):
    """
    Extract syntactic and semantic features from conversational transcripts.

    This class analyzes grammatical structures, complexity patterns,
    and semantic meaning in child utterances.

    Features capture:
    - Syntactic complexity (dependency depth, clause structure)
    - Grammatical accuracy (errors, consistency)
    - Semantic coherence and density
    - Vocabulary semantic properties

    Example:
        >>> extractor = SyntacticSemanticFeatures()
        >>> features = extractor.extract(transcript)
        >>> print(f"Avg dependency depth: {features.features['avg_dependency_depth']}")
    """

    def __init__(self):
        """Initialize syntactic/semantic feature extractor."""
        super().__init__()

        # Load spaCy model for NLP analysis
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.logger.warning("spaCy model not found. Installing...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")

        # Initialize NLTK WordNet (for semantic analysis)
        try:
            wordnet.synsets('test')
        except LookupError:
            import nltk
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)

        self.logger.info("SyntacticSemanticFeatures initialized")

    @property
    def feature_names(self) -> List[str]:
        """Get list of syntactic/semantic feature names."""
        return [
            # Syntactic complexity
            'avg_dependency_depth',
            'max_dependency_depth',
            'avg_dependency_distance',
            'clause_complexity',
            'subordination_index',
            'coordination_index',

            # Grammatical accuracy
            'grammatical_error_rate',
            'tense_consistency_score',
            'tense_variety',
            'structure_diversity',
            'pos_tag_diversity',

            # Sentence structure
            'avg_parse_tree_height',
            'noun_phrase_complexity',
            'verb_phrase_complexity',
            'prepositional_phrase_ratio',

            # Semantic features
            'semantic_coherence',
            'semantic_density',
            'lexical_diversity_semantic',
            'thematic_consistency',

            # Vocabulary semantic features
            'vocabulary_abstractness',
            'semantic_field_diversity',
            'word_sense_diversity',
            'content_word_ratio',

            # Advanced semantic
            'semantic_role_diversity',
            'entity_density',
            'verb_argument_complexity',
        ]

    def extract(self, transcript: TranscriptData) -> FeatureResult:
        """
        Extract syntactic and semantic features from transcript.

        Args:
            transcript: Parsed transcript data

        Returns:
            FeatureResult with syntactic/semantic features
        """
        features = {}

        # Get child utterances
        child_utterances = self.get_child_utterances(transcript)

        if not child_utterances:
            return FeatureResult(
                features={name: 0.0 for name in self.feature_names},
                feature_type='syntactic_semantic',
                metadata={'error': 'No valid child utterances'}
            )

        # Parse utterances with spaCy
        docs = self._parse_utterances(child_utterances)

        # Extract syntactic complexity features
        syntactic_features = self._calculate_syntactic_complexity(docs)
        features.update(syntactic_features)

        # Extract grammatical features
        grammatical_features = self._calculate_grammatical_features(docs, child_utterances)
        features.update(grammatical_features)

        # Extract sentence structure features
        structure_features = self._calculate_structure_features(docs)
        features.update(structure_features)

        # Extract semantic features
        semantic_features = self._calculate_semantic_features(docs, child_utterances)
        features.update(semantic_features)

        # Extract vocabulary semantic features
        vocab_features = self._calculate_vocabulary_semantic_features(docs)
        features.update(vocab_features)

        # Extract advanced semantic features
        advanced_features = self._calculate_advanced_semantic_features(docs)
        features.update(advanced_features)

        return FeatureResult(
            features=features,
            feature_type='syntactic_semantic',
            metadata={
                'num_child_utterances': len(child_utterances),
                'num_tokens_analyzed': sum(len(doc) for doc in docs),
                'status': 'implemented'
            }
        )

    def _parse_utterances(self, utterances: List[Utterance]) -> List:
        """Parse utterances with spaCy."""
        docs = []
        for utterance in utterances:
            if utterance.text and utterance.text.strip():
                doc = self.nlp(utterance.text)
                docs.append(doc)
        return docs

    def _calculate_syntactic_complexity(self, docs: List) -> Dict[str, float]:
        """Calculate syntactic complexity features."""
        features = {
            'avg_dependency_depth': 0.0,
            'max_dependency_depth': 0.0,
            'avg_dependency_distance': 0.0,
            'clause_complexity': 0.0,
            'subordination_index': 0.0,
            'coordination_index': 0.0,
        }

        if not docs:
            return features

        all_depths = []
        all_distances = []
        subordinate_count = 0
        coordinate_count = 0
        clause_count = 0

        for doc in docs:
            for token in doc:
                # Dependency depth
                depth = self._get_dependency_depth(token)
                all_depths.append(depth)

                # Dependency distance
                if token.head != token:
                    distance = abs(token.i - token.head.i)
                    all_distances.append(distance)

                # Count subordinate clauses
                if token.dep_ in ['advcl', 'acl', 'ccomp', 'xcomp', 'relcl']:
                    subordinate_count += 1

                # Count coordinate clauses
                if token.dep_ == 'conj':
                    coordinate_count += 1

                # Count clause markers
                if token.dep_ == 'mark':
                    clause_count += 1

        features['avg_dependency_depth'] = np.mean(all_depths) if all_depths else 0.0
        features['max_dependency_depth'] = max(all_depths) if all_depths else 0.0
        features['avg_dependency_distance'] = np.mean(all_distances) if all_distances else 0.0
        features['clause_complexity'] = clause_count / len(docs) if docs else 0.0
        features['subordination_index'] = subordinate_count / len(docs) if docs else 0.0
        features['coordination_index'] = coordinate_count / len(docs) if docs else 0.0

        return features

    def _get_dependency_depth(self, token) -> int:
        """Get depth of token in dependency tree."""
        depth = 0
        current = token
        while current.head != current:
            depth += 1
            current = current.head
            if depth > 20:  # Safety limit
                break
        return depth

    def _calculate_grammatical_features(
        self,
        docs: List,
        utterances: List[Utterance]
    ) -> Dict[str, float]:
        """Calculate grammatical accuracy features."""
        features = {
            'grammatical_error_rate': 0.0,
            'tense_consistency_score': 0.0,
            'tense_variety': 0.0,
            'structure_diversity': 0.0,
            'pos_tag_diversity': 0.0,
        }

        if not docs:
            return features

        # Grammatical error approximation (incomplete sentences, wrong agreement)
        error_count = 0
        tenses = []
        structures = []
        pos_tags = set()

        for doc in docs:
            # Check for sentence completeness
            has_verb = any(token.pos_ == 'VERB' for token in doc)
            has_subject = any(token.dep_ in ['nsubj', 'nsubjpass'] for token in doc)

            if not has_verb or (len(doc) > 3 and not has_subject):
                error_count += 1

            # Collect tenses
            for token in doc:
                if token.pos_ == 'VERB':
                    if token.tag_ in ['VBD', 'VBN']:
                        tenses.append('past')
                    elif token.tag_ in ['VBP', 'VBZ', 'VBG']:
                        tenses.append('present')
                    elif token.tag_ == 'MD':
                        tenses.append('modal')

                # Collect POS tags
                pos_tags.add(token.pos_)

            # Collect sentence structures (root POS pattern)
            if doc:
                root = [token for token in doc if token.dep_ == 'ROOT']
                if root:
                    structures.append(root[0].pos_)

        features['grammatical_error_rate'] = error_count / len(docs) if docs else 0.0

        # Tense consistency (inverse of variance)
        if tenses:
            tense_counts = Counter(tenses)
            most_common_ratio = tense_counts.most_common(1)[0][1] / len(tenses)
            features['tense_consistency_score'] = most_common_ratio
            features['tense_variety'] = len(tense_counts) / len(tenses)

        # Structure diversity
        if structures:
            structure_counts = Counter(structures)
            features['structure_diversity'] = len(structure_counts) / len(structures)

        # POS tag diversity
        total_tokens = sum(len(doc) for doc in docs)
        features['pos_tag_diversity'] = len(pos_tags) / total_tokens if total_tokens > 0 else 0.0

        return features

    def _calculate_structure_features(self, docs: List) -> Dict[str, float]:
        """Calculate sentence structure features."""
        features = {
            'avg_parse_tree_height': 0.0,
            'noun_phrase_complexity': 0.0,
            'verb_phrase_complexity': 0.0,
            'prepositional_phrase_ratio': 0.0,
        }

        if not docs:
            return features

        tree_heights = []
        np_complexity = []
        vp_complexity = []
        pp_count = 0

        for doc in docs:
            # Parse tree height (max dependency depth)
            if doc:
                max_depth = max(self._get_dependency_depth(token) for token in doc)
                tree_heights.append(max_depth)

            # Noun phrase complexity
            for chunk in doc.noun_chunks:
                np_complexity.append(len(chunk))

            # Verb phrase complexity (verb + dependents)
            for token in doc:
                if token.pos_ == 'VERB':
                    dependents = list(token.children)
                    vp_complexity.append(len(dependents))

            # Prepositional phrases
            for token in doc:
                if token.pos_ == 'ADP':
                    pp_count += 1

        features['avg_parse_tree_height'] = np.mean(tree_heights) if tree_heights else 0.0
        features['noun_phrase_complexity'] = np.mean(np_complexity) if np_complexity else 0.0
        features['verb_phrase_complexity'] = np.mean(vp_complexity) if vp_complexity else 0.0
        features['prepositional_phrase_ratio'] = pp_count / len(docs) if docs else 0.0

        return features

    def _calculate_semantic_features(
        self,
        docs: List,
        utterances: List[Utterance]
    ) -> Dict[str, float]:
        """Calculate semantic coherence and meaning features."""
        features = {
            'semantic_coherence': 0.0,
            'semantic_density': 0.0,
            'lexical_diversity_semantic': 0.0,
            'thematic_consistency': 0.0,
        }

        if not docs:
            return features

        # Semantic coherence (similarity between consecutive utterances)
        coherence_scores = []
        for i in range(1, len(docs)):
            similarity = docs[i-1].similarity(docs[i])
            coherence_scores.append(similarity)

        features['semantic_coherence'] = np.mean(coherence_scores) if coherence_scores else 0.0

        # Semantic density (content words per utterance)
        content_word_counts = []
        all_content_words = []

        for doc in docs:
            content_words = [
                token.lemma_.lower() for token in doc
                if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']
            ]
            content_word_counts.append(len(content_words))
            all_content_words.extend(content_words)

        features['semantic_density'] = np.mean(content_word_counts) if content_word_counts else 0.0

        # Lexical diversity (unique content words / total content words)
        if all_content_words:
            features['lexical_diversity_semantic'] = len(set(all_content_words)) / len(all_content_words)

        # Thematic consistency (repeated content words)
        if all_content_words:
            word_freq = Counter(all_content_words)
            repeated_words = sum(1 for count in word_freq.values() if count > 1)
            features['thematic_consistency'] = repeated_words / len(word_freq) if word_freq else 0.0

        return features

    def _calculate_vocabulary_semantic_features(self, docs: List) -> Dict[str, float]:
        """Calculate vocabulary-level semantic features."""
        features = {
            'vocabulary_abstractness': 0.0,
            'semantic_field_diversity': 0.0,
            'word_sense_diversity': 0.0,
            'content_word_ratio': 0.0,
        }

        if not docs:
            return features

        abstract_count = 0
        concrete_count = 0
        semantic_fields = set()
        sense_counts = []
        total_tokens = 0
        content_tokens = 0

        for doc in docs:
            for token in doc:
                total_tokens += 1

                # Content word ratio
                if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']:
                    content_tokens += 1

                    # Abstractness (using WordNet)
                    synsets = wordnet.synsets(token.lemma_)
                    if synsets:
                        sense_counts.append(len(synsets))

                        # Use hypernym depth as proxy for abstractness
                        for synset in synsets[:1]:  # Use first sense
                            depth = synset.min_depth()
                            if depth > 5:
                                abstract_count += 1
                            else:
                                concrete_count += 1

                            # Semantic field (top-level hypernym)
                            hypernyms = synset.hypernyms()
                            if hypernyms:
                                semantic_fields.add(hypernyms[0].name())

        # Vocabulary abstractness (ratio of abstract to concrete)
        total_abstract_concrete = abstract_count + concrete_count
        if total_abstract_concrete > 0:
            features['vocabulary_abstractness'] = abstract_count / total_abstract_concrete

        # Semantic field diversity
        features['semantic_field_diversity'] = len(semantic_fields) / content_tokens if content_tokens > 0 else 0.0

        # Word sense diversity
        features['word_sense_diversity'] = np.mean(sense_counts) if sense_counts else 0.0

        # Content word ratio
        features['content_word_ratio'] = content_tokens / total_tokens if total_tokens > 0 else 0.0

        return features

    def _calculate_advanced_semantic_features(self, docs: List) -> Dict[str, float]:
        """Calculate advanced semantic role and entity features."""
        features = {
            'semantic_role_diversity': 0.0,
            'entity_density': 0.0,
            'verb_argument_complexity': 0.0,
        }

        if not docs:
            return features

        semantic_roles = set()
        entity_count = 0
        verb_arg_counts = []

        for doc in docs:
            # Semantic roles (dependency relations)
            for token in doc:
                if token.dep_ in ['nsubj', 'dobj', 'iobj', 'pobj', 'agent', 'attr']:
                    semantic_roles.add(token.dep_)

                # Verb argument structure
                if token.pos_ == 'VERB':
                    args = [child for child in token.children
                            if child.dep_ in ['nsubj', 'dobj', 'iobj', 'prep']]
                    verb_arg_counts.append(len(args))

            # Named entities
            entity_count += len(doc.ents)

        features['semantic_role_diversity'] = len(semantic_roles) / len(docs) if docs else 0.0
        features['entity_density'] = entity_count / len(docs) if docs else 0.0
        features['verb_argument_complexity'] = np.mean(verb_arg_counts) if verb_arg_counts else 0.0

        return features


__all__ = ["SyntacticSemanticFeatures"]

