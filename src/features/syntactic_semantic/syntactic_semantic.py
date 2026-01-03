"""
Syntactic and Semantic Feature Extractor (Sections 3.3.5 & 3.3.6)

This module extracts syntactic complexity, grammatical accuracy, and semantic
coherence features from conversational transcripts for ASD detection.

Features implemented:
- Syntactic Complexity (Section 3.3.5):
  - Dependency tree depth (avg/max)
  - Clause complexity and subordination index
  - Coordination index
  
- Grammatical Accuracy (Section 3.3.5):
  - Grammatical error rate (missing subject/verb heuristic)
  - Tense consistency score
  - POS tag diversity (TTR applied to POS tags)
  
- Semantic Features (Section 3.3.6):
  - Semantic coherence (cosine similarity between adjacent utterances)
  - Lexical diversity (TTR of content words)
  - Vocabulary abstractness (WordNet hypernym depth)

References:
- Methodology Sections 3.3.5 (Syntactic) and 3.3.6 (Semantic)
- Children with ASD often show reduced syntactic complexity and semantic coherence

Author: Randil Haturusinghe
"""

import sys
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import Counter

import numpy as np

from src.parsers.chat_parser import TranscriptData, Utterance
from src.utils.helpers import calculate_ratio, safe_divide
from src.utils.logger import get_logger
from ..base_features import BaseFeatureExtractor, FeatureResult

logger = get_logger(__name__)


class SyntacticSemanticFeatures(BaseFeatureExtractor):
    """
    Extract syntactic and semantic features from transcripts (Sections 3.3.5 & 3.3.6).
    
    Features capture:
    - Syntactic complexity (dependency depth, clause structure)
    - Grammatical accuracy (errors, tense consistency)
    - Semantic coherence and density
    - Vocabulary semantic properties (abstractness, diversity)
    
    Key ASD markers:
    - Reduced syntactic complexity (shorter dependency trees)
    - Lower semantic coherence between utterances
    - Less varied vocabulary with fewer abstract concepts
    
    Example:
        >>> extractor = SyntacticSemanticFeatures()
        >>> features = extractor.extract(transcript)
        >>> print(f"Semantic coherence: {features.features['semantic_coherence']}")
    """
    
    # Thresholds for abstractness classification
    ABSTRACT_DEPTH_THRESHOLD = 5  # WordNet hypernym depth > 5 = abstract
    
    def __init__(self):
        """
        Initialize syntactic/semantic feature extractor.
        
        Lazy loads spaCy and NLTK resources to avoid slow API startup times.
        """
        super().__init__()
        
        # Lazy load spaCy
        self._nlp = None
        self._spacy_loaded = False
        self._wordnet_loaded = False
        self._parsing_errors = 0
        
        self.logger.info("SyntacticSemanticFeatures initialized (lazy loading enabled)")
    
    def _ensure_spacy_loaded(self) -> None:
        """Lazy load spaCy model on first use."""
        if self._spacy_loaded:
            return
            
        try:
            import spacy
            try:
                self._nlp = spacy.load("en_core_web_md")
                self.logger.debug("spaCy model 'en_core_web_md' loaded successfully")
            except OSError:
                self.logger.warning("spaCy model not found. Installing...")
                import subprocess
                subprocess.run(
                    [sys.executable, "-m", "spacy", "download", "en_core_web_md"],
                    check=True
                )
                self._nlp = spacy.load("en_core_web_md")
            self._spacy_loaded = True
        except Exception as e:
            self.logger.error(f"Failed to load spaCy: {e}")
            raise
    
    def _ensure_wordnet_loaded(self) -> None:
        """Lazy load NLTK WordNet on first use."""
        if self._wordnet_loaded:
            return
            
        try:
            from nltk.corpus import wordnet
            # Test if wordnet is available
            wordnet.synsets('test')
            self._wordnet_loaded = True
            self.logger.debug("NLTK WordNet loaded successfully")
        except LookupError:
            self.logger.info("Downloading NLTK WordNet data...")
            import nltk
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            self._wordnet_loaded = True
    
    @property
    def feature_names(self) -> List[str]:
        """Get list of syntactic/semantic feature names."""
        return [
            # Syntactic complexity (Section 3.3.5)
            'avg_dependency_depth',
            'max_dependency_depth',
            'avg_dependency_distance',
            'clause_complexity',
            'subordination_index',
            'coordination_index',
            
            # Grammatical accuracy (Section 3.3.5)
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
            
            # Semantic features (Section 3.3.6)
            'semantic_coherence',
            'semantic_density',
            'lexical_diversity_semantic',
            'thematic_consistency',
            
            # Vocabulary semantic features (Section 3.3.6)
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
        
        Implements feature extraction per Methodology Sections 3.3.5 and 3.3.6.
        
        Args:
            transcript: Parsed transcript data
            
        Returns:
            FeatureResult with syntactic/semantic features and metadata
        """
        features = {}
        self._parsing_errors = 0
        
        # Get child utterances
        child_utterances = self.get_child_utterances(transcript)
        
        self.logger.debug(
            f"Extracting syntactic/semantic features from "
            f"{len(child_utterances)} child utterances"
        )
        
        # Handle empty transcripts gracefully
        if not child_utterances:
            self.logger.warning("No valid child utterances found in transcript")
            return FeatureResult(
                features={name: 0.0 for name in self.feature_names},
                feature_type='syntactic_semantic',
                metadata={
                    'error': 'No valid child utterances',
                    'num_child_utterances': 0,
                    'num_sentences_parsed': 0,
                    'parsing_errors': 0,
                    'has_vectors': False
                }
            )
        
        # Ensure spaCy is loaded
        self._ensure_spacy_loaded()
        
        # Parse utterances with spaCy
        docs = self._parse_utterances(child_utterances)
        
        if not docs:
            self.logger.warning("No documents successfully parsed")
            return FeatureResult(
                features={name: 0.0 for name in self.feature_names},
                feature_type='syntactic_semantic',
                metadata={
                    'error': 'No documents parsed successfully',
                    'num_child_utterances': len(child_utterances),
                    'num_sentences_parsed': 0,
                    'parsing_errors': self._parsing_errors,
                    'has_vectors': False
                }
            )
        
        # Check if spaCy model has word vectors
        has_vectors = self._nlp.vocab.vectors.shape[0] > 0
        
        self.logger.debug(f"Successfully parsed {len(docs)} documents")
        
        # Extract syntactic complexity features (Section 3.3.5)
        syntactic_features = self._calculate_syntactic_complexity(docs)
        features.update(syntactic_features)
        
        # Extract grammatical features (Section 3.3.5)
        grammatical_features = self._calculate_grammatical_features(docs, child_utterances)
        features.update(grammatical_features)
        
        # Extract sentence structure features
        structure_features = self._calculate_structure_features(docs)
        features.update(structure_features)
        
        # Extract semantic features (Section 3.3.6)
        semantic_features = self._calculate_semantic_features(docs, child_utterances, has_vectors)
        features.update(semantic_features)
        
        # Extract vocabulary semantic features (Section 3.3.6)
        vocab_features = self._calculate_vocabulary_semantic_features(docs)
        features.update(vocab_features)
        
        # Extract advanced semantic features
        advanced_features = self._calculate_advanced_semantic_features(docs)
        features.update(advanced_features)
        
        self.logger.debug(f"Extracted {len(features)} syntactic/semantic features")
        
        return FeatureResult(
            features=features,
            feature_type='syntactic_semantic',
            metadata={
                'num_child_utterances': len(child_utterances),
                'num_sentences_parsed': len(docs),
                'num_tokens_analyzed': sum(len(doc) for doc in docs),
                'parsing_errors': self._parsing_errors,
                'has_vectors': has_vectors,
                'status': 'success'
            }
        )
    
    def _parse_utterances(self, utterances: List[Utterance]) -> List:
        """
        Parse utterances with spaCy.
        
        Args:
            utterances: List of Utterance objects
            
        Returns:
            List of spaCy Doc objects
        """
        docs = []
        
        for utterance in utterances:
            try:
                if utterance.text and utterance.text.strip():
                    doc = self._nlp(utterance.text)
                    docs.append(doc)
            except Exception as e:
                self._parsing_errors += 1
                self.logger.debug(f"Error parsing utterance: {e}")
                continue
                
        return docs
    
    def _get_dependency_depth(self, token) -> int:
        """
        Get depth of token in dependency tree.
        
        Args:
            token: spaCy Token object
            
        Returns:
            Depth in the dependency tree (0 for ROOT)
        """
        depth = 0
        current = token
        
        # Safety limit to prevent infinite loops
        while current.head != current and depth < 50:
            depth += 1
            current = current.head
            
        return depth
    
    def _calculate_syntactic_complexity(self, docs: List) -> Dict[str, float]:
        """
        Calculate syntactic complexity features (Section 3.3.5).
        
        Key ASD markers: Reduced dependency depth and subordination
        suggest simpler syntactic structures.
        
        Args:
            docs: List of spaCy Doc objects
            
        Returns:
            Dictionary of syntactic complexity features
        """
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
                
                # Dependency distance (absolute distance between token and head)
                if token.head != token:
                    distance = abs(token.i - token.head.i)
                    all_distances.append(distance)
                
                # Count subordinate clauses (adverbial, adjectival, complement)
                if token.dep_ in ['advcl', 'acl', 'ccomp', 'xcomp', 'relcl']:
                    subordinate_count += 1
                
                # Count coordinate clauses
                if token.dep_ == 'conj':
                    coordinate_count += 1
                
                # Count clause markers (subordinating conjunctions, etc.)
                if token.dep_ == 'mark':
                    clause_count += 1
        
        num_docs = len(docs)
        
        features['avg_dependency_depth'] = float(np.mean(all_depths)) if all_depths else 0.0
        features['max_dependency_depth'] = float(max(all_depths)) if all_depths else 0.0
        features['avg_dependency_distance'] = float(np.mean(all_distances)) if all_distances else 0.0
        features['clause_complexity'] = safe_divide(clause_count, num_docs)
        features['subordination_index'] = safe_divide(subordinate_count, num_docs)
        features['coordination_index'] = safe_divide(coordinate_count, num_docs)
        
        return features
    
    def _calculate_grammatical_features(
        self,
        docs: List,
        utterances: List[Utterance]
    ) -> Dict[str, float]:
        """
        Calculate grammatical accuracy features (Section 3.3.5).
        
        Uses heuristics to detect potential grammatical issues:
        - Missing subjects in longer sentences
        - Missing verbs
        - Inconsistent tense usage
        
        Args:
            docs: List of spaCy Doc objects
            utterances: Original utterance objects
            
        Returns:
            Dictionary of grammatical features
        """
        features = {
            'grammatical_error_rate': 0.0,
            'tense_consistency_score': 0.0,
            'tense_variety': 0.0,
            'structure_diversity': 0.0,
            'pos_tag_diversity': 0.0,
        }
        
        if not docs:
            return features
        
        error_count = 0
        tenses = []
        structures = []
        pos_tags: Set[str] = set()
        
        for doc in docs:
            # Check for sentence completeness (heuristic for grammatical errors)
            has_verb = any(token.pos_ == 'VERB' for token in doc)
            has_subject = any(token.dep_ in ['nsubj', 'nsubjpass'] for token in doc)
            
            # Flag as error if: no verb, or long sentence without subject
            if not has_verb or (len(doc) > 3 and not has_subject):
                error_count += 1
            
            # Collect verb tenses
            for token in doc:
                if token.pos_ == 'VERB':
                    if token.tag_ in ['VBD', 'VBN']:
                        tenses.append('past')
                    elif token.tag_ in ['VBP', 'VBZ', 'VBG']:
                        tenses.append('present')
                    elif token.tag_ == 'MD':
                        tenses.append('modal')
                
                # Collect all POS tags
                pos_tags.add(token.pos_)
            
            # Collect sentence structures (root POS pattern)
            roots = [token for token in doc if token.dep_ == 'ROOT']
            if roots:
                structures.append(roots[0].pos_)
        
        num_docs = len(docs)
        
        # Grammatical error rate
        features['grammatical_error_rate'] = safe_divide(error_count, num_docs)
        
        # Tense consistency (ratio of most common tense)
        if tenses:
            tense_counts = Counter(tenses)
            most_common_count = tense_counts.most_common(1)[0][1]
            features['tense_consistency_score'] = safe_divide(most_common_count, len(tenses))
            features['tense_variety'] = safe_divide(len(tense_counts), len(tenses))
        
        # Structure diversity
        if structures:
            structure_counts = Counter(structures)
            features['structure_diversity'] = safe_divide(len(structure_counts), len(structures))
        
        # POS tag diversity (TTR for POS tags)
        total_tokens = sum(len(doc) for doc in docs)
        features['pos_tag_diversity'] = safe_divide(len(pos_tags), total_tokens)
        
        return features
    
    def _calculate_structure_features(self, docs: List) -> Dict[str, float]:
        """
        Calculate sentence structure features.
        
        Measures phrase complexity and prepositional phrase usage.
        
        Args:
            docs: List of spaCy Doc objects
            
        Returns:
            Dictionary of structure features
        """
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
            # Parse tree height (max dependency depth in document)
            if doc:
                max_depth = max(self._get_dependency_depth(token) for token in doc)
                tree_heights.append(max_depth)
            
            # Noun phrase complexity (tokens per noun chunk)
            for chunk in doc.noun_chunks:
                np_complexity.append(len(chunk))
            
            # Verb phrase complexity (number of dependents per verb)
            for token in doc:
                if token.pos_ == 'VERB':
                    dependents = list(token.children)
                    vp_complexity.append(len(dependents))
            
            # Count prepositional phrases (adpositions)
            for token in doc:
                if token.pos_ == 'ADP':
                    pp_count += 1
        
        features['avg_parse_tree_height'] = float(np.mean(tree_heights)) if tree_heights else 0.0
        features['noun_phrase_complexity'] = float(np.mean(np_complexity)) if np_complexity else 0.0
        features['verb_phrase_complexity'] = float(np.mean(vp_complexity)) if vp_complexity else 0.0
        features['prepositional_phrase_ratio'] = safe_divide(pp_count, len(docs))
        
        return features
    
    def _calculate_semantic_features(
        self,
        docs: List,
        utterances: List[Utterance],
        has_vectors: bool
    ) -> Dict[str, float]:
        """
        Calculate semantic coherence and meaning features (Section 3.3.6).
        
        Key ASD marker: Lower semantic coherence between adjacent utterances
        may indicate difficulty maintaining topic continuity.
        
        Args:
            docs: List of spaCy Doc objects
            utterances: Original utterance objects
            has_vectors: Whether spaCy model has word vectors
            
        Returns:
            Dictionary of semantic features
        """
        features = {
            'semantic_coherence': 0.0,
            'semantic_density': 0.0,
            'lexical_diversity_semantic': 0.0,
            'thematic_consistency': 0.0,
        }
        
        if not docs:
            return features
        
        # Semantic coherence (cosine similarity between consecutive utterances)
        coherence_scores = []
        
        if has_vectors and len(docs) > 1:
            for i in range(1, len(docs)):
                try:
                    # spaCy's similarity uses word vectors
                    similarity = docs[i-1].similarity(docs[i])
                    # Filter out NaN values
                    if not np.isnan(similarity):
                        coherence_scores.append(similarity)
                except Exception:
                    # Handle cases where vectors are missing
                    pass
        
        features['semantic_coherence'] = float(np.mean(coherence_scores)) if coherence_scores else 0.0
        
        # Semantic density (content words per utterance)
        content_word_counts = []
        all_content_words = []
        
        for doc in docs:
            content_words = [
                token.lemma_.lower() for token in doc
                if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] and not token.is_stop
            ]
            content_word_counts.append(len(content_words))
            all_content_words.extend(content_words)
        
        features['semantic_density'] = float(np.mean(content_word_counts)) if content_word_counts else 0.0
        
        # Lexical diversity (TTR for content words)
        if all_content_words:
            features['lexical_diversity_semantic'] = calculate_ratio(
                len(set(all_content_words)), len(all_content_words)
            )
        
        # Thematic consistency (proportion of repeated content words)
        if all_content_words:
            word_freq = Counter(all_content_words)
            repeated_words = sum(1 for count in word_freq.values() if count > 1)
            features['thematic_consistency'] = calculate_ratio(repeated_words, len(word_freq))
        
        return features
    
    def _calculate_vocabulary_semantic_features(self, docs: List) -> Dict[str, float]:
        """
        Calculate vocabulary-level semantic features (Section 3.3.6).
        
        Uses WordNet hypernym depth as a proxy for abstractness.
        Words with deeper hypernym trees tend to be more concrete.
        
        Args:
            docs: List of spaCy Doc objects
            
        Returns:
            Dictionary of vocabulary semantic features
        """
        features = {
            'vocabulary_abstractness': 0.0,
            'semantic_field_diversity': 0.0,
            'word_sense_diversity': 0.0,
            'content_word_ratio': 0.0,
        }
        
        if not docs:
            return features
        
        # Lazy load WordNet
        self._ensure_wordnet_loaded()
        from nltk.corpus import wordnet
        
        abstract_count = 0
        concrete_count = 0
        semantic_fields: Set[str] = set()
        sense_counts = []
        total_tokens = 0
        content_tokens = 0
        
        for doc in docs:
            for token in doc:
                total_tokens += 1
                
                # Content word ratio
                if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] and not token.is_stop:
                    content_tokens += 1
                    
                    # Abstractness using WordNet
                    try:
                        synsets = wordnet.synsets(token.lemma_)
                        if synsets:
                            sense_counts.append(len(synsets))
                            
                            # Use first sense for depth analysis
                            first_synset = synsets[0]
                            depth = first_synset.min_depth()
                            
                            if depth > self.ABSTRACT_DEPTH_THRESHOLD:
                                abstract_count += 1
                            else:
                                concrete_count += 1
                            
                            # Semantic field (top-level hypernym)
                            hypernyms = first_synset.hypernyms()
                            if hypernyms:
                                semantic_fields.add(hypernyms[0].name())
                    except Exception:
                        # WordNet lookup failed for this token
                        pass
        
        # Vocabulary abstractness ratio
        total_classified = abstract_count + concrete_count
        if total_classified > 0:
            features['vocabulary_abstractness'] = calculate_ratio(abstract_count, total_classified)
        
        # Semantic field diversity
        if content_tokens > 0:
            features['semantic_field_diversity'] = calculate_ratio(len(semantic_fields), content_tokens)
        
        # Word sense diversity (average number of senses per word)
        features['word_sense_diversity'] = float(np.mean(sense_counts)) if sense_counts else 0.0
        
        # Content word ratio
        features['content_word_ratio'] = calculate_ratio(content_tokens, total_tokens)
        
        return features
    
    def _calculate_advanced_semantic_features(self, docs: List) -> Dict[str, float]:
        """
        Calculate advanced semantic role and entity features.
        
        Analyzes verb argument structure and named entity density.
        
        Args:
            docs: List of spaCy Doc objects
            
        Returns:
            Dictionary of advanced semantic features
        """
        features = {
            'semantic_role_diversity': 0.0,
            'entity_density': 0.0,
            'verb_argument_complexity': 0.0,
        }
        
        if not docs:
            return features
        
        semantic_roles: Set[str] = set()
        entity_count = 0
        verb_arg_counts = []
        
        for doc in docs:
            # Semantic roles (dependency relations that indicate arguments)
            for token in doc:
                if token.dep_ in ['nsubj', 'dobj', 'iobj', 'pobj', 'agent', 'attr']:
                    semantic_roles.add(token.dep_)
                
                # Verb argument structure
                if token.pos_ == 'VERB':
                    args = [
                        child for child in token.children
                        if child.dep_ in ['nsubj', 'dobj', 'iobj', 'prep', 'ccomp', 'xcomp']
                    ]
                    verb_arg_counts.append(len(args))
            
            # Named entities
            entity_count += len(doc.ents)
        
        num_docs = len(docs)
        
        features['semantic_role_diversity'] = safe_divide(len(semantic_roles), num_docs)
        features['entity_density'] = safe_divide(entity_count, num_docs)
        features['verb_argument_complexity'] = float(np.mean(verb_arg_counts)) if verb_arg_counts else 0.0
        
        return features


__all__ = ["SyntacticSemanticFeatures"]
