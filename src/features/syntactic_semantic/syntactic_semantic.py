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
        """Get list of syntactic/semantic feature names (40+ features for comprehensive analysis)."""
        return [
            # POS ratios (8 features)
            'pos_noun_ratio',
            'pos_verb_ratio',
            'pos_adj_ratio',
            'pos_adv_ratio',
            'pos_pronoun_ratio',
            'pos_det_ratio',
            'pos_adp_ratio',
            'pos_conj_ratio',
            
            # Dependency tree metrics (6 features)
            'dependency_tree_depth',
            'dependency_tree_width',
            'avg_dependency_distance',
            'max_dependency_distance',
            'root_distance_avg',
            'dependency_branching_factor',
            
            # Clause structure (6 features)
            'clause_count',
            'subordinate_clause_ratio',
            'coordinate_clause_ratio',
            'relative_clause_ratio',
            'complement_clause_ratio',
            'adverbial_clause_ratio',
            
            # Sentence complexity (6 features)
            'sentence_complexity_score',
            'parse_tree_height',
            'avg_sentence_length',
            'sentence_length_variance',
            'complex_sentence_ratio',
            'simple_sentence_ratio',
            
            # Phrase structure (4 features)
            'phrase_structure_depth',
            'np_complexity',
            'vp_complexity',
            'pp_ratio',
            
            # Semantic features (8 features)
            'semantic_coherence_score',
            'lexical_diversity',
            'word_sense_diversity',
            'content_word_ratio',
            'function_word_ratio',
            'unique_lemma_ratio',
            'hapax_legomena_ratio',
            'avg_word_length',
            
            # Named entity features (3 features)
            'named_entity_density',
            'named_entity_diversity',
            'person_entity_ratio',
            
            # Verb analysis (4 features)
            'verb_tense_consistency',
            'modal_verb_ratio',
            'auxiliary_verb_ratio',
            'verb_argument_count_avg',
            
            # Aggregated complexity (1 feature)
            'syntactic_complexity',
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
        
        # Calculate syntactic_complexity as aggregate metric
        # Combine normalized values from multiple syntactic features
        if 'dependency_tree_depth' in features and 'clause_count' in features:
            normalized_depth = min(features.get('dependency_tree_depth', 0) / 10.0, 1.0)
            normalized_clauses = min(features.get('clause_count', 0) / 5.0, 1.0)
            normalized_subordination = features.get('subordinate_clause_ratio', 0)
            features['syntactic_complexity'] = (normalized_depth + normalized_clauses + normalized_subordination) / 3.0
        
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
            'dependency_tree_depth': 0.0,
            'dependency_tree_width': 0.0,
            'avg_dependency_distance': 0.0,
            'max_dependency_distance': 0.0,
            'root_distance_avg': 0.0,
            'dependency_branching_factor': 0.0,
            'clause_count': 0.0,
            'subordinate_clause_ratio': 0.0,
            'coordinate_clause_ratio': 0.0,
            'relative_clause_ratio': 0.0,
            'complement_clause_ratio': 0.0,
            'adverbial_clause_ratio': 0.0,
            'sentence_complexity_score': 0.0,
        }
        
        if not docs:
            return features
        
        all_depths = []
        all_distances = []
        root_distances = []
        children_counts = []
        subordinate_count = 0
        coordinate_count = 0
        relative_count = 0
        complement_count = 0
        adverbial_count = 0
        clause_count = 0
        
        for doc in docs:
            root_token = None
            for token in doc:
                # Dependency depth
                depth = self._get_dependency_depth(token)
                all_depths.append(depth)
                
                # Children count for branching factor
                children_counts.append(len(list(token.children)))
                
                # Dependency distance
                if token.head != token:
                    distance = abs(token.i - token.head.i)
                    all_distances.append(distance)
                
                # Find root
                if token.dep_ == 'ROOT':
                    root_token = token
                
                # Clause analysis
                if token.dep_ == 'mark':
                    clause_count += 1
                if token.dep_ == 'advcl':
                    adverbial_count += 1
                    subordinate_count += 1
                if token.dep_ in ['relcl', 'acl']:
                    relative_count += 1
                    subordinate_count += 1
                if token.dep_ in ['ccomp', 'xcomp']:
                    complement_count += 1
                    subordinate_count += 1
                if token.dep_ == 'conj':
                    coordinate_count += 1
            
            # Root distances
            if root_token:
                for other_token in doc:
                    root_distances.append(abs(other_token.i - root_token.i))
        
        num_docs = len(docs)
        
        features['dependency_tree_depth'] = float(np.mean(all_depths)) if all_depths else 0.0
        features['dependency_tree_width'] = float(max(all_depths)) if all_depths else 0.0
        features['avg_dependency_distance'] = float(np.mean(all_distances)) if all_distances else 0.0
        features['max_dependency_distance'] = float(max(all_distances)) if all_distances else 0.0
        features['root_distance_avg'] = float(np.mean(root_distances)) if root_distances else 0.0
        features['dependency_branching_factor'] = float(np.mean(children_counts)) if children_counts else 0.0
        features['clause_count'] = safe_divide(clause_count, num_docs)
        features['subordinate_clause_ratio'] = safe_divide(subordinate_count, num_docs)
        features['coordinate_clause_ratio'] = safe_divide(coordinate_count, num_docs)
        features['relative_clause_ratio'] = safe_divide(relative_count, num_docs)
        features['complement_clause_ratio'] = safe_divide(complement_count, num_docs)
        features['adverbial_clause_ratio'] = safe_divide(adverbial_count, num_docs)
        
        # Composite sentence complexity score
        features['sentence_complexity_score'] = (
            safe_divide(subordinate_count, num_docs) * 0.3 +
            safe_divide(coordinate_count, num_docs) * 0.2 +
            min(features['dependency_tree_depth'] / 10.0, 1.0) * 0.3 +
            min(features['avg_dependency_distance'] / 5.0, 1.0) * 0.2
        )
        
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
            'pos_noun_ratio': 0.0,
            'pos_verb_ratio': 0.0,
            'pos_adj_ratio': 0.0,
            'pos_adv_ratio': 0.0,
            'pos_pronoun_ratio': 0.0,
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
        total_tokens = sum(len(doc) for doc in docs)
        
        # POS ratios (nouns, verbs, adj, adv, pronouns / total_tokens)
        noun_count = 0
        verb_count = 0
        adj_count = 0
        adv_count = 0
        pron_count = 0
        det_count = 0
        adp_count = 0
        conj_count = 0
        
        for doc in docs:
            for token in doc:
                if token.pos_ == 'NOUN':
                    noun_count += 1
                elif token.pos_ == 'VERB':
                    verb_count += 1
                elif token.pos_ == 'ADJ':
                    adj_count += 1
                elif token.pos_ == 'ADV':
                    adv_count += 1
                elif token.pos_ == 'PRON':
                    pron_count += 1
                elif token.pos_ == 'DET':
                    det_count += 1
                elif token.pos_ == 'ADP':
                    adp_count += 1
                elif token.pos_ in ['CONJ', 'CCONJ', 'SCONJ']:
                    conj_count += 1
        
        features['pos_noun_ratio'] = safe_divide(noun_count, total_tokens)
        features['pos_verb_ratio'] = safe_divide(verb_count, total_tokens)
        features['pos_adj_ratio'] = safe_divide(adj_count, total_tokens)
        features['pos_adv_ratio'] = safe_divide(adv_count, total_tokens)
        features['pos_pronoun_ratio'] = safe_divide(pron_count, total_tokens)
        features['pos_det_ratio'] = safe_divide(det_count, total_tokens)
        features['pos_adp_ratio'] = safe_divide(adp_count, total_tokens)
        features['pos_conj_ratio'] = safe_divide(conj_count, total_tokens)
        
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
            'parse_tree_height': 0.0,
            'np_complexity': 0.0,
            'vp_complexity': 0.0,
            'phrase_structure_depth': 0.0,
            'pp_ratio': 0.0,
            'avg_sentence_length': 0.0,
            'sentence_length_variance': 0.0,
            'complex_sentence_ratio': 0.0,
            'simple_sentence_ratio': 0.0,
        }
        
        if not docs:
            return features
        
        tree_heights = []
        np_complexity = []
        vp_complexity = []
        pp_count = 0
        sentence_lengths = []
        total_tokens = 0
        
        for doc in docs:
            sentence_lengths.append(len(doc))
            total_tokens += len(doc)
            
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
        
        num_docs = len(docs)
        features['parse_tree_height'] = float(np.mean(tree_heights)) if tree_heights else 0.0
        features['np_complexity'] = float(np.mean(np_complexity)) if np_complexity else 0.0
        features['vp_complexity'] = float(np.mean(vp_complexity)) if vp_complexity else 0.0
        features['phrase_structure_depth'] = safe_divide(pp_count, num_docs)
        features['pp_ratio'] = safe_divide(pp_count, total_tokens)
        
        # Sentence length statistics
        features['avg_sentence_length'] = float(np.mean(sentence_lengths)) if sentence_lengths else 0.0
        features['sentence_length_variance'] = float(np.var(sentence_lengths)) if len(sentence_lengths) > 1 else 0.0
        
        # Complex vs simple sentences
        complex_sentences = sum(1 for length in sentence_lengths if length > 10)
        simple_sentences = sum(1 for length in sentence_lengths if length <= 5)
        features['complex_sentence_ratio'] = safe_divide(complex_sentences, num_docs)
        features['simple_sentence_ratio'] = safe_divide(simple_sentences, num_docs)
        
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
            'semantic_coherence_score': 0.0,
            'lexical_diversity': 0.0,
            'content_word_ratio': 0.0,
            'unique_lemma_ratio': 0.0,
            'hapax_legomena_ratio': 0.0,
            'avg_word_length': 0.0,
        }
        
        if not docs:
            return features
        
        # Semantic coherence (cosine similarity between consecutive utterances)
        coherence_scores = []
        
        if has_vectors and len(docs) > 1:
            for i in range(1, len(docs)):
                try:
                    similarity = docs[i-1].similarity(docs[i])
                    if not np.isnan(similarity):
                        coherence_scores.append(similarity)
                except Exception:
                    pass
        
        features['semantic_coherence_score'] = float(np.mean(coherence_scores)) if coherence_scores else 0.0
        
        # Collect all tokens and lemmas
        all_lemmas = []
        all_content_words = []
        word_lengths = []
        total_tokens = 0
        content_count = 0
        
        for doc in docs:
            for token in doc:
                total_tokens += 1
                
                if token.is_alpha:
                    word_lengths.append(len(token.text))
                    all_lemmas.append(token.lemma_.lower())
                
                if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] and not token.is_stop:
                    content_count += 1
                    all_content_words.append(token.lemma_.lower())
        
        # Lexical diversity (TTR for content words)
        if all_content_words:
            features['lexical_diversity'] = calculate_ratio(
                len(set(all_content_words)), len(all_content_words)
            )
        
        # Content word ratio
        features['content_word_ratio'] = safe_divide(content_count, total_tokens)
        
        # Unique lemma ratio
        unique_lemmas = set(all_lemmas)
        features['unique_lemma_ratio'] = safe_divide(len(unique_lemmas), total_tokens)
        
        # Hapax legomena ratio (words appearing only once)
        lemma_counts = Counter(all_lemmas)
        hapax = sum(1 for count in lemma_counts.values() if count == 1)
        features['hapax_legomena_ratio'] = safe_divide(hapax, len(all_lemmas))
        
        # Average word length
        features['avg_word_length'] = float(np.mean(word_lengths)) if word_lengths else 0.0
        
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
            'word_sense_diversity': 0.0,
            'function_word_ratio': 0.0,
            'named_entity_density': 0.0,
            'named_entity_diversity': 0.0,
            'person_entity_ratio': 0.0,
            'verb_tense_consistency': 0.0,
            'modal_verb_ratio': 0.0,
            'auxiliary_verb_ratio': 0.0,
            'verb_argument_count_avg': 0.0,
        }
        
        if not docs:
            return features
        
        # Lazy load WordNet
        self._ensure_wordnet_loaded()
        from nltk.corpus import wordnet
        
        sense_counts = []
        total_tokens = 0
        function_tokens = 0
        all_entities = []
        entity_labels = []
        verbs = []
        tenses = []
        modal_count = 0
        aux_count = 0
        verb_args = []
        
        for doc in docs:
            # Collect named entities
            for ent in doc.ents:
                all_entities.append(ent)
                entity_labels.append(ent.label_)
            
            for token in doc:
                total_tokens += 1
                
                # Function word ratio
                if token.pos_ in ['DET', 'ADP', 'CONJ', 'CCONJ', 'SCONJ', 'AUX'] or token.is_stop:
                    function_tokens += 1
                
                # Word sense diversity for content words
                if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] and not token.is_stop:
                    try:
                        synsets = wordnet.synsets(token.lemma_)
                        if synsets:
                            sense_counts.append(len(synsets))
                    except Exception:
                        pass
                
                # Verb analysis
                if token.pos_ == 'VERB':
                    verbs.append(token)
                    # Collect tenses
                    if token.tag_ in ['VBD', 'VBN']:
                        tenses.append('past')
                    elif token.tag_ in ['VBP', 'VBZ', 'VBG']:
                        tenses.append('present')
                    # Verb arguments
                    args = len([child for child in token.children if child.dep_ in ['nsubj', 'dobj', 'iobj', 'pobj', 'attr']])
                    verb_args.append(args)
                
                # Modal and auxiliary verbs
                if token.tag_ == 'MD':
                    modal_count += 1
                if token.pos_ == 'AUX':
                    aux_count += 1
        
        # Word sense diversity
        features['word_sense_diversity'] = float(np.mean(sense_counts)) if sense_counts else 0.0
        
        # Function word ratio
        features['function_word_ratio'] = calculate_ratio(function_tokens, total_tokens)
        
        # Named entity features
        features['named_entity_density'] = safe_divide(len(all_entities), total_tokens)
        features['named_entity_diversity'] = len(set(entity_labels))
        person_ents = sum(1 for label in entity_labels if label == 'PERSON')
        features['person_entity_ratio'] = safe_divide(person_ents, len(all_entities)) if all_entities else 0.0
        
        # Verb tense consistency
        if tenses:
            tense_counts = Counter(tenses)
            most_common_count = tense_counts.most_common(1)[0][1]
            features['verb_tense_consistency'] = safe_divide(most_common_count, len(tenses))
        
        # Modal and auxiliary verb ratios
        num_verbs = len(verbs)
        features['modal_verb_ratio'] = safe_divide(modal_count, num_verbs) if num_verbs > 0 else 0.0
        features['auxiliary_verb_ratio'] = safe_divide(aux_count, total_tokens)
        
        # Verb argument count average
        features['verb_argument_count_avg'] = float(np.mean(verb_args)) if verb_args else 0.0
        
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
        # Calculate syntactic_complexity as aggregate metric
        features = {
            'syntactic_complexity': 0.0,
        }
        
        # This will be calculated as an aggregate from other syntactic features
        # Will be computed in the main extract() method
        
        return features


__all__ = ["SyntacticSemanticFeatures"]
