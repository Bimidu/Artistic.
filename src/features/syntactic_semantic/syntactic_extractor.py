"""
Syntactic & Semantic Feature Extractor

Comprehensive NLP-based feature extractor for syntactic and semantic analysis
of conversational transcripts for ASD detection.

Features extracted (40+ features):
- POS tag ratios (noun, verb, adj, adv, pronoun, etc.)
- Dependency tree metrics (depth, width, distance)
- Clause structure (subordinate, coordinate, relative clauses)
- Sentence complexity metrics
- Semantic coherence scores
- Lexical diversity measures
- Named entity statistics
- Verb argument structure

Author: Randil Haturusinghe
"""

import sys
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
from collections import Counter
from multiprocessing import Pool, cpu_count

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SyntacticFeatureExtractor:
    """
    Real NLP-based syntactic and semantic feature extraction.
    
    Uses spaCy for NLP processing and NLTK for additional semantic analysis.
    Extracts 40+ linguistically meaningful features for ASD detection.
    """
    
    def __init__(self):
        """Initialize syntactic feature extractor with lazy loading."""
        self._nlp = None
        self._spacy_loaded = False
        self._wordnet_loaded = False
        
        # Comprehensive feature names (56 features - added 10 new robust features)
        self.feature_names = [
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
            
            # NEW: Repetition & Echolalia features (3 features)
            'word_repetition_ratio',
            'phrase_repetition_ratio',
            'immediate_repetition_count',
            
            # NEW: Pronoun usage features (3 features)
            'first_person_pronoun_ratio',
            'pronoun_reversal_indicators',
            'pronoun_diversity',
            
            # NEW: Question & Interaction features (4 features)
            'question_ratio',
            'wh_question_ratio',
            'yes_no_question_ratio',
            'imperative_ratio',
            
            # Aggregated complexity (1 feature)
            'syntactic_complexity',
        ]
        
        logger.info(f"SyntacticFeatureExtractor initialized with {len(self.feature_names)} real NLP features")
    
    def _ensure_spacy_loaded(self) -> None:
        """Lazy load spaCy model on first use."""
        if self._spacy_loaded:
            return
            
        try:
            import spacy
            try:
                self._nlp = spacy.load("en_core_web_md")
                logger.debug("spaCy model 'en_core_web_md' loaded successfully")
            except OSError:
                logger.warning("spaCy model not found. Installing...")
                import subprocess
                subprocess.run(
                    [sys.executable, "-m", "spacy", "download", "en_core_web_md"],
                    check=True
                )
                self._nlp = spacy.load("en_core_web_md")
            self._spacy_loaded = True
        except Exception as e:
            logger.error(f"Failed to load spaCy: {e}")
            raise
    
    def _ensure_wordnet_loaded(self) -> None:
        """Lazy load NLTK WordNet on first use."""
        if self._wordnet_loaded:
            return
            
        try:
            from nltk.corpus import wordnet
            wordnet.synsets('test')
            self._wordnet_loaded = True
            logger.debug("NLTK WordNet loaded successfully")
        except LookupError:
            logger.info("Downloading NLTK WordNet data...")
            import nltk
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            self._wordnet_loaded = True
    
    def _get_dependency_depth(self, token) -> int:
        """Get depth of token in dependency tree."""
        depth = 0
        current = token
        while current.head != current and depth < 50:
            depth += 1
            current = current.head
        return depth
    
    def _safe_divide(self, numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safe division to avoid division by zero."""
        return numerator / denominator if denominator > 0 else default
    
    def extract_from_text(self, text: str) -> Dict[str, float]:
        """
        Extract syntactic and semantic features from text using NLP.
        
        Args:
            text: Input text
        
        Returns:
            Dictionary of feature values
        """
        logger.info(f"Extracting syntactic features from text (length: {len(text)})")
        
        if not text or not text.strip():
            logger.warning("Empty text provided, returning zero features")
            return {name: 0.0 for name in self.feature_names}
        
        # Load spaCy model
        self._ensure_spacy_loaded()
        
        # Process text
        doc = self._nlp(text)
        
        # Initialize feature dict
        features = {name: 0.0 for name in self.feature_names}
        
        if len(doc) == 0:
            return features
        
        # Split into sentences
        sentences = list(doc.sents)
        num_sentences = len(sentences) if sentences else 1
        total_tokens = len(doc)
        
        # ===== POS RATIOS =====
        pos_counts = Counter(token.pos_ for token in doc)
        features['pos_noun_ratio'] = self._safe_divide(pos_counts.get('NOUN', 0), total_tokens)
        features['pos_verb_ratio'] = self._safe_divide(pos_counts.get('VERB', 0), total_tokens)
        features['pos_adj_ratio'] = self._safe_divide(pos_counts.get('ADJ', 0), total_tokens)
        features['pos_adv_ratio'] = self._safe_divide(pos_counts.get('ADV', 0), total_tokens)
        features['pos_pronoun_ratio'] = self._safe_divide(pos_counts.get('PRON', 0), total_tokens)
        features['pos_det_ratio'] = self._safe_divide(pos_counts.get('DET', 0), total_tokens)
        features['pos_adp_ratio'] = self._safe_divide(pos_counts.get('ADP', 0), total_tokens)
        conj_count = pos_counts.get('CONJ', 0) + pos_counts.get('CCONJ', 0) + pos_counts.get('SCONJ', 0)
        features['pos_conj_ratio'] = self._safe_divide(conj_count, total_tokens)
        
        # ===== DEPENDENCY TREE METRICS =====
        all_depths = []
        all_distances = []
        root_distances = []
        children_counts = []
        
        for token in doc:
            depth = self._get_dependency_depth(token)
            all_depths.append(depth)
            
            if token.head != token:
                distance = abs(token.i - token.head.i)
                all_distances.append(distance)
            
            if token.dep_ == 'ROOT':
                for other_token in doc:
                    root_distances.append(abs(other_token.i - token.i))
            
            children_counts.append(len(list(token.children)))
        
        features['dependency_tree_depth'] = float(np.mean(all_depths)) if all_depths else 0.0
        features['dependency_tree_width'] = float(max(all_depths)) if all_depths else 0.0
        features['avg_dependency_distance'] = float(np.mean(all_distances)) if all_distances else 0.0
        features['max_dependency_distance'] = float(max(all_distances)) if all_distances else 0.0
        features['root_distance_avg'] = float(np.mean(root_distances)) if root_distances else 0.0
        features['dependency_branching_factor'] = float(np.mean(children_counts)) if children_counts else 0.0
        
        # ===== CLAUSE STRUCTURE =====
        subordinate_count = 0
        coordinate_count = 0
        relative_count = 0
        complement_count = 0
        adverbial_count = 0
        clause_markers = 0
        
        for token in doc:
            if token.dep_ == 'mark':
                clause_markers += 1
            if token.dep_ in ['advcl']:
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
        
        features['clause_count'] = self._safe_divide(clause_markers, num_sentences)
        features['subordinate_clause_ratio'] = self._safe_divide(subordinate_count, num_sentences)
        features['coordinate_clause_ratio'] = self._safe_divide(coordinate_count, num_sentences)
        features['relative_clause_ratio'] = self._safe_divide(relative_count, num_sentences)
        features['complement_clause_ratio'] = self._safe_divide(complement_count, num_sentences)
        features['adverbial_clause_ratio'] = self._safe_divide(adverbial_count, num_sentences)
        
        # ===== SENTENCE COMPLEXITY =====
        sentence_lengths = [len(sent) for sent in sentences]
        features['avg_sentence_length'] = float(np.mean(sentence_lengths)) if sentence_lengths else 0.0
        features['sentence_length_variance'] = float(np.var(sentence_lengths)) if len(sentence_lengths) > 1 else 0.0
        
        complex_sentences = sum(1 for sent in sentences if len(sent) > 10)
        simple_sentences = sum(1 for sent in sentences if len(sent) <= 5)
        features['complex_sentence_ratio'] = self._safe_divide(complex_sentences, num_sentences)
        features['simple_sentence_ratio'] = self._safe_divide(simple_sentences, num_sentences)
        
        tree_heights = []
        for sent in sentences:
            if len(sent) > 0:
                max_depth = max(self._get_dependency_depth(token) for token in sent)
                tree_heights.append(max_depth)
        features['parse_tree_height'] = float(np.mean(tree_heights)) if tree_heights else 0.0
        
        # Sentence complexity score (composite)
        features['sentence_complexity_score'] = (
            features['subordinate_clause_ratio'] * 0.3 +
            features['coordinate_clause_ratio'] * 0.2 +
            min(features['avg_sentence_length'] / 20.0, 1.0) * 0.3 +
            min(features['parse_tree_height'] / 10.0, 1.0) * 0.2
        )
        
        # ===== PHRASE STRUCTURE =====
        np_lengths = [len(chunk) for chunk in doc.noun_chunks]
        vp_complexity = []
        pp_count = 0
        
        for token in doc:
            if token.pos_ == 'VERB':
                vp_complexity.append(len(list(token.children)))
            if token.pos_ == 'ADP':
                pp_count += 1
        
        features['phrase_structure_depth'] = self._safe_divide(pp_count, num_sentences)
        features['np_complexity'] = float(np.mean(np_lengths)) if np_lengths else 0.0
        features['vp_complexity'] = float(np.mean(vp_complexity)) if vp_complexity else 0.0
        features['pp_ratio'] = self._safe_divide(pp_count, total_tokens)
        
        # ===== SEMANTIC FEATURES =====
        # Content vs function words
        content_words = [token for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] and not token.is_stop]
        function_words = [token for token in doc if token.pos_ in ['DET', 'ADP', 'CONJ', 'CCONJ', 'SCONJ', 'AUX'] or token.is_stop]
        
        features['content_word_ratio'] = self._safe_divide(len(content_words), total_tokens)
        features['function_word_ratio'] = self._safe_divide(len(function_words), total_tokens)
        
        # Lexical diversity
        lemmas = [token.lemma_.lower() for token in doc if token.is_alpha]
        unique_lemmas = set(lemmas)
        features['lexical_diversity'] = self._safe_divide(len(unique_lemmas), len(lemmas))
        features['unique_lemma_ratio'] = self._safe_divide(len(unique_lemmas), total_tokens)
        
        # Hapax legomena (words appearing only once)
        lemma_counts = Counter(lemmas)
        hapax = sum(1 for count in lemma_counts.values() if count == 1)
        features['hapax_legomena_ratio'] = self._safe_divide(hapax, len(lemmas))
        
        # Average word length
        word_lengths = [len(token.text) for token in doc if token.is_alpha]
        features['avg_word_length'] = float(np.mean(word_lengths)) if word_lengths else 0.0
        
        # Word sense diversity (using WordNet)
        self._ensure_wordnet_loaded()
        try:
            from nltk.corpus import wordnet
            sense_counts = []
            for token in content_words[:50]:  # Limit for performance
                synsets = wordnet.synsets(token.lemma_)
                if synsets:
                    sense_counts.append(len(synsets))
            features['word_sense_diversity'] = float(np.mean(sense_counts)) if sense_counts else 0.0
        except Exception:
            features['word_sense_diversity'] = 0.0
        
        # Semantic coherence (using word vectors if available)
        if self._nlp.vocab.vectors.shape[0] > 0 and len(sentences) > 1:
            coherence_scores = []
            for i in range(1, len(sentences)):
                try:
                    sent_prev = sentences[i-1].as_doc()
                    sent_curr = sentences[i].as_doc()
                    if sent_prev.vector_norm > 0 and sent_curr.vector_norm > 0:
                        similarity = sent_prev.similarity(sent_curr)
                        if not np.isnan(similarity):
                            coherence_scores.append(similarity)
                except Exception:
                    pass
            features['semantic_coherence_score'] = float(np.mean(coherence_scores)) if coherence_scores else 0.0
        
        # ===== NAMED ENTITY FEATURES =====
        ents = list(doc.ents)
        features['named_entity_density'] = self._safe_divide(len(ents), total_tokens)
        features['named_entity_diversity'] = len(set(ent.label_ for ent in ents))
        person_ents = sum(1 for ent in ents if ent.label_ == 'PERSON')
        features['person_entity_ratio'] = self._safe_divide(person_ents, len(ents)) if ents else 0.0
        
        # ===== VERB ANALYSIS =====
        verbs = [token for token in doc if token.pos_ == 'VERB']
        
        # Tense consistency
        tenses = []
        for token in verbs:
            if token.tag_ in ['VBD', 'VBN']:
                tenses.append('past')
            elif token.tag_ in ['VBP', 'VBZ', 'VBG']:
                tenses.append('present')
        if tenses:
            tense_counts = Counter(tenses)
            most_common_count = tense_counts.most_common(1)[0][1]
            features['verb_tense_consistency'] = self._safe_divide(most_common_count, len(tenses))
        
        # Modal and auxiliary verbs
        modal_count = sum(1 for token in doc if token.tag_ == 'MD')
        aux_count = sum(1 for token in doc if token.pos_ == 'AUX')
        features['modal_verb_ratio'] = self._safe_divide(modal_count, len(verbs)) if verbs else 0.0
        features['auxiliary_verb_ratio'] = self._safe_divide(aux_count, total_tokens)
        
        # Verb argument count
        verb_args = []
        for verb in verbs:
            args = len([child for child in verb.children if child.dep_ in ['nsubj', 'dobj', 'iobj', 'pobj', 'attr']])
            verb_args.append(args)
        features['verb_argument_count_avg'] = float(np.mean(verb_args)) if verb_args else 0.0
        
        # ===== NEW: REPETITION & ECHOLALIA FEATURES =====
        words = [token.text.lower() for token in doc if token.is_alpha]
        word_counts = Counter(words)
        repeated_words = sum(1 for count in word_counts.values() if count > 1)
        features['word_repetition_ratio'] = self._safe_divide(repeated_words, len(word_counts))
        
        # Phrase repetition (bigrams)
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        bigram_counts = Counter(bigrams)
        repeated_bigrams = sum(1 for count in bigram_counts.values() if count > 1)
        features['phrase_repetition_ratio'] = self._safe_divide(repeated_bigrams, len(bigram_counts))
        
        # Immediate repetition (consecutive identical words)
        immediate_reps = sum(1 for i in range(len(words)-1) if words[i] == words[i+1])
        features['immediate_repetition_count'] = float(immediate_reps)
        
        # ===== NEW: PRONOUN USAGE FEATURES =====
        pronouns = [token for token in doc if token.pos_ == 'PRON']
        first_person_pronouns = [p for p in pronouns if p.text.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']]
        features['first_person_pronoun_ratio'] = self._safe_divide(len(first_person_pronouns), len(pronouns)) if pronouns else 0.0
        
        # Pronoun reversal indicators (you/me confusion patterns)
        reversal_indicators = 0
        for i, token in enumerate(doc):
            if token.text.lower() in ['you', 'me', 'i']:
                # Check for unusual patterns (simplified heuristic)
                if i > 0 and doc[i-1].text.lower() in ['want', 'give', 'help']:
                    reversal_indicators += 1
        features['pronoun_reversal_indicators'] = float(reversal_indicators)
        
        # Pronoun diversity
        unique_pronouns = len(set(p.text.lower() for p in pronouns))
        features['pronoun_diversity'] = float(unique_pronouns)
        
        # ===== NEW: QUESTION & INTERACTION FEATURES =====
        questions = [sent for sent in sentences if sent.text.strip().endswith('?')]
        features['question_ratio'] = self._safe_divide(len(questions), num_sentences)
        
        # WH-questions (what, where, when, why, who, how)
        wh_questions = sum(1 for sent in questions if any(
            token.text.lower() in ['what', 'where', 'when', 'why', 'who', 'how']
            for token in sent
        ))
        features['wh_question_ratio'] = self._safe_divide(wh_questions, len(questions)) if questions else 0.0
        
        # Yes/No questions (auxiliary verb at start)
        yn_questions = sum(1 for sent in questions if len(sent) > 0 and sent[0].pos_ == 'AUX')
        features['yes_no_question_ratio'] = self._safe_divide(yn_questions, len(questions)) if questions else 0.0
        
        # Imperative sentences (commands)
        imperatives = sum(1 for sent in sentences if len(sent) > 0 and sent[0].pos_ == 'VERB' and sent[0].tag_ == 'VB')
        features['imperative_ratio'] = self._safe_divide(imperatives, num_sentences)
        
        # ===== AGGREGATED SYNTACTIC COMPLEXITY =====
        features['syntactic_complexity'] = (
            min(features['dependency_tree_depth'] / 10.0, 1.0) * 0.25 +
            features['subordinate_clause_ratio'] * 0.25 +
            features['sentence_complexity_score'] * 0.25 +
            min(features['np_complexity'] / 5.0, 1.0) * 0.25
        )
        
        logger.info(f"Extracted {len(features)} syntactic/semantic features")
        return features
    
    def extract_from_transcript(self, transcript_data: Any) -> Dict[str, float]:
        """
        Extract syntactic features from transcript data.
        
        Args:
            transcript_data: Transcript data (TranscriptData object or dict)
        
        Returns:
            Dictionary of feature values
        """
        logger.info("Extracting syntactic features from transcript")
        
        # Handle different transcript formats
        text = ""
        
        if hasattr(transcript_data, 'utterances'):
            # TranscriptData object - include BOTH child and adult utterances for context
            all_utterances = []
            for u in transcript_data.utterances:
                if hasattr(u, 'speaker') and hasattr(u, 'text') and u.text:
                    # Include child and adult utterances (MOT, FAT, etc.)
                    speaker = u.speaker.upper() if u.speaker else ''
                    if speaker in ['CHI', 'CHILD', 'TARGET_CHILD', 'MOT', 'MOTHER', 'FAT', 'FATHER', 'INV', 'INVESTIGATOR']:
                        all_utterances.append(u.text)
            text = " ".join(all_utterances)
        elif isinstance(transcript_data, dict):
            # Dictionary format
            if 'text' in transcript_data:
                text = transcript_data['text']
            elif 'utterances' in transcript_data:
                text = " ".join(u.get('text', '') for u in transcript_data['utterances'])
        elif isinstance(transcript_data, str):
            text = transcript_data
        
        # Ensure minimum text length for reliable analysis
        if len(text.strip()) < 50:
            logger.warning(f"Text too short ({len(text)} chars), features may be unreliable")
        
        return self.extract_from_text(text)
    
    def _extract_child_text_from_chat(self, file_path: Path) -> tuple[str, Optional[str]]:
        """
        Extract utterances and diagnosis from CHAT file manually (fallback parser).
        Now extracts BOTH child and adult utterances for better context.
        
        Args:
            file_path: Path to CHAT file
            
        Returns:
            Tuple of (concatenated utterances, diagnosis)
        """
        all_utterances = []
        diagnosis = None
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    
                    # Extract diagnosis from @Types header
                    if line.startswith('@Types:'):
                        # Check if ASD or TD is mentioned in the types
                        types_upper = line.upper()
                        if 'ASD' in types_upper:
                            diagnosis = 'ASD'
                        elif 'TD' in types_upper or 'TYP' in types_upper:
                            diagnosis = 'TD'
                    
                    # Skip empty lines and other header lines
                    if not line or line.startswith('@'):
                        continue
                    
                    # Look for speaker lines (child and adult)
                    if line.startswith('*'):
                        # Extract speaker code
                        if ':' in line:
                            speaker_part = line.split(':', 1)[0]
                            speaker = speaker_part[1:].upper()  # Remove * and uppercase
                            
                            # Include child and adult speakers
                            if speaker in ['CHI', 'TARGET_CHILD', 'MOT', 'MOTHER', 'FAT', 'FATHER', 'INV', 'INVESTIGATOR']:
                                # Extract text after speaker code
                                text = line.split(':', 1)[1].strip()
                                # Remove CHAT annotations (e.g., [+ exc], [* m:+ed], etc.)
                                import re
                                text = re.sub(r'\[.*?\]', '', text)
                                text = re.sub(r'<.*?>', '', text)
                                text = re.sub(r'\+\w+', '', text)
                                text = text.strip()
                                if text:
                                    all_utterances.append(text)
        except Exception as e:
            logger.warning(f"Error parsing CHAT file {file_path}: {e}")
            return "", None
        
        return " ".join(all_utterances), diagnosis

    
    def _process_single_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Process a single transcript file (helper for parallel processing).
        
        Args:
            file_path: Path to transcript file
            
        Returns:
            Dictionary of features or None if error
        """
        try:
            # Import CHAT parser
            try:
                from src.parsers.chat_parser import ChatParser
                parser = ChatParser()
            except ImportError:
                parser = None
            
            extracted_diagnosis = None
            
            if parser:
                # Parse with CHAT parser
                try:
                    transcript = parser.parse_file(str(file_path))
                    features = self.extract_from_transcript(transcript)
                except Exception as e:
                    logger.warning(f"ChatParser failed for {file_path}, using fallback: {e}")
                    # Fallback to manual parsing
                    text, extracted_diagnosis = self._extract_child_text_from_chat(file_path)
                    features = self.extract_from_text(text)
            else:
                # Fallback: manual CHAT parsing
                text, extracted_diagnosis = self._extract_child_text_from_chat(file_path)
                features = self.extract_from_text(text)
            
            # Determine diagnosis (prioritize extracted from @Types, then path-based)
            if extracted_diagnosis:
                features['diagnosis'] = extracted_diagnosis
            else:
                # Infer diagnosis from path as fallback
                path_str = str(file_path).upper()
                if '/ASD/' in path_str or '_ASD_' in path_str or 'ASD' in file_path.parent.name.upper():
                    features['diagnosis'] = 'ASD'
                elif '/TD/' in path_str or '/TYP/' in path_str or '_TD_' in path_str:
                    features['diagnosis'] = 'TD'
                else:
                    features['diagnosis'] = None  # Unknown
            
            features['file_path'] = str(file_path)
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features from {file_path}: {e}")
            return None
    
    def extract_from_directory(self, directory: Path, parallel: bool = True, n_workers: Optional[int] = None) -> pd.DataFrame:
        """
        Extract features from all CHAT files in directory.
        
        Args:
            directory: Directory path
            parallel: Whether to use parallel processing (default: True)
            n_workers: Number of worker processes (default: CPU count - 1)
        
        Returns:
            DataFrame with features
        """
        logger.info(f"Extracting syntactic features from directory: {directory}")
        
        # Find transcript files
        directory = Path(directory)
        transcript_files = list(directory.rglob('*.cha'))
        
        if not transcript_files:
            logger.warning(f"No .cha files found in {directory}")
            return pd.DataFrame()
        
        logger.info(f"Found {len(transcript_files)} transcript files")
        
        if parallel and len(transcript_files) > 1:
            # Parallel processing
            if n_workers is None:
                n_workers = max(1, cpu_count() - 2) 
            
            logger.info(f"Using parallel processing with {n_workers} workers")
            
            # Create a pool of workers
            with Pool(processes=n_workers) as pool:
                results = pool.map(self._process_single_file, transcript_files)
            
            # Filter out None results
            data = [r for r in results if r is not None]
        else:
            # Sequential processing (fallback)
            logger.info("Using sequential processing")
            data = []
            for transcript_file in transcript_files:
                result = self._process_single_file(transcript_file)
                if result is not None:
                    data.append(result)
        
        logger.info(f"Extracted features from {len(data)} transcript files")
        return pd.DataFrame(data)


