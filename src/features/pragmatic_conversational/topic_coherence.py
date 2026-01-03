"""
Topic Maintenance and Semantic Coherence Feature Extractor (Section 3.3.2)

This module extracts features related to topic maintenance and semantic coherence,
which are often impaired in children with ASD. Based on methodology section 3.3.2.

Features implemented:
- LDA Topic Modeling for topic identification
- Word Embedding similarity (using spaCy vectors)
- Semantic coherence scores
- Topic shift detection
- Inter-speaker and within-speaker consistency

References:
- Ellis et al. (2021): ASD children had significantly lower semantic similarity scores
- Hybrid approach combining LDA and word embeddings

Author: Bimidu Gunathilake
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import re

from src.parsers.chat_parser import TranscriptData, Utterance
from src.utils.helpers import safe_divide, calculate_ratio
from src.utils.logger import get_logger
from ..base_features import BaseFeatureExtractor, FeatureResult

logger = get_logger(__name__)

# Try to import NLP libraries
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not available. Some semantic features will be limited.")

try:
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer
    LDA_AVAILABLE = True
except ImportError:
    LDA_AVAILABLE = False
    logger.warning("sklearn LDA not available. Topic modeling features will be limited.")


class TopicCoherenceFeatures(BaseFeatureExtractor):
    """
    Extract topic maintenance and semantic coherence features (Section 3.3.2).
    
    Features capture:
    - Topic coherence using LDA
    - Semantic similarity using word embeddings
    - Topic shift detection
    - Inter-speaker semantic consistency
    - Within-speaker topic maintenance
    
    Based on Ellis et al. (2021) approach using word embeddings for semantic analysis.
    
    Example:
        >>> extractor = TopicCoherenceFeatures()
        >>> features = extractor.extract(transcript)
        >>> print(features.features['semantic_coherence_score'])
    """
    
    # Topic shift thresholds
    TOPIC_SHIFT_THRESHOLD = 0.3  # Cosine similarity threshold
    N_TOPICS_DEFAULT = 5  # Default number of LDA topics
    WINDOW_SIZE = 3  # Window size for topic shift detection
    
    def __init__(self, n_topics: int = 5):
        """
        Initialize topic coherence extractor.
        
        Args:
            n_topics: Number of topics for LDA modeling
        """
        super().__init__()
        self.n_topics = n_topics
        self._nlp = None
        self._initialize_nlp()
    
    def _initialize_nlp(self):
        """Initialize spaCy NLP model for semantic analysis.
        
        Tries models in order of preference:
        1. en_core_web_lg (large, 300D vectors, best quality)
        2. en_core_web_md (medium, 300D vectors, good quality)
        3. en_core_web_sm (small, no vectors, fallback only)
        """
        if not SPACY_AVAILABLE:
            logger.warning("spaCy not available, semantic features will use fallback methods")
            return
        
        # Try models in order: lg (best) -> md (good) -> sm (fallback)
        model_preferences = ["en_core_web_lg", "en_core_web_md", "en_core_web_sm"]
        
        last_error = None
        for model_name in model_preferences:
            try:
                self._nlp = spacy.load(model_name)
                has_vectors = self._nlp.vocab.vectors.size > 0
                
                if model_name == "en_core_web_lg":
                    logger.info(f"Loaded spaCy {model_name} model (large, 300D vectors) for semantic analysis")
                elif model_name == "en_core_web_md":
                    logger.info(f"Loaded spaCy {model_name} model (medium, 300D vectors) for semantic analysis")
                else:
                    if has_vectors:
                        logger.info(f"Loaded spaCy {model_name} model for semantic analysis")
                    else:
                        logger.warning(f"Loaded spaCy {model_name} model (no word vectors - similarity may be limited)")
                
                return  # Successfully loaded, exit
                
            except OSError as e:
                last_error = str(e)
                logger.debug(f"Failed to load {model_name}: {last_error}")
                continue  # Try next model
        
        # If all models failed, provide helpful error message
        logger.warning(
            f"No spaCy model found. Semantic features will be limited.\n"
            f"To install a model, run: python -m spacy download en_core_web_lg\n"
            f"Last error: {last_error if last_error else 'Model not found'}"
        )
        self._nlp = None
    
    @property
    def feature_names(self) -> List[str]:
        """Get list of feature names."""
        return [
            # Semantic coherence (Ellis et al. 2021 inspired)
            'semantic_coherence_score',
            'semantic_coherence_std',
            'min_semantic_similarity',
            'max_semantic_similarity',
            
            # Child-specific semantic coherence
            'child_semantic_coherence',
            'child_response_relevance',  # Similarity to previous adult turn
            'child_response_relevance_std',
            
            # Inter-speaker similarity
            'inter_speaker_similarity_mean',
            'inter_speaker_similarity_std',
            'child_to_adult_similarity',
            'adult_to_child_similarity',
            
            # Within-speaker consistency
            'child_within_consistency',
            'adult_within_consistency',
            'child_topic_drift',  # Change in similarity over conversation
            
            # Topic shift detection
            'topic_shift_count',
            'topic_shift_ratio',
            'abrupt_topic_shift_count',
            'avg_topic_duration_turns',
            'topic_return_count',  # Returning to previous topics
            
            # LDA topic features
            'topic_diversity',  # How many topics used
            'dominant_topic_ratio',  # Concentration on one topic
            'topic_entropy',  # Distribution across topics
            'child_topic_consistency',
            
            # Vocabulary-based coherence
            'lexical_overlap_mean',
            'lexical_overlap_child',
            'content_word_overlap',
            'novel_word_ratio',  # New words introduced
            
            # Contextual appropriateness
            'on_topic_response_ratio',
            'off_topic_response_count',
            'tangential_response_ratio',
        ]
    
    def extract(self, transcript: TranscriptData) -> FeatureResult:
        """
        Extract topic coherence features from transcript.
        
        Args:
            transcript: Parsed transcript data
            
        Returns:
            FeatureResult with topic coherence features
        """
        features = {}
        
        all_utterances = transcript.valid_utterances
        child_utterances = self.get_child_utterances(transcript)
        adult_utterances = self.get_adult_utterances(transcript)
        
        logger.debug(f"Extracting topic coherence features from {len(all_utterances)} utterances")
        
        # Get utterance texts
        all_texts = [self._clean_text(u.text) for u in all_utterances]
        child_texts = [self._clean_text(u.text) for u in child_utterances]
        adult_texts = [self._clean_text(u.text) for u in adult_utterances]
        
        # Semantic coherence features (embedding-based)
        semantic_features = self._calculate_semantic_coherence(
            all_utterances, child_utterances, all_texts, child_texts
        )
        features.update(semantic_features)
        
        # Inter-speaker similarity
        inter_speaker_features = self._calculate_inter_speaker_similarity(
            all_utterances
        )
        features.update(inter_speaker_features)
        
        # Within-speaker consistency
        within_features = self._calculate_within_speaker_consistency(
            child_utterances, adult_utterances, child_texts, adult_texts
        )
        features.update(within_features)
        
        # Topic shift detection
        shift_features = self._calculate_topic_shifts(all_utterances, all_texts)
        features.update(shift_features)
        
        # LDA topic features
        lda_features = self._calculate_lda_features(all_texts, child_texts)
        features.update(lda_features)
        
        # Vocabulary-based coherence
        vocab_features = self._calculate_vocabulary_coherence(
            all_utterances, child_utterances
        )
        features.update(vocab_features)
        
        # Contextual appropriateness
        context_features = self._calculate_contextual_appropriateness(all_utterances)
        features.update(context_features)
        
        logger.debug(f"Extracted {len(features)} topic coherence features")
        
        return FeatureResult(
            features=features,
            feature_type='topic_coherence',
            metadata={
                'total_utterances': len(all_utterances),
                'has_spacy': self._nlp is not None,
                'has_lda': LDA_AVAILABLE,
                'n_topics': self.n_topics
            }
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean utterance text for analysis."""
        if not text:
            return ""
        
        # Remove CHAT-specific markers
        text = re.sub(r'\[.*?\]', '', text)  # Remove annotations in brackets
        text = re.sub(r'<.*?>', '', text)  # Remove overlap markers
        text = re.sub(r'&=\w+', '', text)  # Remove paralinguistic markers
        text = re.sub(r'\(\.\)', '', text)  # Remove pause markers
        text = re.sub(r'xxx', '', text)  # Remove unintelligible markers
        text = re.sub(r'www', '', text)  # Remove untranscribed markers
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        return text.strip().lower()
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding vector for text using spaCy."""
        if self._nlp is None or not text:
            return None
        
        doc = self._nlp(text)
        if doc.vector_norm == 0:
            return None
        
        return doc.vector
    
    def _cosine_similarity(
        self,
        vec1: Optional[np.ndarray],
        vec2: Optional[np.ndarray]
    ) -> float:
        """Calculate cosine similarity between two vectors."""
        if vec1 is None or vec2 is None:
            return 0.0
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def _calculate_semantic_coherence(
        self,
        all_utterances: List[Utterance],
        child_utterances: List[Utterance],
        all_texts: List[str],
        child_texts: List[str]
    ) -> Dict[str, float]:
        """
        Calculate semantic coherence features using embeddings.
        
        Based on Ellis et al. (2021) approach: compute cosine similarity
        between consecutive utterances.
        """
        features = {
            'semantic_coherence_score': 0.0,
            'semantic_coherence_std': 0.0,
            'min_semantic_similarity': 0.0,
            'max_semantic_similarity': 0.0,
            'child_semantic_coherence': 0.0,
            'child_response_relevance': 0.0,
            'child_response_relevance_std': 0.0,
        }
        
        if len(all_texts) < 2:
            return features
        
        # Get embeddings
        embeddings = [self._get_embedding(text) for text in all_texts]
        
        # Calculate consecutive similarities
        similarities = []
        for i in range(1, len(embeddings)):
            sim = self._cosine_similarity(embeddings[i-1], embeddings[i])
            if sim > 0:  # Only count valid similarities
                similarities.append(sim)
        
        if similarities:
            features['semantic_coherence_score'] = float(np.mean(similarities))
            features['semantic_coherence_std'] = float(np.std(similarities))
            features['min_semantic_similarity'] = float(np.min(similarities))
            features['max_semantic_similarity'] = float(np.max(similarities))
        
        # Child-specific coherence
        child_embeddings = [self._get_embedding(text) for text in child_texts]
        child_sims = []
        for i in range(1, len(child_embeddings)):
            sim = self._cosine_similarity(child_embeddings[i-1], child_embeddings[i])
            if sim > 0:
                child_sims.append(sim)
        
        if child_sims:
            features['child_semantic_coherence'] = float(np.mean(child_sims))
        
        # Child response relevance (similarity to preceding adult turn)
        response_relevances = self._calculate_response_relevance(all_utterances)
        if response_relevances:
            features['child_response_relevance'] = float(np.mean(response_relevances))
            features['child_response_relevance_std'] = float(np.std(response_relevances))
        
        return features
    
    def _calculate_response_relevance(
        self,
        utterances: List[Utterance]
    ) -> List[float]:
        """Calculate semantic relevance of child responses to adult prompts."""
        relevances = []
        adult_codes = {'MOT', 'FAT', 'INV', 'INV1', 'INV2', 'EXA', 'EXP'}
        
        for i in range(1, len(utterances)):
            if utterances[i].speaker == 'CHI' and utterances[i-1].speaker in adult_codes:
                prev_text = self._clean_text(utterances[i-1].text)
                curr_text = self._clean_text(utterances[i].text)
                
                prev_emb = self._get_embedding(prev_text)
                curr_emb = self._get_embedding(curr_text)
                
                sim = self._cosine_similarity(prev_emb, curr_emb)
                if sim > 0:
                    relevances.append(sim)
        
        return relevances
    
    def _calculate_inter_speaker_similarity(
        self,
        utterances: List[Utterance]
    ) -> Dict[str, float]:
        """Calculate semantic similarity between speakers."""
        features = {
            'inter_speaker_similarity_mean': 0.0,
            'inter_speaker_similarity_std': 0.0,
            'child_to_adult_similarity': 0.0,
            'adult_to_child_similarity': 0.0,
        }
        
        adult_codes = {'MOT', 'FAT', 'INV', 'INV1', 'INV2', 'EXA', 'EXP'}
        
        child_to_adult = []
        adult_to_child = []
        
        for i in range(1, len(utterances)):
            prev = utterances[i-1]
            curr = utterances[i]
            
            prev_text = self._clean_text(prev.text)
            curr_text = self._clean_text(curr.text)
            
            prev_emb = self._get_embedding(prev_text)
            curr_emb = self._get_embedding(curr_text)
            
            sim = self._cosine_similarity(prev_emb, curr_emb)
            
            if prev.speaker == 'CHI' and curr.speaker in adult_codes and sim > 0:
                adult_to_child.append(sim)
            elif prev.speaker in adult_codes and curr.speaker == 'CHI' and sim > 0:
                child_to_adult.append(sim)
        
        all_inter = child_to_adult + adult_to_child
        
        if all_inter:
            features['inter_speaker_similarity_mean'] = float(np.mean(all_inter))
            features['inter_speaker_similarity_std'] = float(np.std(all_inter))
        
        if child_to_adult:
            features['child_to_adult_similarity'] = float(np.mean(child_to_adult))
        
        if adult_to_child:
            features['adult_to_child_similarity'] = float(np.mean(adult_to_child))
        
        return features
    
    def _calculate_within_speaker_consistency(
        self,
        child_utterances: List[Utterance],
        adult_utterances: List[Utterance],
        child_texts: List[str],
        adult_texts: List[str]
    ) -> Dict[str, float]:
        """Calculate within-speaker topic consistency."""
        features = {
            'child_within_consistency': 0.0,
            'adult_within_consistency': 0.0,
            'child_topic_drift': 0.0,
        }
        
        # Child consistency
        if len(child_texts) >= 2:
            child_embeddings = [self._get_embedding(t) for t in child_texts]
            child_sims = []
            for i in range(1, len(child_embeddings)):
                sim = self._cosine_similarity(child_embeddings[i-1], child_embeddings[i])
                if sim > 0:
                    child_sims.append(sim)
            
            if child_sims:
                features['child_within_consistency'] = float(np.mean(child_sims))
                
                # Topic drift: compare first half to second half similarities
                if len(child_sims) >= 4:
                    mid = len(child_sims) // 2
                    first_half = np.mean(child_sims[:mid])
                    second_half = np.mean(child_sims[mid:])
                    features['child_topic_drift'] = float(first_half - second_half)
        
        # Adult consistency
        if len(adult_texts) >= 2:
            adult_embeddings = [self._get_embedding(t) for t in adult_texts]
            adult_sims = []
            for i in range(1, len(adult_embeddings)):
                sim = self._cosine_similarity(adult_embeddings[i-1], adult_embeddings[i])
                if sim > 0:
                    adult_sims.append(sim)
            
            if adult_sims:
                features['adult_within_consistency'] = float(np.mean(adult_sims))
        
        return features
    
    def _calculate_topic_shifts(
        self,
        utterances: List[Utterance],
        texts: List[str]
    ) -> Dict[str, float]:
        """
        Detect topic shifts using sliding window approach.
        
        Methodology: compare embedding centroids across windows to detect
        abrupt topic changes.
        """
        features = {
            'topic_shift_count': 0,
            'topic_shift_ratio': 0.0,
            'abrupt_topic_shift_count': 0,
            'avg_topic_duration_turns': 0.0,
            'topic_return_count': 0,
        }
        
        if len(texts) < self.WINDOW_SIZE * 2:
            return features
        
        # Get embeddings
        embeddings = [self._get_embedding(t) for t in texts]
        valid_embeddings = [(i, e) for i, e in enumerate(embeddings) if e is not None]
        
        if len(valid_embeddings) < self.WINDOW_SIZE * 2:
            return features
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(1, len(embeddings)):
            sim = self._cosine_similarity(embeddings[i-1], embeddings[i])
            similarities.append(sim)
        
        # Detect topic shifts (low similarity)
        topic_shifts = []
        topic_durations = []
        current_duration = 1
        
        for i, sim in enumerate(similarities):
            if sim < self.TOPIC_SHIFT_THRESHOLD:
                topic_shifts.append(i)
                topic_durations.append(current_duration)
                current_duration = 1
            else:
                current_duration += 1
        
        # Add final topic duration
        topic_durations.append(current_duration)
        
        features['topic_shift_count'] = len(topic_shifts)
        features['topic_shift_ratio'] = len(topic_shifts) / len(similarities) if similarities else 0.0
        
        # Abrupt shifts (very low similarity)
        abrupt_shifts = sum(1 for sim in similarities if sim < self.TOPIC_SHIFT_THRESHOLD / 2)
        features['abrupt_topic_shift_count'] = abrupt_shifts
        
        if topic_durations:
            features['avg_topic_duration_turns'] = float(np.mean(topic_durations))
        
        # Topic return detection (similar to earlier content)
        features['topic_return_count'] = self._detect_topic_returns(embeddings)
        
        return features
    
    def _detect_topic_returns(
        self,
        embeddings: List[Optional[np.ndarray]],
        lookback: int = 10
    ) -> int:
        """Detect when conversation returns to earlier topics."""
        returns = 0
        
        for i in range(lookback, len(embeddings)):
            current = embeddings[i]
            if current is None:
                continue
            
            # Check if similar to any earlier embedding (not immediately previous)
            for j in range(max(0, i - lookback), i - 2):
                earlier = embeddings[j]
                sim = self._cosine_similarity(current, earlier)
                
                if sim > 0.7:  # High similarity threshold for topic return
                    returns += 1
                    break
        
        return returns
    
    def _calculate_lda_features(
        self,
        all_texts: List[str],
        child_texts: List[str]
    ) -> Dict[str, float]:
        """
        Calculate LDA topic modeling features.
        
        Uses Latent Dirichlet Allocation to identify underlying topics
        and measure topic coherence.
        """
        features = {
            'topic_diversity': 0.0,
            'dominant_topic_ratio': 0.0,
            'topic_entropy': 0.0,
            'child_topic_consistency': 0.0,
        }
        
        if not LDA_AVAILABLE or len(all_texts) < self.n_topics:
            return features
        
        # Filter out empty texts
        valid_texts = [t for t in all_texts if len(t.split()) >= 2]
        
        if len(valid_texts) < self.n_topics:
            return features
        
        try:
            # Create document-term matrix
            vectorizer = CountVectorizer(
                max_df=0.95,
                min_df=1,
                stop_words='english',
                max_features=500
            )
            doc_term_matrix = vectorizer.fit_transform(valid_texts)
            
            if doc_term_matrix.shape[1] < self.n_topics:
                return features
            
            # Fit LDA
            lda = LatentDirichletAllocation(
                n_components=min(self.n_topics, doc_term_matrix.shape[1]),
                random_state=42,
                max_iter=10
            )
            topic_distributions = lda.fit_transform(doc_term_matrix)
            
            # Topic diversity (number of topics with significant presence)
            avg_distribution = np.mean(topic_distributions, axis=0)
            significant_topics = np.sum(avg_distribution > 0.1)
            features['topic_diversity'] = significant_topics / self.n_topics
            
            # Dominant topic ratio
            dominant_topics = np.argmax(topic_distributions, axis=1)
            topic_counts = Counter(dominant_topics)
            most_common_count = topic_counts.most_common(1)[0][1]
            features['dominant_topic_ratio'] = most_common_count / len(dominant_topics)
            
            # Topic entropy (distribution uniformity)
            avg_distribution = avg_distribution + 1e-10  # Avoid log(0)
            entropy = -np.sum(avg_distribution * np.log2(avg_distribution))
            max_entropy = np.log2(self.n_topics)
            features['topic_entropy'] = entropy / max_entropy if max_entropy > 0 else 0.0
            
            # Child topic consistency
            if child_texts:
                valid_child = [t for t in child_texts if len(t.split()) >= 2]
                if valid_child:
                    child_matrix = vectorizer.transform(valid_child)
                    child_topics = lda.transform(child_matrix)
                    child_dominant = np.argmax(child_topics, axis=1)
                    child_topic_counts = Counter(child_dominant)
                    if child_topic_counts:
                        most_common = child_topic_counts.most_common(1)[0][1]
                        features['child_topic_consistency'] = most_common / len(child_dominant)
            
        except Exception as e:
            logger.warning(f"LDA feature extraction failed: {e}")
        
        return features
    
    def _calculate_vocabulary_coherence(
        self,
        all_utterances: List[Utterance],
        child_utterances: List[Utterance]
    ) -> Dict[str, float]:
        """Calculate vocabulary-based coherence features."""
        features = {
            'lexical_overlap_mean': 0.0,
            'lexical_overlap_child': 0.0,
            'content_word_overlap': 0.0,
            'novel_word_ratio': 0.0,
        }
        
        # Function words to exclude for content word analysis
        function_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'because',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'about', 'from',
            'he', 'she', 'it', 'they', 'we', 'you', 'i', 'me', 'my', 'your',
            'this', 'that', 'these', 'those', 'what', 'who', 'which', 'where', 'when'
        }
        
        if len(all_utterances) < 2:
            return features
        
        # Calculate lexical overlaps
        all_overlaps = []
        child_overlaps = []
        content_overlaps = []
        
        for i in range(1, len(all_utterances)):
            prev_words = set(self._clean_text(all_utterances[i-1].text).split())
            curr_words = set(self._clean_text(all_utterances[i].text).split())
            
            if prev_words and curr_words:
                # Standard Jaccard overlap
                intersection = len(prev_words & curr_words)
                union = len(prev_words | curr_words)
                overlap = intersection / union if union > 0 else 0
                all_overlaps.append(overlap)
                
                # Content word overlap
                prev_content = prev_words - function_words
                curr_content = curr_words - function_words
                if prev_content and curr_content:
                    content_intersection = len(prev_content & curr_content)
                    content_union = len(prev_content | curr_content)
                    content_overlaps.append(content_intersection / content_union)
                
                # Track child-specific overlaps
                if all_utterances[i].speaker == 'CHI':
                    child_overlaps.append(overlap)
        
        if all_overlaps:
            features['lexical_overlap_mean'] = float(np.mean(all_overlaps))
        
        if child_overlaps:
            features['lexical_overlap_child'] = float(np.mean(child_overlaps))
        
        if content_overlaps:
            features['content_word_overlap'] = float(np.mean(content_overlaps))
        
        # Novel word ratio for child
        all_words_seen = set()
        novel_words = 0
        total_child_words = 0
        
        for u in all_utterances:
            words = set(self._clean_text(u.text).split())
            if u.speaker == 'CHI':
                novel = words - all_words_seen
                novel_words += len(novel)
                total_child_words += len(words)
            all_words_seen.update(words)
        
        features['novel_word_ratio'] = novel_words / total_child_words if total_child_words > 0 else 0.0
        
        return features
    
    def _calculate_contextual_appropriateness(
        self,
        utterances: List[Utterance]
    ) -> Dict[str, float]:
        """
        Calculate contextual appropriateness of responses.
        
        Determines if child responses are on-topic or tangential.
        """
        features = {
            'on_topic_response_ratio': 0.0,
            'off_topic_response_count': 0,
            'tangential_response_ratio': 0.0,
        }
        
        adult_codes = {'MOT', 'FAT', 'INV', 'INV1', 'INV2', 'EXA', 'EXP'}
        
        on_topic = 0
        off_topic = 0
        tangential = 0
        total_responses = 0
        
        for i in range(1, len(utterances)):
            if utterances[i].speaker == 'CHI' and utterances[i-1].speaker in adult_codes:
                total_responses += 1
                
                prev_text = self._clean_text(utterances[i-1].text)
                curr_text = self._clean_text(utterances[i].text)
                
                prev_emb = self._get_embedding(prev_text)
                curr_emb = self._get_embedding(curr_text)
                
                sim = self._cosine_similarity(prev_emb, curr_emb)
                
                if sim > 0.5:
                    on_topic += 1
                elif sim > 0.2:
                    tangential += 1
                else:
                    off_topic += 1
        
        if total_responses > 0:
            features['on_topic_response_ratio'] = on_topic / total_responses
            features['off_topic_response_count'] = off_topic
            features['tangential_response_ratio'] = tangential / total_responses
        
        return features


__all__ = ["TopicCoherenceFeatures"]






