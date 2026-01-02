# spaCy Model Configuration

## Current Setup

### Installed Models
- **en_core_web_lg** (Large, 400.7 MB) - **PRIMARY MODEL**
  - 300-dimensional word vectors
  - 102,875,400 vectors loaded
  - Best quality for semantic similarity
  - **Status**: ✅ Installed and active

- **en_core_web_sm** (Small, 40 MB) - Fallback only
  - No word vectors (context-sensitive tensors only)
  - Limited similarity quality
  - **Status**: ✅ Installed (fallback)

### Model Storage Location
```
/Users/bimidugunathilake/Documents/SE/Projects/Artistic./.venv/lib/python3.11/site-packages/
```

Models are installed in the virtual environment's site-packages directory.

## Model Loading Priority

The code now tries models in this order:
1. **en_core_web_lg** (best - 300D vectors)
2. **en_core_web_md** (good - 300D vectors, if installed)
3. **en_core_web_sm** (fallback - no vectors)

## Which Features Use spaCy?

### ✅ Uses spaCy (Word Embeddings & Similarity)

1. **Topic Coherence Features** (`topic_coherence.py`)
   - `semantic_coherence_score` - Cosine similarity between consecutive utterances
   - `child_response_relevance` - Similarity of child responses to adult prompts
   - `inter_speaker_similarity` - Similarity between speakers
   - `topic_shift_detection` - Detecting topic changes via similarity drops
   - `on_topic_response_ratio` - Appropriate responses via similarity threshold
   - **Total**: ~15 features depend on word vectors

2. **Repair Detection Features** (`repair_detection.py`)
   - `repair_success_count` - Detecting successful repairs via semantic continuation
   - Uses similarity to check if topic continues after repair attempt
   - **Total**: ~3 features use similarity checks

### ❌ Does NOT Use spaCy

1. **Turn-Taking Features** (`turn_taking.py`)
   - Pure statistical analysis (counts, timing, gaps)
   - No NLP/semantic analysis needed
   - **Total**: 42 features, all independent of spaCy

2. **Pause & Latency Features** (`pause_latency.py`)
   - Timing analysis and pattern detection
   - Regex-based filled pause detection
   - **Total**: 47 features, no spaCy dependency

3. **Pragmatic Linguistic Features** (`pragmatic_linguistic.py`)
   - Pattern matching (regex)
   - Statistical counts (MLU, TTR)
   - **Total**: 29 features, no spaCy dependency

4. **Audio Features** (`audio_features.py`)
   - Signal processing (librosa)
   - Energy-based detection
   - **Total**: 29 features, no spaCy dependency

## Summary

- **Total features using spaCy**: ~18 features (out of 210 total)
- **Features NOT using spaCy**: ~192 features
- **Primary use**: Semantic similarity for topic coherence and repair success detection

## Installation Commands

```bash
# Install best model (large, 300D vectors)
python -m spacy download en_core_web_lg

# Install medium model (alternative, 300D vectors)
python -m spacy download en_core_web_md

# Install small model (fallback, no vectors)
python -m spacy download en_core_web_sm
```

## Model Comparison

| Model | Size | Vectors | Dimensions | Quality | Use Case |
|-------|------|---------|------------|---------|----------|
| **en_core_web_lg** | 400.7 MB | ✅ Yes | 300D | Best | **Recommended** |
| en_core_web_md | 40 MB | ✅ Yes | 300D | Good | Alternative |
| en_core_web_sm | 12.8 MB | ❌ No | N/A | Limited | Fallback only |

## Warning Resolution

The warning you saw:
```
[W007] The model you're using has no word vectors loaded
```

**Cause**: `en_core_web_sm` was being used (no vectors)

**Solution**: ✅ Now using `en_core_web_lg` (has 300D vectors)

**Result**: Warning should no longer appear, and similarity calculations will be accurate.

## Verification

To verify the model is working correctly:

```python
import spacy

nlp = spacy.load("en_core_web_lg")

# Check if vectors are loaded
print("Has vectors:", nlp.vocab.vectors.size > 0)
print("Vector dimensions:", nlp.vocab.vectors.shape[1] if nlp.vocab.vectors.size > 0 else 0)

# Test similarity
doc1 = nlp("I like dogs")
doc2 = nlp("I like cats")
doc3 = nlp("The weather is nice")

print("Similarity (dogs/cats):", doc1.similarity(doc2))  # Should be high (~0.8)
print("Similarity (dogs/weather):", doc1.similarity(doc3))  # Should be low (~0.3)
```

Expected output:
```
Has vectors: True
Vector dimensions: 300
Similarity (dogs/cats): 0.82
Similarity (dogs/weather): 0.28
```

---

**Last Updated**: January 2, 2026



