"""
Shared Constants for Pragmatic Conversational Feature Extractors

All hardcoded patterns, thresholds, and vocabulary lists used across the
pragmatic_conversational feature extractors live here.  Centralising them
makes it easy to review, tune, or extend values without having to trace
through each individual extractor file.

Groups:
  - turn_taking thresholds        (turn_taking.py)
  - topic_coherence thresholds    (topic_coherence.py)
  - pause_latency patterns        (pause_latency.py)
  - repair_detection patterns     (repair_detection.py)
  - pragmatic_linguistic lists    (pragmatic_linguistic.py)

Author: Bimidu Gunathilake
"""


# ── turn_taking ────────────────────────────────────────────────────────────────

# A gap of <100 ms between turn-end and next turn-start is treated as
# simultaneous speech.  Values below ~100 ms fall within normal human
# reaction time and would produce excessive false positives.
OVERLAP_THRESHOLD_MS = 100

# A speaker change occurring within 500 ms of the previous speaker still
# talking is treated as an interruption rather than a smooth handoff.
INTERRUPTION_THRESHOLD_MS = 500

# Any inter-turn gap exceeding 1 s deviates from typical conversational
# flow and is flagged as a notable pause.
LONG_PAUSE_THRESHOLD_SEC = 1.0


# ── topic_coherence ────────────────────────────────────────────────────────────

# Cosine similarity below this value between adjacent utterance windows
# is classified as a topic shift.
TOPIC_SHIFT_THRESHOLD = 0.3

# Default number of latent topics for LDA modelling.
N_TOPICS_DEFAULT = 5

# Sliding window size (number of utterances) used for topic-shift detection.
WINDOW_SIZE = 3


# ── pause_latency ──────────────────────────────────────────────────────────────

# Filled pause (hesitation) markers, including CHAT &-prefixed annotations.
FILLED_PAUSE_PATTERNS = [
    r'\bum+\b', r'\buh+\b', r'\ber+\b', r'\bah+\b', r'\behm+\b',
    r'\bem+\b', r'\bmm+\b', r'\bhmm+\b', r'\buhm+\b', r'\bumm+\b',
    r'\bmhm+\b', r'\buh-huh\b', r'\buh huh\b',
    r'\bmmm+\b', r'\bhuh+\b',
    r'&-um', r'&-uh', r'&-er', r'&-ah', r'&-eh', r'&-hm',  # CHAT format
]

# CHAT pause marker symbols → approximate duration in seconds.
# Used to estimate pause length when audio timestamps are unavailable.
PAUSE_MARKERS = {
    '(.)': 0.5,      # Short pause (~0.5 sec)
    '(..)': 1.0,     # Medium pause (~1 sec)
    '(...)': 1.5,    # Long pause (~1.5 sec)
    '(pause)': 2.0,  # Extended pause
}

# Response-latency thresholds derived from GMM clustering on ASDBank data.
# See src/tools/ml_pause_clustering.py for the analysis that produced these.
#   Cluster 1 — Rapid      : mean ~0.2 s  → upper boundary → NORMAL_RESPONSE_TIME
#   Cluster 2 — Processing : mean ~1.25 s → upper boundary → LONG_PAUSE_THRESHOLD
#   Cluster 3 — Disengaged : mean ~4.3 s  → centre         → VERY_LONG_PAUSE_THRESHOLD
NORMAL_RESPONSE_TIME = 0.45
LONG_PAUSE_THRESHOLD = 2.00
VERY_LONG_PAUSE_THRESHOLD = 4.32


# ── repair_detection ───────────────────────────────────────────────────────────

# Lexical phrases a speaker uses to rephrase their own utterance mid-stream.
# ASD children repair less frequently and less effectively than typical peers.
SELF_REPAIR_PATTERNS = [
    r'\bi mean\b',
    r'\bno wait\b',
    r'\bsorry\b',
    r'\boh sorry\b',
    r'\bactually\b',
    r'\bno\s+i\s+mean\b',
    r'\bwell\s+not\b',
    r'\bor\s+rather\b',
    r'\blet me\s+rephrase\b',
    r'\blet me start over\b',
    r'\bwhat i mean is\b',
    r'\bi meant\b',
    r'\bi mean to say\b',
    r'\bi should say\b',
    r'\bi guess\b',
    r'\bwait\b',
    r'\bhold on\b',
    r'\bno no\b',
    r'\bnot that\b',
    r'\bthat is\b',
    r'\bthat was\b',
    r'\bor maybe\b',
    r'\bmore like\b',
    r'\bbetter said\b',
]

# Standard CLAN/CHAT symbols inserted by transcribers to annotate retraces
# and reformulations in .cha files.
CHAT_RETRACE_MARKERS = [
    r'\[/\]',      # Retrace without correction
    r'\[//\]',     # Retrace with correction
    r'\[///\]',    # Reformulation
    r'\[\?\]',     # Best guess
]

# Verbal signals that a listener did not understand and is requesting repair.
# Keep a high-precision core set plus child-style variants, then combine both.
CLARIFICATION_PATTERNS_CORE = [
    r'\bwhat(?:\?|\.|,)?\b',
    r'\bhuh(?:\?)?\b',
    r'\bpardon(?:\?)?\b',
    r'\bpardon me\b',
    r'\bexcuse me(?:\?)?\b',
    r'\bsay again\b',
    r'\bsay that again\b',
    r'\bcan you say that again\b',
    r'\bcould you say that again\b',
    r'\bcan you repeat(?: that)?\b',
    r'\bcould you repeat(?: that)?\b',
    r'\bcan you repeat that please\b',
    r'\bwhat did you\b',
    r'\bi don\'?t understand\b',
    r'\bi didn\'?t catch(?: that)?\b',
    r'\bi can\'?t hear you\b',
    r'\bi can\'?t hear that\b',
    r'\bwhat was that\b',
    r'\bwhat did you say\b',
    r'\bwhat do you mean\b',
    r'\bwhat do you mean by that\b',
    r'\bcould you explain\b',
    r'\bcan you explain(?: that)?\b',
    r'\bcan you clarify(?: that)?\b',
    r'\bcould you clarify(?: that)?\b',
    r'\bcome again\b',
    r'\bsorry(?:\?)?\b',
    r'\bsorry, what\b',
    r'\bexcuse me, what\b',
    r'\bplease repeat\b',
    r'\bplease say that again\b',
    r'\bwhat do you mean\b',
]

# Child-like clarification forms are often shorter, less grammatical,
# or fragment-based in spontaneous conversation.
CLARIFICATION_PATTERNS_CHILD_VARIANTS = [
    r'\beh(?:\?)?\b',
    r'\behh+\b',
    r'\bhm+(?:\?)?\b',
    r'\bmm+(?:\?)?\b',
    r'\bm(?:\?)?\b',
    r'\bagain(?:\?)?\b',
    r'\bcome again\b',
    r'\bwhat that(?:\?)?\b',
    r'\bwhat you said(?:\?)?\b',
    r'\bwhat this(?:\?)?\b',
    r'\bwhat mean(?:\?)?\b',
    r'\bwhat you mean(?:\?)?\b',
    r'\bwhat that mean(?:\?)?\b',
    r'\bno understand\b',
    r'\bdon\'?t know\b',
    r'\bdon\'?t get it\b',
    r'\bnot get(?:ting)? it\b',
    r'\bno get\b',
    r'\bi no understand\b',
    r'\bwhat\?\b',
    r'\bhuh\?\b',
    r'\bsorry\?\b',
]

# Backward-compatible name used by extractors.
CLARIFICATION_PATTERNS = (
    CLARIFICATION_PATTERNS_CORE + CLARIFICATION_PATTERNS_CHILD_VARIANTS
)

# Phrases used to confirm or echo back a paraphrase of what was just said.
CONFIRMATION_PATTERNS = [
    r'\bdo you mean\b',
    r'\bso you\b',
    r'\blike\s+a\b',
    r'\byou mean\b',
    r'\bis that\b',
    r'\bright(?:\?)?\b',
    r'\bokay(?:\?)?\b',
    r'\bcorrect(?:\?)?\b',
    r'\bdid you mean\b',
    r'\byou mean(?: that)?(?:\?)?\b',
    r'\bis this what you mean\b',
    r'\bis that what you mean\b',
    r'\bso you mean\b',
    r'\bso that means\b',
    r'\blet me check\b',
    r'\blet me make sure\b',
    r'\bif i understand (?:you|correctly)\b',
    r'\bif i got (?:it|you) right\b',
    r'\bam i right\b',
    r'\bis that right\b',
]

# Uptake markers that signal a listener has accepted a repair or understands.
ACKNOWLEDGMENT_PATTERNS = [
    r'\boh\b',
    r'\bi see\b',
    r'\bokay\b',
    r'\byes\b',
    r'\boh okay\b',
    r'\bah\b',
    r'\bgo on\b',
    r'\bi got it\b',
    r'\bok\b',
    r'\bokay then\b',
    r'\ball right\b',
    r'\bright\b',
    r'\byeah\b',
    r'\byeah okay\b',
    r'\byep\b',
    r'\buh huh\b',
    r'\bmhm\b',
    r'\bgot it\b',
    r'\bunderstood\b',
    r'\bthat makes sense\b',
    r'\bnow i get it\b',
    r'\bi understand\b',
    r'\bmm hm\b',
]


# ── Supporting · Pragmatic Linguistic ────────────────────────────────────────

# Conventional social phrases used to assess social communication competence.
SOCIAL_PHRASES = [
    'please', 'thank you', 'thanks', 'thank u', 'thanks a lot',
    'sorry', 'i am sorry', "i'm sorry", 'excuse me', 'pardon me',
    'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
    'bye', 'goodbye', 'see you', 'see you later', 'take care',
    'yes please', 'no thank you', 'you are welcome', "you're welcome",
    'welcome', 'nice to meet you', 'good to see you', 'after you',
    'may i', 'could i', 'can i please'
]

# WH-question words used to classify and count question types.
QUESTION_WORDS = [
    'what', 'where', 'when', 'who', 'why', 'how',
    'which', 'whose', 'whom', 'is', 'are', 'am', 'do', 'does', 'did',
    'can', 'could', 'will', 'would', 'should', 'may', 'might', 'have', 'has', 'had'
]

# Discourse markers grouped by conversational function.
# Hesitation markers are minimal here; detailed pause patterns live in
# FILLED_PAUSE_PATTERNS (pause_latency).
DISCOURSE_MARKERS = {
    'topic_intro': [
        'so', 'well', 'anyway', 'by the way', 'now', 'okay so', 'first',
        'to start', 'let us talk about', 'about', 'speaking of'
    ],
    'topic_continuation': [
        'and', 'also', 'too', 'then', 'then again', 'next', 'after that',
        'besides', 'moreover', 'in addition', 'plus', 'as well'
    ],
    'acknowledgment': [
        'okay', 'ok', 'yeah', 'yes', 'mhm', 'uh huh', 'right', 'got it',
        'i see', 'understood', 'alright', 'hmm'
    ],
    'hesitation': ['um', 'uh', 'er', 'ah', 'eh', 'hmm', 'mm', 'uhm', 'umm'],
}

# Non-verbal and paralinguistic event markers from CHAT &= notation.
BEHAVIORAL_MARKERS = [
    '&=laughs', '&=cries', '&=screams', '&=sighs',
    '&=gasps', '&=whispers', '&=hums', '&=sings',
    '&=squeals', '&=yells', '&=breathes', '&=groans',
    '&=claps', '&=points', '&=nods',
    '&=coughs', '&=sneezes', '&=sniffs', '&=clears_throat',
    '&=giggles', '&=chuckles', '&=laughing', '&=crying',
    '&=shouts', '&=mumbles', '&=mutters', '&=stutters',
    '&=whines', '&=snorts', '&=pants', '&=grunts',
    '&=taps', '&=stomps', '&=shrugs', '&=shakes_head', '&=smiles'
]
