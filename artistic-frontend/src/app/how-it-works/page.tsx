'use client';
import { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';

interface Feature {
  name: string;
  label: string;
  description: string;
  method: string;
  asdRelevance: string;
}

type IconProps = {
  className?: string;
};

function IconMic({ className }: IconProps) {
  return (
    <svg
      viewBox="0 0 24 24"
      aria-hidden="true"
      className={className}
    >
      <rect x="9" y="3" width="6" height="12" rx="3" className="fill-current" />
      <path d="M5 11a7 7 0 0014 0" className="stroke-current" strokeWidth="1.5" fill="none" />
      <path d="M12 18v3" className="stroke-current" strokeWidth="1.5" fill="none" />
      <path d="M8 21h8" className="stroke-current" strokeWidth="1.5" fill="none" />
    </svg>
  );
}

function IconTranscript({ className }: IconProps) {
  return (
    <svg
      viewBox="0 0 24 24"
      aria-hidden="true"
      className={className}
    >
      <rect x="5" y="3" width="14" height="18" rx="2" className="stroke-current" strokeWidth="1.5" fill="none" />
      <path d="M8 8h8M8 12h5M8 16h6" className="stroke-current" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

function IconChart({ className }: IconProps) {
  return (
    <svg
      viewBox="0 0 24 24"
      aria-hidden="true"
      className={className}
    >
      <path d="M5 19V9" className="stroke-current" strokeWidth="1.5" strokeLinecap="round" />
      <path d="M11 19V5" className="stroke-current" strokeWidth="1.5" strokeLinecap="round" />
      <path d="M17 19v-7" className="stroke-current" strokeWidth="1.5" strokeLinecap="round" />
      <path d="M4 19h16" className="stroke-current" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

function IconCounterfactual({ className }: IconProps) {
  return (
    <svg
      viewBox="0 0 24 24"
      aria-hidden="true"
      className={className}
    >
      <path
        d="M7 7h7a4 4 0 014 4v1"
        className="stroke-current"
        strokeWidth="1.5"
        strokeLinecap="round"
        fill="none"
      />
      <path
        d="M9 5l-2 2 2 2"
        className="stroke-current"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />
      <path
        d="M17 17h-7a4 4 0 01-4-4v-1"
        className="stroke-current"
        strokeWidth="1.5"
        strokeLinecap="round"
        fill="none"
      />
      <path
        d="M15 15l2 2-2 2"
        className="stroke-current"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />
    </svg>
  );
}

function IconNote({ className }: IconProps) {
  return (
    <svg
      viewBox="0 0 24 24"
      aria-hidden="true"
      className={className}
    >
      <rect x="5" y="3" width="14" height="18" rx="2" className="stroke-current" strokeWidth="1.5" fill="none" />
      <path d="M8 8h8M8 12h4" className="stroke-current" strokeWidth="1.5" strokeLinecap="round" />
      <path
        d="M14.5 15.5l3-3"
        className="stroke-current"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
      <path
        d="M13.5 17.5l1-2 2-2 1.5 1.5-2 2-2 1z"
        className="stroke-current"
        strokeWidth="1.2"
        fill="none"
      />
    </svg>
  );
}

function IconWarning({ className }: IconProps) {
  return (
    <svg
      viewBox="0 0 24 24"
      aria-hidden="true"
      className={className}
    >
      <path
        d="M12 3L3 21h18L12 3z"
        className="stroke-current"
        strokeWidth="1.5"
        fill="none"
        strokeLinejoin="round"
      />
      <path
        d="M12 9v5"
        className="stroke-current"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
      <circle cx="12" cy="16.5" r="0.8" className="fill-current" />
    </svg>
  );
}

function IconTurnTaking({ className }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className={className}>
      <path d="M5 8h9" className="stroke-current" strokeWidth="1.5" strokeLinecap="round" />
      <path d="M9 5l3 3-3 3" className="stroke-current" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M19 16h-9" className="stroke-current" strokeWidth="1.5" strokeLinecap="round" />
      <path d="M15 19l-3-3 3-3" className="stroke-current" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function IconTopic({ className }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className={className}>
      <circle cx="12" cy="12" r="4" className="stroke-current" strokeWidth="1.5" fill="none" />
      <circle cx="6" cy="8" r="2" className="stroke-current" strokeWidth="1.5" fill="none" />
      <circle cx="18" cy="8" r="2" className="stroke-current" strokeWidth="1.5" fill="none" />
      <circle cx="8" cy="18" r="2" className="stroke-current" strokeWidth="1.5" fill="none" />
      <circle cx="16" cy="18" r="2" className="stroke-current" strokeWidth="1.5" fill="none" />
      <path d="M8 10.5L10 11.5M16 10.5L14 11.5M9 16L10.5 14.5M15 16L13.5 14.5" className="stroke-current" strokeWidth="1.3" />
    </svg>
  );
}

function IconClock({ className }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className={className}>
      <circle cx="12" cy="12" r="7" className="stroke-current" strokeWidth="1.5" fill="none" />
      <path d="M12 9v4l2.5 1.5" className="stroke-current" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function IconRepair({ className }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className={className}>
      <path
        d="M7 7l3 3M11 3l-1.5 1.5a3 3 0 004.24 4.24L15.5 7"
        className="stroke-current"
        strokeWidth="1.5"
        fill="none"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M7 13l-1.5 1.5a2.5 2.5 0 003.54 3.54L10.5 17"
        className="stroke-current"
        strokeWidth="1.5"
        fill="none"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M14 14l3 3"
        className="stroke-current"
        strokeWidth="1.5"
        fill="none"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function IconBubble({ className }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className={className}>
      <path
        d="M5 7a3 3 0 013-3h8a3 3 0 013 3v6a3 3 0 01-3 3h-4l-3.5 3L9 16H8a3 3 0 01-3-3V7z"
        className="stroke-current"
        strokeWidth="1.5"
        fill="none"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path d="M9 9h6" className="stroke-current" strokeWidth="1.5" strokeLinecap="round" />
      <path d="M9 12h3.5" className="stroke-current" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

function IconWaveform({ className }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className={className}>
      <path d="M4 13v-2" className="stroke-current" strokeWidth="1.5" strokeLinecap="round" />
      <path d="M8 17V7" className="stroke-current" strokeWidth="1.5" strokeLinecap="round" />
      <path d="M12 19V5" className="stroke-current" strokeWidth="1.5" strokeLinecap="round" />
      <path d="M16 17V7" className="stroke-current" strokeWidth="1.5" strokeLinecap="round" />
      <path d="M20 13v-2" className="stroke-current" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

function PragmaticIcon({ id, className }: { id: string; className?: string }) {
  switch (id) {
    case 'turn_taking':
      return <IconTurnTaking className={className} />;
    case 'topic_coherence':
      return <IconTopic className={className} />;
    case 'pause_latency':
      return <IconClock className={className} />;
    case 'repair_detection':
      return <IconRepair className={className} />;
    case 'pragmatic_linguistic':
      return <IconBubble className={className} />;
    case 'audio_pragmatic':
      return <IconWaveform className={className} />;
    default:
      return <IconBubble className={className} />;
  }
}
// ── ACOUSTIC ICONS ───────────────────────────────────────────────────────────
function IconPitch({ className }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className={className}>
      <path d="M3 14 Q6 6 9 12 Q12 18 15 10 Q18 4 21 10" className="stroke-current" strokeWidth="1.5" fill="none" strokeLinecap="round" />
    </svg>
  );
}

function IconSpectral({ className }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className={className}>
      <path d="M3 18 L6 14 L9 10 L12 6 L15 10 L18 14 L21 18" className="stroke-current" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M3 18h18" className="stroke-current" strokeWidth="1" strokeLinecap="round" />
    </svg>
  );
}

function IconMFCC({ className }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className={className}>
      <rect x="3" y="14" width="3" height="7" rx="0.5" className="fill-current opacity-40" />
      <rect x="7" y="10" width="3" height="11" rx="0.5" className="fill-current opacity-60" />
      <rect x="11" y="6" width="3" height="15" rx="0.5" className="fill-current opacity-80" />
      <rect x="15" y="9" width="3" height="12" rx="0.5" className="fill-current opacity-60" />
      <rect x="19" y="13" width="3" height="8" rx="0.5" className="fill-current opacity-40" />
    </svg>
  );
}

function IconVoice({ className }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className={className}>
      <circle cx="12" cy="8" r="4" className="stroke-current" strokeWidth="1.5" fill="none" />
      <path d="M6 21v-1a6 6 0 0112 0v1" className="stroke-current" strokeWidth="1.5" fill="none" strokeLinecap="round" />
      <path d="M18 11.5a2.5 2.5 0 010 5" className="stroke-current" strokeWidth="1.5" fill="none" strokeLinecap="round" />
    </svg>
  );
}

function IconEnergy({ className }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className={className}>
      <path d="M13 2L4 14h7l-1 8 9-12h-7l2-8z" className="stroke-current" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function IconRhythm({ className }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className={className}>
      <path d="M3 12h3l2-6 4 12 2-8 2 4 2-2h3" className="stroke-current" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function IconFormant({ className }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className={className}>
      <ellipse cx="9" cy="12" rx="3" ry="5" className="stroke-current" strokeWidth="1.5" fill="none" />
      <ellipse cx="16" cy="10" rx="2.5" ry="4" className="stroke-current" strokeWidth="1.5" fill="none" />
      <path d="M3 20h18" className="stroke-current" strokeWidth="1" strokeLinecap="round" />
    </svg>
  );
}

function IconChroma({ className }: IconProps) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className={className}>
      <circle cx="12" cy="12" r="8" className="stroke-current" strokeWidth="1.5" fill="none" />
      <path d="M12 4v16M4 12h16M6.3 6.3l11.4 11.4M17.7 6.3L6.3 17.7" className="stroke-current" strokeWidth="1" strokeLinecap="round" />
    </svg>
  );
}
function AcousticIcon({ id, className }: { id: string; className?: string }) {
  switch (id) {
    case 'pitch': return <IconPitch className={className} />;
    case 'mfcc': return <IconMFCC className={className} />;
    case 'spectral': return <IconSpectral className={className} />;
    case 'voice_quality': return <IconVoice className={className} />;
    case 'formants': return <IconFormant className={className} />;
    case 'energy': return <IconEnergy className={className} />;
    case 'rhythm': return <IconRhythm className={className} />;
    case 'chroma': return <IconChroma className={className} />;
    default: return <IconWaveform className={className} />;
  }
}
// ── TURN-TAKING (44 features) ────────────────────────────────────────────────
const TURN_TAKING_FEATURES: Feature[] = [
  { name: 'total_turns', label: 'Total Conversation Turns', description: 'The total number of back-and-forth exchanges between all speakers during the recording.', method: 'Counted directly from transcript speaker labels in the CHAT file', asdRelevance: 'Sets the baseline length and richness of the interaction for normalizing other features.' },
  { name: 'child_turns', label: "Child's Turn Count", description: 'How many times the child took a turn to speak during the session.', method: 'Count of utterances with child speaker code (e.g. CHI) in CHAT file', asdRelevance: 'Children with ASD often take fewer turns, reflecting reduced engagement in back-and-forth exchange.' },
  { name: 'adult_turns', label: "Adult's Turn Count", description: 'How many times the adult (parent, clinician, or examiner) took a turn to speak.', method: 'Count of utterances with adult speaker codes (e.g. MOT, FAT, INV)', asdRelevance: 'Used alongside child turns to understand the conversational balance.' },
  { name: 'turns_per_minute', label: 'Conversation Pace', description: 'The average number of conversational turns happening per minute. A higher number means a faster, more fluid back-and-forth.', method: 'Total turns divided by total recording duration in minutes', asdRelevance: 'Slower pace may indicate processing delays or reduced reciprocal engagement.' },
  { name: 'child_turn_ratio', label: "Child's Share of Conversation", description: "What fraction of all turns in the conversation belong to the child — how much of the conversation the child is actively participating in.", method: "Child turns divided by total turns", asdRelevance: 'Children with ASD often contribute a smaller share of turns, indicating passive participation.' },
  { name: 'avg_turn_length_words', label: 'Average Turn Length', description: 'On average, how many words are spoken per turn, across all speakers.', method: 'Mean word count per utterance across all speakers', asdRelevance: 'Reflects overall verbosity and linguistic output in the session.' },
  { name: 'avg_child_turn_length', label: "Child's Average Turn Length", description: "On average, how many words the child uses each time they speak.", method: 'Mean word count per child utterance', asdRelevance: 'Shorter turns may reflect limited expressive language or difficulty sustaining discourse.' },
  { name: 'avg_adult_turn_length', label: "Adult's Average Turn Length", description: 'On average, how many words the adult uses per turn.', method: 'Mean word count per adult utterance', asdRelevance: 'Contrasting adult and child turn lengths reveals communication asymmetry.' },
  { name: 'max_child_turn_length', label: "Child's Longest Turn", description: "The most words the child used in any single turn during the session.", method: 'Maximum word count across all child utterances', asdRelevance: 'Peak linguistic output; reveals capacity vs. typical performance.' },
  { name: 'min_child_turn_length', label: "Child's Shortest Turn", description: 'The fewest words the child used in any single turn — often just one or two words, or a minimal response like "yes".', method: 'Minimum word count across all child utterances', asdRelevance: 'Very short responses may indicate avoidance or minimal engagement.' },
  { name: 'child_turn_length_std', label: "Variability in Child's Turn Length", description: "How much the child's turn lengths vary — do they consistently say about the same amount, or does it swing widely?", method: 'Standard deviation of word counts across child utterances', asdRelevance: 'High variability may indicate inconsistent engagement or topic-dependent verbosity.' },
  { name: 'child_turn_length_cv', label: "Consistency of Child's Turns (Normalised)", description: "A normalised measure of how consistently the child uses similar amounts of words each turn, regardless of their average length.", method: 'Standard deviation divided by mean of child turn word counts (coefficient of variation)', asdRelevance: 'Controls for overall verbosity; highlights inconsistency as an independent signal.' },
  { name: 'adult_turn_length_std', label: "Variability in Adult's Turn Length", description: "How much the adult's turn lengths vary throughout the conversation.", method: 'Standard deviation of word counts across adult utterances', asdRelevance: 'Adults may adjust turn length in response to the child; high variability reflects scaffolding behavior.' },
  { name: 'avg_turn_duration_sec', label: 'Average Turn Duration', description: 'On average, how many seconds each turn lasts, using timing information from the audio or CHAT timestamps.', method: 'Mean of (turn_end_time minus turn_start_time) in seconds from %snd tier or Whisper timestamps', asdRelevance: 'Duration complements word count; slow articulation may inflate duration relative to words.' },
  { name: 'child_turn_duration_mean', label: "Child's Average Speaking Time per Turn", description: "On average, how long in seconds the child is speaking each time they take a turn.", method: 'Mean duration of child utterance segments from timing annotations', asdRelevance: "Combines with word count to estimate speech rate, relevant for prosodic assessment." },
  { name: 'child_turn_duration_std', label: "Variability in Child's Speaking Duration", description: "How much the length of time the child spends speaking each turn varies.", method: 'Standard deviation of child utterance durations', asdRelevance: 'High duration variability may indicate inconsistent engagement across topics.' },
  { name: 'inter_turn_gap_mean', label: 'Average Gap Between Turns', description: "The average amount of time (in seconds) that passes between one speaker finishing and the next speaker beginning.", method: 'Mean of (next_turn_start minus previous_turn_end) from timing data', asdRelevance: 'Larger gaps indicate processing delays or difficulty with turn-taking cues.' },
  { name: 'inter_turn_gap_median', label: 'Typical Gap Between Turns', description: 'The middle value of all gaps between turns — less affected by unusually long silences than the average.', method: 'Median of inter-turn gap durations', asdRelevance: 'More robust measure of typical response timing; children with ASD often show elevated medians.' },
  { name: 'inter_turn_gap_std', label: 'Inconsistency in Turn Gaps', description: 'How much the gaps between turns vary — does the conversation flow at a steady rhythm, or are some gaps very short and others very long?', method: 'Standard deviation of inter-turn gap durations', asdRelevance: 'High variability suggests unpredictable response timing, a known ASD marker.' },
  { name: 'inter_turn_gap_max', label: 'Longest Gap in Conversation', description: 'The single longest silence that occurred between turns during the entire session.', method: 'Maximum value across all inter-turn gap durations', asdRelevance: 'Very long gaps may indicate disengagement or communication breakdown.' },
  { name: 'child_response_latency_mean', label: "Child's Average Response Delay", description: "On average, how long it takes the child to start speaking after the adult finishes their turn.", method: 'Mean gap from adult turn end to next child turn start', asdRelevance: 'A key ASD indicator — delayed responses reflect difficulty processing conversational cues.' },
  { name: 'child_response_latency_std', label: "Variability in Child's Response Delays", description: "How consistently (or inconsistently) the child responds to the adult's turns.", method: "Standard deviation of child response latencies", asdRelevance: 'Inconsistent latency may indicate variable attention or context-dependent processing.' },
  { name: 'adult_response_latency_mean', label: "Adult's Average Response Delay", description: "On average, how long it takes the adult to respond to the child.", method: 'Mean gap from child turn end to next adult turn start', asdRelevance: 'Used as a baseline; adults typically respond faster and more consistently.' },
  { name: 'long_pause_count', label: 'Number of Long Pauses', description: 'How many times a pause of more than 2 seconds occurred between turns during the conversation.', method: 'Count of inter-turn gaps > 2.0 seconds (GMM-derived threshold from ASDBank data)', asdRelevance: 'Frequent long pauses are strongly associated with ASD, reflecting processing or engagement difficulties.' },
  { name: 'long_pause_ratio', label: 'Proportion of Long Pauses', description: 'What fraction of all turn transitions involved a pause longer than 2 seconds.', method: 'Long pause count divided by total turn transitions', asdRelevance: 'Normalises for conversation length; even in short sessions, a high ratio is diagnostically relevant.' },
  { name: 'overlap_count', label: 'Speaking Overlaps', description: 'How many times both speakers were talking at the same time during the conversation.', method: 'Count of utterances where timing intervals from speaker A and speaker B overlap', asdRelevance: 'Children with ASD may have fewer overlaps (reduced simultaneous engagement) or more (difficulty reading turn-end cues).' },
  { name: 'overlap_duration_total', label: 'Total Overlap Duration', description: 'The total number of seconds during which both speakers were talking simultaneously.', method: 'Sum of all overlapping time intervals between speaker utterances', asdRelevance: 'High total overlap may indicate difficulty recognising turn-yielding cues.' },
  { name: 'overlap_ratio', label: 'Proportion of Overlapping Speech', description: 'What fraction of the total conversation time involves simultaneous speech.', method: 'Total overlap duration divided by total conversation duration', asdRelevance: 'A high overlap ratio reflects difficulties in following conversational turn-taking conventions.' },
  { name: 'child_overlaps_adult_count', label: 'Times Child Talks Over Adult', description: 'How many times the child started speaking before the adult had finished their turn.', method: 'Count of child utterances whose start time overlaps with the end of the preceding adult utterance', asdRelevance: 'May reflect impulsivity, difficulty reading turn-end signals, or echolalic responses.' },
  { name: 'adult_overlaps_child_count', label: 'Times Adult Talks Over Child', description: 'How many times the adult started speaking before the child had finished.', method: 'Count of adult utterances overlapping with prior child utterances', asdRelevance: "May indicate the adult's scaffolding behavior when the child pauses mid-utterance." },
  { name: 'interruption_count', label: 'Total Interruptions', description: 'The total number of times either speaker broke in and stopped the other mid-turn.', method: 'Count of turn starts occurring before >80% of the prior utterance duration has elapsed', asdRelevance: 'Both high and low interruption rates can be ASD-relevant depending on context.' },
  { name: 'child_interruption_count', label: "Child's Interruptions", description: 'How many times the child interrupted the adult.', method: 'Count of child turn starts during active adult utterances', asdRelevance: 'Impulsive interruption may indicate difficulty with inhibition or conversational cue-reading.' },
  { name: 'adult_interruption_count', label: "Adult's Interruptions", description: 'How many times the adult interrupted the child.', method: 'Count of adult turn starts during active child utterances', asdRelevance: 'High adult interruptions may reflect the adult filling gaps due to child communication difficulties.' },
  { name: 'interruption_ratio', label: 'Proportion of Turns with Interruptions', description: 'What fraction of all turns involved an interruption.', method: 'Interruption count divided by total turn count', asdRelevance: 'Normalises interruption behaviour for session length.' },
  { name: 'child_initiated_turns', label: 'Turns Child Started', description: 'How many conversation turns the child initiated themselves, rather than just responding to the adult.', method: 'Count of child utterances following a pause above threshold after the prior adult turn ended', asdRelevance: 'Spontaneous initiation is a key pragmatic skill; low counts indicate passive communication style.' },
  { name: 'adult_initiated_turns', label: 'Turns Adult Started', description: 'How many turns the adult initiated — typically prompts, questions, or new topics introduced by the adult.', method: 'Count of adult utterances following a silence (not a direct response)', asdRelevance: 'A high adult initiation ratio suggests the child rarely initiates, relying on adult prompts.' },
  { name: 'child_initiation_ratio', label: "Child's Initiation Share", description: "What fraction of conversation-starting turns belong to the child — how often they introduce a new topic or start a new exchange.", method: 'Child-initiated turns divided by (child-initiated + adult-initiated turns)', asdRelevance: 'A low ratio is strongly associated with ASD — children often wait for or rely on adult prompting.' },
  { name: 'turn_switches', label: 'Speaker Changes', description: 'The total number of times the conversation switched from one speaker to the other.', method: 'Count of consecutive utterances with different speaker codes', asdRelevance: 'Frequent switching indicates reciprocal, back-and-forth interaction.' },
  { name: 'avg_turns_before_switch', label: 'Average Turns Before Speaker Switches', description: 'On average, how many turns one speaker takes before the other gets to speak.', method: 'Total turns divided by number of speaker switches', asdRelevance: 'High values indicate one-sided discourse (monologuing); a key ASD marker.' },
  { name: 'turn_switch_rate', label: 'Rate of Speaker Switching', description: 'How frequently speaker changes happen per minute.', method: 'Turn switches divided by total recording duration in minutes', asdRelevance: 'Low switch rate correlates with monologue-style communication patterns common in ASD.' },
  { name: 'max_consecutive_child_turns', label: "Child's Longest Monologue Run", description: "The most turns in a row that the child took without the adult getting a word in.", method: 'Maximum count of consecutive child utterances before an adult utterance appears', asdRelevance: 'Long monologues, especially on restricted topics, are a hallmark ASD communication pattern.' },
  { name: 'max_consecutive_adult_turns', label: "Adult's Longest Consecutive Turns", description: 'The longest run of consecutive adult turns without the child responding.', method: 'Maximum count of consecutive adult utterances before a child utterance', asdRelevance: "High values suggest the child is not responding, possibly due to disengagement or processing difficulty." },
  { name: 'child_monologue_ratio', label: "Child's Monologue Proportion", description: "What fraction of the conversation consists of the child speaking multiple turns in a row without the adult getting a turn.", method: 'Proportion of child turns that are part of consecutive child turn sequences of length 3 or more', asdRelevance: 'Elevated monologue ratios are associated with restricted interests and one-sided discourse in ASD.' },
];

// ── TOPIC COHERENCE (31 features) ───────────────────────────────────────────
const TOPIC_COHERENCE_FEATURES: Feature[] = [
  { name: 'semantic_coherence_score', label: 'Overall Topic Consistency Score', description: 'A score measuring how related consecutive utterances are in meaning — does the conversation stay on topic from one turn to the next?', method: 'Mean cosine similarity between spaCy word vector representations of consecutive utterances', asdRelevance: 'Lower coherence scores indicate frequent topic jumps, a common pattern in ASD conversation.' },
  { name: 'semantic_coherence_std', label: 'Topic Consistency Variability', description: 'How much the topic-relatedness between consecutive turns varies throughout the conversation.', method: 'Standard deviation of pairwise cosine similarity scores between consecutive utterances', asdRelevance: 'High variability indicates unpredictable topic shifts — some exchanges are coherent, others are not.' },
  { name: 'min_semantic_similarity', label: 'Most Disconnected Exchange', description: 'The lowest coherence score observed — the pair of consecutive utterances that were least related to each other in the entire session.', method: 'Minimum cosine similarity between any two consecutive utterances', asdRelevance: 'Captures the worst-case topic jump, which may indicate a sudden diversion to a restricted interest.' },
  { name: 'max_semantic_similarity', label: 'Most Coherent Exchange', description: 'The highest coherence score — the pair of consecutive utterances that were most closely related.', method: 'Maximum cosine similarity between any two consecutive utterances', asdRelevance: 'Contextualises the minimum; helps distinguish general low coherence from occasional breakdowns.' },
  { name: 'child_semantic_coherence', label: "Child's Topic Consistency", description: "How well the child's utterances relate to what was said just before them.", method: 'Mean cosine similarity of child utterances to their immediately preceding utterance', asdRelevance: "Isolates the child's contribution to topic coherence, a core pragmatic skill." },
  { name: 'child_response_relevance', label: "How Relevant the Child's Responses Are", description: "On average, how relevant the child's response is to the adult's preceding turn — not just any utterance, but specifically after the adult speaks.", method: "Mean cosine similarity between each child utterance and the immediately preceding adult utterance", asdRelevance: 'Directly measures topical responsiveness — a key pragmatic skill often impaired in ASD.' },
  { name: 'child_response_relevance_std', label: "Variability in Child's Response Relevance", description: "How consistently (or inconsistently) relevant the child's responses are to the adult's turns.", method: "Standard deviation of child-to-adult cosine similarity scores", asdRelevance: 'Inconsistent relevance may indicate topic-specific engagement or intermittent attention.' },
  { name: 'inter_speaker_similarity_mean', label: 'How Well Speakers Match Topics', description: 'On average, how similar in meaning the turns of different speakers are — a measure of mutual engagement on shared topics.', method: 'Mean cosine similarity between alternating adult and child utterances', asdRelevance: 'Low inter-speaker similarity suggests parallel rather than shared conversation.' },
  { name: 'inter_speaker_similarity_std', label: 'Variability in Speaker Topic Matching', description: 'How much topic similarity between speakers varies throughout the session.', method: 'Standard deviation of inter-speaker cosine similarity across all turn transitions', asdRelevance: 'Indicates whether topic alignment is consistently poor or only breaks down in certain contexts.' },
  { name: 'child_to_adult_similarity', label: "Child Following Adult's Topics", description: "How similar the child's utterances are to the adult's preceding turns — how well the child follows the adult's conversational lead.", method: "Mean cosine similarity between child utterances and the preceding adult utterances", asdRelevance: "Measures child's ability to stay on topic when prompted — often reduced in ASD." },
  { name: 'adult_to_child_similarity', label: "Adult Following Child's Topics", description: "How similar the adult's utterances are to what the child just said — how often the adult builds on the child's topics.", method: "Mean cosine similarity between adult utterances and the preceding child utterances", asdRelevance: 'High values indicate the adult is scaffolding around the child; may reflect child topic inflexibility.' },
  { name: 'child_within_consistency', label: "Consistency of Child's Own Topics", description: "How consistent the child's own consecutive turns are in topic — are they talking about the same thing from one of their turns to the next?", method: "Mean cosine similarity between consecutive child utterances (same-speaker pairs only)", asdRelevance: "High consistency combined with low inter-speaker similarity suggests restricted and repetitive topic focus." },
  { name: 'adult_within_consistency', label: "Consistency of Adult's Own Topics", description: "How consistent the adult's own consecutive turns are — used as a baseline for comparison.", method: "Mean cosine similarity between consecutive adult utterances", asdRelevance: 'Adults typically show higher within-speaker consistency; lower child values are diagnostically meaningful.' },
  { name: 'child_topic_drift', label: "How Much Child's Topics Drift Over Time", description: "A measure of how much the child's topic changes over the course of the session — do they stay on one thing or wander unpredictably?", method: "Negative correlation of cosine similarity with time index across child utterances", asdRelevance: "Increasing drift over time may indicate reducing engagement or shifting into restricted interests." },
  { name: 'topic_shift_count', label: 'Number of Topic Changes', description: 'How many times the conversation shifted to a noticeably different topic.', method: 'Count of consecutive utterance pairs with cosine similarity below 0.30', asdRelevance: 'Frequent topic changes indicate difficulty maintaining conversational thread.' },
  { name: 'topic_shift_ratio', label: 'Proportion of Topic Changes', description: 'What fraction of all turn transitions involved a topic shift.', method: 'Topic shift count divided by total consecutive utterance pairs', asdRelevance: 'Normalised measure of topic maintenance difficulty.' },
  { name: 'abrupt_topic_shift_count', label: 'Number of Abrupt Topic Changes', description: 'How many times the conversation jumped very suddenly to a completely unrelated topic — not just a gentle shift but a hard cut.', method: 'Count of consecutive utterance pairs with cosine similarity below 0.15 (abrupt threshold)', asdRelevance: 'Abrupt shifts to unrelated topics, often to restricted interests, are a strong ASD marker.' },
  { name: 'avg_topic_duration_turns', label: 'How Long Topics Are Maintained', description: 'On average, how many turns a conversation topic is maintained before a topic shift occurs.', method: 'Average length of sequences of consecutive utterance pairs with cosine similarity above 0.30', asdRelevance: 'Short topic maintenance indicates difficulty sustaining mutual conversational focus.' },
  { name: 'topic_return_count', label: 'Returning to Previous Topics', description: 'How many times the conversation returned to a topic that had already been discussed earlier in the session.', method: 'Count of utterances with cosine similarity above 0.70 to an utterance from more than 5 turns ago', asdRelevance: 'Perseverative return to the same topic is a hallmark of restricted interests in ASD.' },
  { name: 'topic_diversity', label: 'Variety of Topics Discussed', description: 'How many distinct topics were covered throughout the conversation, based on statistical topic modelling.', method: 'Number of LDA topics (out of 5) with at least 10% probability in any utterance', asdRelevance: 'Low diversity suggests restricted topic range — a key ASD characteristic.' },
  { name: 'dominant_topic_ratio', label: 'How Much One Topic Dominates', description: "What fraction of the conversation is dominated by a single topic — a high value means most of the conversation revolved around one main theme.", method: 'Proportion of utterances where a single LDA topic has probability above 0.5', asdRelevance: 'A high dominant topic ratio may reflect restricted and repetitive interests.' },
  { name: 'topic_entropy', label: 'Evenness of Topic Distribution', description: 'A measure of how evenly distributed the conversation is across different topics — a high value means topics were varied and balanced.', method: 'Shannon entropy of the LDA topic probability distribution across all utterances', asdRelevance: 'Low entropy confirms a narrow topic range; higher entropy suggests more flexible conversation.' },
  { name: 'child_topic_consistency', label: "Consistency of Child's Topics (Statistical Model)", description: "How consistently the child talks about the same set of topics throughout the session, measured by a statistical topic model.", method: "Mean probability of the child's dominant LDA topic across the child's utterances", asdRelevance: "High consistency may indicate restricted topic focus; a strong ASD predictor." },
  { name: 'lexical_overlap_mean', label: 'Word Repetition Across Turns', description: 'On average, how many words from one turn also appear in the next turn — a measure of shared vocabulary between consecutive speakers.', method: 'Mean Jaccard coefficient of vocabulary between consecutive turns', asdRelevance: 'Low overlap may indicate topic fragmentation; very high overlap may indicate echolalia.' },
  { name: 'lexical_overlap_child', label: "Child's Word Repetition", description: "How much the child repeats words from previous turns in their own subsequent turns.", method: "Mean Jaccard coefficient of child utterances with their preceding utterances", asdRelevance: 'High child lexical overlap may indicate echolalic or perseverative speech patterns.' },
  { name: 'content_word_overlap', label: 'Meaningful Word Repetition', description: 'How many meaningful words (nouns, verbs, adjectives) are shared between consecutive turns, ignoring filler words like "the" or "and".', method: 'Jaccard similarity restricted to content words identified by spaCy POS tagging', asdRelevance: 'More sensitive than raw overlap; reflects genuine thematic continuity vs. grammatical filler.' },
  { name: 'novel_word_ratio', label: 'Introduction of New Words', description: 'What fraction of words used in each turn are completely new — not appearing in recent turns.', method: 'Proportion of words in each utterance not appearing in the previous 3 utterances', asdRelevance: 'Low novel word introduction may indicate repetitive language; very high may indicate topic jumps.' },
  { name: 'on_topic_response_ratio', label: 'On-Topic Response Rate', description: "What fraction of the child's responses are clearly related to what the adult just said.", method: 'Proportion of child responses with cosine similarity to prior adult utterance above 0.30', asdRelevance: "Directly quantifies the child's ability to produce contextually relevant responses." },
  { name: 'off_topic_response_count', label: 'Off-Topic Responses', description: "How many times the child responded with something clearly unrelated to what the adult just said.", method: 'Count of child responses with cosine similarity to prior adult utterance below 0.15', asdRelevance: 'Off-topic responses are a direct indicator of pragmatic language difficulty in ASD.' },
  { name: 'tangential_response_ratio', label: 'Tangential Responses', description: "What fraction of the child's responses are only loosely related to the adult's turn — not completely off-topic, but not directly relevant either.", method: 'Proportion of child responses with cosine similarity between 0.15 and 0.30', asdRelevance: 'Tangential responses reflect partial engagement — related in theme but missing the conversational point.' },
];

// ── PAUSE & LATENCY (47 features) ───────────────────────────────────────────
const PAUSE_LATENCY_FEATURES: Feature[] = [
  { name: 'response_latency_mean', label: 'Average Response Delay (Overall)', description: 'The average time (in seconds) between any speaker finishing their turn and the next speaker starting to speak.', method: 'Mean of all inter-turn gap durations across the full transcript', asdRelevance: 'Elevated average latency is a consistent ASD marker across multiple studies.' },
  { name: 'response_latency_median', label: 'Typical Response Delay (Overall)', description: 'The middle value of all response delays — a more stable measure than the average when some gaps are unusually long.', method: 'Median of all inter-turn gap durations', asdRelevance: 'The median is robust to outlier pauses; persistently elevated median indicates chronic delay.' },
  { name: 'response_latency_std', label: 'Variability in Response Delays', description: 'How much response times vary — is timing predictable or erratic?', method: 'Standard deviation of all inter-turn gap durations', asdRelevance: 'High variability suggests inconsistent processing or attention.' },
  { name: 'response_latency_max', label: 'Longest Response Delay', description: 'The single longest time anyone waited before responding during the entire session.', method: 'Maximum value across all inter-turn gaps', asdRelevance: 'Extreme outlier pauses may indicate complete communication breakdown or disengagement.' },
  { name: 'response_latency_min', label: 'Shortest Response Delay', description: 'The shortest time between turns — the fastest anyone responded.', method: 'Minimum value across all inter-turn gaps', asdRelevance: 'Very short minimums combined with high variance further evidence timing irregularity.' },
  { name: 'child_response_latency_mean', label: "Child's Average Response Delay", description: "On average, how many seconds the child takes to respond after the adult finishes speaking.", method: 'Mean of gaps from adult turn end to subsequent child turn start', asdRelevance: 'One of the strongest single ASD predictors — GMM clustering on ASDBank data identified 3 latency clusters.' },
  { name: 'child_response_latency_median', label: "Child's Typical Response Delay", description: "The middle value of all the child's response times — a stable measure of typical response speed.", method: 'Median of child response latencies', asdRelevance: 'Less affected by occasional very long pauses; reflects habitual response pattern.' },
  { name: 'child_response_latency_std', label: "Variability in Child's Response Timing", description: "How much the child's response times vary — are they consistently fast, consistently slow, or unpredictable?", method: 'Standard deviation of child response latencies', asdRelevance: 'Inconsistent latency may indicate variable attention or context-dependent processing speed.' },
  { name: 'adult_response_latency_mean', label: "Adult's Average Response Delay", description: "On average, how quickly the adult responds to the child.", method: 'Mean of gaps from child turn end to subsequent adult turn start', asdRelevance: 'Baseline measure; adults typically respond faster. Used to calculate asymmetry.' },
  { name: 'delayed_response_count', label: 'Number of Slow Responses', description: 'How many times the child took more than 0.45 seconds to respond — the threshold for a "normal" conversational response.', method: 'Count of child response latencies exceeding 0.45 seconds (GMM rapid-cluster boundary)', asdRelevance: 'Threshold derived from Gaussian Mixture Model fit on ASDBank corpus. High count is ASD-associated.' },
  { name: 'delayed_response_ratio', label: 'Proportion of Slow Responses', description: 'What fraction of the child\'s responses were "delayed" — taking more than the typical response threshold.', method: 'Delayed response count divided by total child responses', asdRelevance: 'Normalised measure; even brief sessions with a high ratio are diagnostically relevant.' },
  { name: 'very_delayed_response_count', label: 'Number of Very Long Delays', description: 'How many times the child took more than 2 seconds to respond — indicating a substantial processing delay.', method: 'Count of child response latencies exceeding 2.00 seconds (GMM long-pause boundary)', asdRelevance: 'Very long delays correspond to the "processing" and "disengaged" latency clusters from GMM analysis.' },
  { name: 'immediate_response_ratio', label: 'Proportion of Immediate Responses', description: 'What fraction of responses happened almost instantly — within 0.45 seconds of the prior turn ending.', method: 'Proportion of all response latencies below 0.45 seconds', asdRelevance: 'Low immediate response ratio is a consistent ASD signal; TD children respond quickly more often.' },
  { name: 'filled_pause_count', label: 'Total Filler Words Used', description: 'How many times any speaker used a filler word or hesitation sound — like "um", "uh", "er", or "ah" — during the conversation.', method: 'Regex pattern matching on transcript text: \\bum\\b, \\buh\\b, \\ber\\b, \\bah\\b, \\behm\\b, and CHAT disfluency markers &-um, &-uh', asdRelevance: 'Excessive filler use indicates word-finding difficulty or uncertainty.' },
  { name: 'filled_pause_ratio', label: 'Rate of Filler Words', description: 'How many filler words appear per total words spoken — a rate that normalises for how much was said overall.', method: 'Filled pause count divided by total word count', asdRelevance: 'A high ratio suggests significant speech disfluency even when accounting for total output.' },
  { name: 'filled_pause_per_utterance', label: 'Filler Words per Turn', description: 'On average, how many filler sounds appear in each utterance.', method: 'Total filled pauses divided by total utterance count', asdRelevance: 'Per-utterance normalisation reveals whether fillers are clustered in specific turns or pervasive.' },
  { name: 'child_filled_pause_count', label: "Child's Filler Words", description: "How many filler sounds (um, uh, er) the child specifically used.", method: 'Filled pause count restricted to child utterances only', asdRelevance: "Isolates the child's hesitation patterns from the adult's." },
  { name: 'child_filled_pause_ratio', label: "Rate of Child's Filler Words", description: "What fraction of the child's words are fillers.", method: "Child filled pause count divided by child total word count", asdRelevance: 'Higher ratio in children with ASD reflects greater word-finding difficulty.' },
  { name: 'um_count', label: 'Count of "um"', description: 'How many times the word "um" was used — the most common English filled pause.', method: 'Exact token match for "um" and CHAT marker &-um in transcript text', asdRelevance: 'Separately tracked as different fillers may have different prosodic and communicative functions.' },
  { name: 'uh_count', label: 'Count of "uh"', description: 'How many times "uh" was used — another very common filled pause.', method: 'Exact token match for "uh" and CHAT marker &-uh in transcript text', asdRelevance: '"uh" and "um" have been shown to signal different types of planning difficulty in speech production research.' },
  { name: 'other_filler_count', label: 'Count of Other Filler Sounds', description: 'How many times other hesitation markers were used — including "er", "ah", "ehm", and similar sounds.', method: 'Count of matches for \\ber\\b, \\bah\\b, \\behm\\b patterns in transcript text', asdRelevance: 'Captures the full range of disfluency markers beyond the most common um/uh.' },
  { name: 'unfilled_pause_count', label: 'Number of Silent Pauses', description: 'How many completely silent pauses were marked in the transcript — moments of silence explicitly coded by the transcriber.', method: 'Count of CHAT pause markers: (.) = brief, (..) = medium, (...) = long, (pause) = extended', asdRelevance: 'Unfilled pauses within utterances indicate mid-speech processing difficulty.' },
  { name: 'unfilled_pause_total_duration', label: 'Total Silent Pause Time', description: 'The total amount of time spent in silent pauses throughout the conversation.', method: 'Sum of pause durations: (.)=0.5s, (..)=1.0s, (...)=1.5s, (pause)=2.0s from CHAT conventions', asdRelevance: 'Total silent pause time reflects overall disruption to speech fluency.' },
  { name: 'unfilled_pause_mean_duration', label: 'Average Silent Pause Duration', description: 'On average, how long the silent pauses within utterances are.', method: 'Mean duration of all unfilled pauses weighted by their CHAT duration codes', asdRelevance: 'Longer average pauses indicate greater mid-speech processing delays.' },
  { name: 'long_pause_count', label: 'Number of Long Silent Pauses', description: 'How many pauses lasted more than 1 second (categorised as long pauses).', method: 'Count of CHAT pause markers with encoded duration above 1.0 second', asdRelevance: 'Long pauses within utterances disrupt communicative flow and are ASD-associated.' },
  { name: 'very_long_pause_count', label: 'Number of Very Long Pauses', description: 'How many pauses lasted more than 2 seconds — indicating substantial speech disruption.', method: 'Count of (pause) markers and (...) markers in CHAT transcript', asdRelevance: 'Very long pauses may represent complete mid-utterance abandonment, common in ASD.' },
  { name: 'pause_per_utterance', label: 'Pauses per Utterance', description: 'On average, how many pause markers appear inside each utterance.', method: 'Total unfilled pause markers divided by total utterance count', asdRelevance: 'High within-utterance pause density indicates fragmented speech production.' },
  { name: 'child_pause_count', label: "Child's Pause Count", description: "How many pause markers appear within the child's utterances specifically.", method: "Count of CHAT pause markers within child speaker tier utterances", asdRelevance: "Isolates the child's speech fluency from the adult's baseline." },
  { name: 'child_pause_ratio', label: "Child's Pause Rate", description: "How many pauses per word the child uses — normalised for total words spoken.", method: "Child pause count divided by child total word count", asdRelevance: 'Higher rates indicate speech planning difficulties in the child.' },
  { name: 'child_long_pause_ratio', label: "Proportion of Child's Long Pauses", description: "What fraction of the child's pauses are long ones (over 1 second).", method: "Child long pause count divided by child total pause count", asdRelevance: 'A high proportion of long pauses indicates severe disfluency rather than minor hesitations.' },
  { name: 'estimated_speaking_time', label: 'Total Speaking Time', description: 'An estimate of the total amount of time spent actually speaking (as opposed to being silent) during the session.', method: 'Sum of utterance durations from CHAT timing tiers or Whisper segment timestamps', asdRelevance: 'Low speaking time relative to session length indicates reduced verbal participation.' },
  { name: 'estimated_silence_time', label: 'Total Silence Time', description: 'An estimate of the total amount of time no one was speaking during the session.', method: 'Total session duration minus estimated speaking time', asdRelevance: 'High silence time reflects reduced engagement or communication difficulty.' },
  { name: 'speaking_silence_ratio', label: 'Speaking vs Silence Ratio', description: 'How much of the session was spent speaking compared to silence.', method: 'Estimated speaking time divided by estimated silence time', asdRelevance: 'A low ratio indicates a session dominated by silence — a strong ASD signal.' },
  { name: 'fluency_score', label: 'Overall Fluency Score', description: 'A combined measure of how fluent the conversation was, accounting for pauses, fillers, and gaps.', method: 'Composite score: 1 minus weighted sum of (pause_rate + filled_pause_ratio + delayed_response_ratio)', asdRelevance: 'Single summary fluency metric; lower scores are associated with ASD in the training data.' },
  { name: 'pause_distribution_skewness', label: 'Shape of Pause Distribution (Skew)', description: 'Whether pause lengths are skewed — for example, if most pauses are short but there are a few very long ones that pull the average up.', method: 'Statistical skewness of the distribution of all pause durations', asdRelevance: 'Positive skew (many short pauses, rare very long ones) vs. a flatter distribution separates TD from ASD groups.' },
  { name: 'pause_distribution_kurtosis', label: 'Shape of Pause Distribution (Peakedness)', description: 'How "peaked" or "spread out" the pause duration distribution is — are most pauses similar in length, or very varied?', method: 'Statistical kurtosis of the distribution of all pause durations', asdRelevance: 'High kurtosis (many similar-length pauses) vs. heavy tails (extreme variability) differs between groups.' },
  { name: 'pause_cv', label: 'Consistency of Pause Lengths', description: 'A normalised measure of how varied the pause lengths are relative to their average length.', method: 'Standard deviation of pause durations divided by mean pause duration (coefficient of variation)', asdRelevance: 'High CV indicates highly irregular pause timing — a consistent ASD feature.' },
  { name: 'latency_exponential_lambda', label: 'Response Timing Pattern (Lambda)', description: 'A mathematical parameter describing the rate at which response delays occur — essentially fitting an exponential model to response timing.', method: 'Maximum likelihood estimate of the exponential distribution rate parameter fit to response latency data', asdRelevance: 'Lower lambda values (slower rate of "normal" responses) indicate systematically delayed responding.' },
  { name: 'latency_percentile_75', label: '75th Percentile Response Latency', description: 'The response time below which 75% of all responses fall — a measure of the upper typical range.', method: '75th percentile of the response latency distribution', asdRelevance: 'A high 75th percentile means even typical responses are slow; complements the mean.' },
  { name: 'latency_percentile_90', label: '90th Percentile Response Latency', description: 'The response time below which 90% of responses fall — captures the boundary of the slowest responses.', method: '90th percentile of the response latency distribution', asdRelevance: 'The top 10% of latencies captures the most extreme delays.' },
  { name: 'latency_iqr', label: 'Spread of Response Latencies', description: 'The range of the middle 50% of response times — how wide the typical range is.', method: 'Interquartile range (Q75 minus Q25) of response latency distribution', asdRelevance: 'Wide IQR indicates highly variable response timing; narrow IQR shows consistent (slow or fast) responding.' },
  { name: 'hesitation_density', label: 'Hesitation Density', description: 'How many hesitation markers (fillers plus pauses) appear per unit of conversation time.', method: 'Sum of filled and unfilled pause counts divided by total recording duration in minutes', asdRelevance: 'High density reflects pervasive speech planning difficulty throughout the session.' },
  { name: 'false_start_count', label: 'False Starts (Restarts)', description: 'How many times the speaker started saying something and then restarted from the beginning.', method: 'Count of CHAT retrace marker [/] (repetition/false start) in transcript', asdRelevance: 'False starts indicate incomplete sentence planning — common in ASD language production.' },
  { name: 'word_repetition_count', label: 'Word Repetitions', description: 'How many times a word or phrase was immediately repeated — like saying "I I want" instead of "I want".', method: 'Count of CHAT revision marker [//] combined with text-based detection of consecutive identical tokens', asdRelevance: 'Word-level repetitions are a surface marker of disfluency and speech planning difficulty.' },
];

// ── REPAIR DETECTION (38 features) ──────────────────────────────────────────
const REPAIR_DETECTION_FEATURES: Feature[] = [
  { name: 'self_repair_count', label: 'Total Self-Corrections', description: 'How many times any speaker corrected or fixed something they had just said — catching their own mistake and trying again.', method: 'Count of CHAT retrace markers [/], [//], [///], [?] plus linguistic patterns: "I mean", "no wait", "actually", "or rather"', asdRelevance: 'Frequent self-repairs indicate difficulties with language planning and self-monitoring.' },
  { name: 'self_repair_ratio', label: 'Rate of Self-Corrections', description: 'What fraction of all utterances involved a self-correction.', method: 'Self-repair count divided by total utterance count', asdRelevance: 'Normalises for session length; a high ratio indicates pervasive production difficulties.' },
  { name: 'child_self_repair_count', label: "Child's Self-Corrections", description: "How many times the child corrected something they had just said.", method: 'CHAT retrace markers and linguistic repair phrases in child utterances only', asdRelevance: "Isolates the child's repair behaviour from the adult baseline." },
  { name: 'child_self_repair_ratio', label: "Rate of Child's Self-Corrections", description: "What fraction of the child's utterances involved a self-correction.", method: "Child self-repair count divided by child utterance count", asdRelevance: 'High rates indicate the child frequently misstarts or revises their own speech.' },
  { name: 'adult_self_repair_count', label: "Adult's Self-Corrections", description: "How many times the adult corrected themselves.", method: 'CHAT retrace markers in adult utterances', asdRelevance: 'Baseline measure; adults typically self-repair less than children with language difficulties.' },
  { name: 'retrace_count', label: 'Retraces (Going Back to Redo)', description: 'How many times a speaker went back to an earlier part of their utterance and repeated it from there.', method: 'Count of CHAT [/] (retrace without correction) markers in transcript', asdRelevance: 'Retracing indicates mid-utterance planning difficulty and disrupted fluency.' },
  { name: 'reformulation_count', label: 'Reformulations (Rephrasing)', description: 'How many times a speaker started saying something one way, stopped, and then said it a different way.', method: 'Count of CHAT [//] (retrace with correction/reformulation) markers', asdRelevance: 'Reformulations indicate the speaker found their original formulation inadequate and rephrased.' },
  { name: 'other_initiated_repair_count', label: 'Repairs Prompted by the Other Speaker', description: 'How many times a speaker made a correction because the other speaker indicated they had not understood.', method: 'Count of utterances following a clarification request that contain correction language or CHAT markers', asdRelevance: 'High counts suggest the child frequently produces unclear or unintelligible utterances requiring repair.' },
  { name: 'child_repair_after_clarification', label: 'Child Repairing After Being Asked', description: 'How many times the child successfully repaired their utterance after the adult asked for clarification.', method: 'Count of child repair turns that immediately follow an adult clarification request', asdRelevance: 'Low count suggests the child struggles to repair when explicitly prompted — a pragmatic skill deficit.' },
  { name: 'adult_repair_after_clarification', label: 'Adult Repairing After Being Asked', description: 'How many times the adult repaired their utterance after the child indicated they did not understand.', method: 'Count of adult repair turns following a child clarification request', asdRelevance: 'Baseline measure of repair responsiveness.' },
  { name: 'clarification_request_count', label: 'Total Clarification Requests', description: 'How many times any speaker asked for something to be repeated, explained, or clarified.', method: 'Pattern matching: "what?", "huh?", "pardon?", "say again", "I don\'t understand", "what do you mean", "come again"', asdRelevance: 'Frequent clarification requests from the adult indicate the child\'s speech is often unclear or off-topic.' },
  { name: 'clarification_request_ratio', label: 'Rate of Clarification Requests', description: 'What fraction of utterances are clarification requests.', method: 'Clarification request count divided by total utterance count', asdRelevance: 'High rates indicate persistent communication breakdown throughout the session.' },
  { name: 'child_clarification_count', label: "Child's Clarification Requests", description: "How many times the child asked for something to be clarified or repeated.", method: 'Clarification pattern matches in child utterances only', asdRelevance: "Child clarification requests may indicate comprehension difficulty." },
  { name: 'adult_clarification_count', label: "Adult's Clarification Requests", description: 'How many times the adult had to ask for clarification.', method: 'Clarification pattern matches in adult utterances only', asdRelevance: "High adult clarification requests indicate the child's speech is frequently unclear or tangential." },
  { name: 'clarification_to_child_count', label: 'Clarifications Directed at Child', description: 'How many times the adult specifically asked the child for clarification.', method: 'Adult clarification patterns followed by child response turns', asdRelevance: "Directly measures how often the child's speech needs clarification." },
  { name: 'clarification_to_adult_count', label: 'Clarifications Directed at Adult', description: 'How many times the child asked the adult for clarification.', method: 'Child clarification patterns followed by adult response turns', asdRelevance: 'Measures bidirectional nature of communication breakdowns.' },
  { name: 'confirmation_check_count', label: 'Confirmation Checks', description: 'How many times any speaker checked whether they had been understood or asked for agreement — like "okay?" or "right?"', method: 'Pattern matching for confirmation-seeking phrases at turn ends: "okay?", "right?", "yeah?", "you know?"', asdRelevance: 'Frequent confirmation-seeking may indicate uncertainty about communicative success.' },
  { name: 'child_confirmation_check_count', label: "Child's Confirmation Checks", description: "How many times the child checked whether the adult understood them.", method: 'Confirmation patterns in child utterances', asdRelevance: 'Shows the child\'s meta-awareness of communication — may be absent in ASD ("theory of mind" link).' },
  { name: 'repetition_repair_count', label: 'Repairs by Repetition', description: 'How many times a speaker repaired a miscommunication by simply repeating what they had just said.', method: 'Count of utterances with high lexical overlap (>80%) with the previous utterance and following a clarification request', asdRelevance: 'Repetition-based repair is a simpler strategy than reformulation; prevalent in ASD.' },
  { name: 'partial_repetition_count', label: 'Partial Repetition Repairs', description: 'How many times a speaker repaired by repeating only part of what they said, keeping the same words but adding or changing something.', method: 'Count of utterances with 40–80% lexical overlap with preceding utterance following a clarification request', asdRelevance: 'Indicates moderate repair capability — the speaker can expand but not fully rephrase.' },
  { name: 'exact_repetition_count', label: 'Exact Repetition Repairs', description: 'How many times the speaker said exactly the same thing again without any change after a clarification request.', method: 'Count of utterances with >80% lexical overlap with preceding utterance following a clarification', asdRelevance: 'Exact repetition without modification is a limited repair strategy common in ASD.' },
  { name: 'expansion_repair_count', label: 'Repairs by Expanding', description: 'How many times a speaker repaired by adding more information or elaborating, rather than just repeating.', method: 'Count of repair turns that are longer and contain new content words compared to the preceding utterance', asdRelevance: 'Expansion repair shows more sophisticated communicative ability — less common in ASD.' },
  { name: 'repair_success_count', label: 'Successful Repairs', description: 'How many times a repair attempt led to the conversation continuing normally — meaning the other speaker understood and moved on.', method: 'Count of repair sequences followed by an on-topic response rather than another clarification request', asdRelevance: 'High repair success indicates the child can resolve misunderstandings when they occur.' },
  { name: 'repair_success_rate', label: 'Repair Success Rate', description: 'What fraction of repair attempts were successful.', method: 'Repair success count divided by total repair attempt count', asdRelevance: 'Low success rate indicates the child repeatedly fails to achieve mutual understanding.' },
  { name: 'repair_failure_count', label: 'Failed Repairs', description: 'How many times a repair attempt did not resolve the communication breakdown.', method: 'Count of repair sequences followed by a repeated clarification request from the other speaker', asdRelevance: 'Frequent repair failures indicate persistent communication difficulty that cannot be easily resolved.' },
  { name: 'repair_attempt_rate', label: 'Rate of Repair Attempts', description: 'How many repair attempts happen per minute of conversation.', method: 'Total repair attempts divided by total recording duration in minutes', asdRelevance: 'Normalised measure of repair frequency; high rates indicate pervasive communication difficulty.' },
  { name: 'avg_repair_sequence_length', label: 'Average Repair Sequence Length', description: 'On average, how many back-and-forth turns it takes to resolve a communication breakdown.', method: 'Mean number of turns from initial repair initiation to resolution or abandonment', asdRelevance: 'Longer sequences indicate more difficulty achieving mutual understanding.' },
  { name: 'max_repair_sequence_length', label: 'Longest Repair Sequence', description: 'The most turns it ever took to resolve a single communication breakdown in the session.', method: 'Maximum number of turns in any single repair sequence', asdRelevance: 'Captures worst-case communication breakdowns that dominate the session.' },
  { name: 'extended_repair_count', label: 'Extended Repair Sequences', description: 'How many repair sequences required more than 3 turns to resolve.', method: 'Count of repair sequences with length greater than 3 turns', asdRelevance: 'Extended repairs significantly disrupt conversational flow and are ASD-associated.' },
  { name: 'repair_acknowledgment_count', label: 'Repair Acknowledgments', description: 'How many times after a repair the other speaker explicitly acknowledged the repair — saying things like "oh, I see" or "ah okay".', method: 'Pattern matching for acknowledgment tokens (oh, I see, ah, ok, right) after repair sequences', asdRelevance: 'Acknowledgments confirm repair success and indicate mutual understanding was achieved.' },
  { name: 'repair_uptake_ratio', label: 'Repair Uptake Rate', description: 'What fraction of repairs were explicitly acknowledged by the other speaker.', method: 'Repair acknowledgment count divided by repair success count', asdRelevance: 'Low uptake ratio may indicate repairs were only partially successful.' },
  { name: 'child_repair_effectiveness', label: "Child's Repair Effectiveness", description: "A combined measure of how effectively the child handles communication breakdowns — taking into account how often they try and how often they succeed.", method: 'Composite: child repair success rate weighted by child repair attempt rate', asdRelevance: "Key metric for assessing the child's pragmatic repair capability." },
  { name: 'child_needs_repair_ratio', label: 'How Often Child Needs Repair', description: "What fraction of the child's utterances are followed by a clarification request from the adult — indicating the adult didn't understand.", method: "Child utterances followed by adult clarification request, divided by total child utterances", asdRelevance: 'Measures communication clarity; high values indicate frequently unclear or confusing child speech.' },
  { name: 'child_provides_repair_ratio', label: 'How Often Child Successfully Repairs', description: "What fraction of the time the child successfully provides a repair when the adult asks for clarification.", method: 'Child successful repair count divided by total adult clarification requests directed at child', asdRelevance: 'Low values indicate the child cannot or does not repair when prompted — a clear ASD marker.' },
  { name: 'repair_strategy_diversity', label: 'Variety of Repair Strategies', description: 'How many different types of repair strategies (repetition, expansion, reformulation, etc.) the child uses.', method: 'Count of distinct repair strategy types observed across all child repair sequences', asdRelevance: 'Children with ASD typically use fewer and simpler repair strategies.' },
  { name: 'dominant_repair_strategy', label: 'Most-Used Repair Strategy', description: 'The type of repair the child relies on most — e.g. exact repetition, partial repetition, or reformulation.', method: 'Mode of repair strategy type labels across all child repair sequences', asdRelevance: 'Reliance on exact repetition (the simplest strategy) is strongly associated with ASD.' },
  { name: 'breakdown_count', label: 'Communication Breakdowns', description: 'How many times a complete communication breakdown occurred — where the conversation could not continue without explicit repair work.', method: 'Count of sequences with 2+ consecutive clarification requests or repair failures', asdRelevance: 'Communication breakdowns are clinically significant and strongly ASD-associated.' },
  { name: 'breakdown_resolution_rate', label: 'Breakdown Resolution Rate', description: 'What fraction of communication breakdowns were ultimately resolved and the conversation was able to continue.', method: 'Resolved breakdown count divided by total breakdown count', asdRelevance: 'Low resolution rates indicate the child cannot recover from communication difficulties.' },
  { name: 'unresolved_breakdown_count', label: 'Unresolved Breakdowns', description: 'How many times a communication breakdown occurred and was never resolved — the conversation either stopped or moved on without resolution.', method: 'Count of breakdown sequences with no subsequent successful repair or topic continuation', asdRelevance: 'Unresolved breakdowns represent the most severe communication failures in the session.' },
];

// ── PRAGMATIC LINGUISTIC (35 features) ──────────────────────────────────────
const PRAGMATIC_LINGUISTIC_FEATURES: Feature[] = [
  { name: 'mlu_words', label: 'Mean Length of Utterance (Words)', description: "On average, how many words the child uses in each utterance. This is one of the most established measures of a child's language development.", method: 'Mean word count per child utterance after removing CHAT markup, annotations, and non-speech tokens', asdRelevance: 'MLU is one of the strongest indicators of language development stage and is often reduced in ASD.' },
  { name: 'mlu_morphemes', label: 'Mean Length of Utterance (Morphemes)', description: "A more precise version of MLU that counts individual meaning units (morphemes) rather than whole words — so 'walked' counts as 2 (walk + -ed), and 'cats' counts as 2 (cat + -s).", method: 'Parsed from CHAT %mor (morphology) tier annotations; each morpheme token counted separately', asdRelevance: 'Morpheme-based MLU is more sensitive to grammatical complexity than word-based MLU.' },
  { name: 'avg_word_length_chars', label: 'Average Word Length', description: "On average, how many letters long the child's words are. Longer words generally reflect a more advanced vocabulary.", method: 'Mean character count per word token in child utterances, excluding punctuation', asdRelevance: 'Very short average word length may indicate reliance on simple, high-frequency vocabulary.' },
  { name: 'max_utterance_length', label: "Child's Longest Utterance", description: "The most words the child used in any single utterance — their peak linguistic output in the session.", method: 'Maximum word count across all child utterances', asdRelevance: 'Reflects ceiling linguistic capacity; contrasted with mean MLU to reveal consistency.' },
  { name: 'total_words', label: 'Total Words Spoken', description: "The total number of words the child produced in the entire session.", method: 'Sum of word counts across all child utterances', asdRelevance: 'Low total word count indicates reduced verbal output, common in minimally verbal ASD.' },
  { name: 'unique_words', label: 'Vocabulary Size (Unique Words)', description: "How many different words the child used — the size of their active vocabulary in this session.", method: 'Count of distinct word types (case-insensitive) across all child utterances', asdRelevance: 'Limited vocabulary diversity is associated with ASD, especially at younger ages.' },
  { name: 'type_token_ratio', label: 'Vocabulary Diversity (TTR)', description: "The ratio of unique words to total words — a classic measure of how varied someone's vocabulary is. A ratio of 1.0 means every word was different; 0.0 means the same word was said over and over.", method: 'Unique words divided by total words (type-token ratio)', asdRelevance: 'Low TTR indicates repetitive, stereotyped language; a consistent ASD marker.' },
  { name: 'corrected_ttr', label: 'Corrected Vocabulary Diversity', description: "A mathematically corrected version of vocabulary diversity that accounts for the fact that longer texts naturally have lower TTR just because there are more words. This makes it fair to compare across children who spoke different amounts.", method: 'Unique words divided by the square root of (2 times total words) — corrected type-token ratio formula', asdRelevance: 'More reliable than raw TTR for comparing children with very different total word counts.' },
  { name: 'lexical_density', label: 'Meaningful Word Ratio', description: "What fraction of the child's words are meaningful content words (nouns, verbs, adjectives, adverbs) rather than grammatical filler words (like 'the', 'and', 'is').", method: 'Count of NOUN, VERB, ADJ, ADV POS-tagged tokens divided by total token count using spaCy', asdRelevance: 'Low lexical density may indicate formulaic or scripted language patterns.' },
  { name: 'utterance_complexity_score', label: 'Utterance Complexity Score', description: "A combined score of how complex the child's utterances are, taking into account length, vocabulary diversity, and grammatical variety.", method: 'Composite of MLU, TTR, and average dependency tree depth, normalised to 0–1 scale', asdRelevance: 'Lower complexity scores are associated with less developed pragmatic language in ASD.' },
  { name: 'echolalia_ratio', label: 'Echolalia Ratio', description: "How often the child repeats back what the adult just said, either word-for-word or nearly so. Echolalia is when someone echoes speech they've heard rather than producing original language.", method: 'Proportion of child utterances with >60% word overlap with the immediately preceding adult utterance', asdRelevance: 'Echolalia is a hallmark feature of ASD speech — children echo rather than produce novel language.' },
  { name: 'immediate_echolalia_count', label: 'Immediate Echolalia', description: "How many times the child repeated back the adult's most recent utterance almost immediately after hearing it.", method: 'Count of child utterances with exact or near-exact match to the preceding adult utterance (>80% overlap)', asdRelevance: 'Immediate echolalia is strongly diagnostic — it replaces functional communicative response.' },
  { name: 'delayed_echolalia_count', label: 'Delayed Echolalia', description: "How many times the child appeared to echo something that was said earlier in the conversation — not right away, but after several turns had passed.", method: 'Count of child utterances with >60% overlap with any adult utterance in the previous 10 turns', asdRelevance: 'Delayed echolalia reflects scripted, memorised language rather than contextual production.' },
  { name: 'partial_repetition_ratio', label: 'Partial Repetition Rate', description: "How often the child repeats part of what was recently said — not a complete echo, but borrowing key words or phrases from recent turns.", method: 'Proportion of child utterances with 40–60% word overlap with the 3 most recent utterances', asdRelevance: 'Partial repetition is an intermediate form of echolalia and indicates scripted language patterns.' },
  { name: 'question_ratio', label: 'Rate of Questions', description: "What fraction of the child's utterances are questions.", method: "Proportion of child utterances ending with '?' or containing question word patterns (who, what, where, when, why, how)", asdRelevance: 'Children with ASD ask fewer spontaneous questions, reflecting reduced social curiosity.' },
  { name: 'question_diversity', label: 'Variety of Questions', description: "How many different types of questions the child asks — do they only ask yes/no questions, or do they use a range of question forms?", method: 'Count of distinct question types (yes/no, wh-, tag, alternative) in child utterances', asdRelevance: 'Low question diversity indicates a limited communicative repertoire.' },
  { name: 'yes_no_question_ratio', label: 'Yes/No Questions', description: "What fraction of the child's questions are simple yes/no questions.", method: 'Proportion of question utterances not beginning with wh- words', asdRelevance: 'Over-reliance on yes/no questions may indicate limited conversational range.' },
  { name: 'wh_question_ratio', label: 'Wh-Questions', description: "What fraction of the child's questions use who, what, where, when, why, or how.", method: "Proportion of question utterances beginning with wh- question words", asdRelevance: 'Wh-questions require theory of mind and social curiosity — often reduced in ASD.' },
  { name: 'pronoun_usage_ratio', label: 'Pronoun Usage Rate', description: "How often the child uses pronouns (I, you, he, she, they, etc.) relative to total words — pronouns are a fundamental tool for efficient communication.", method: 'Count of PRP and PRP$ POS-tagged tokens divided by total token count', asdRelevance: 'Pronoun difficulties are a classic ASD feature, particularly pronoun reversal and avoidance.' },
  { name: 'first_person_pronoun_ratio', label: 'First-Person Pronoun Rate', description: "How often the child refers to themselves using 'I', 'me', 'my', 'mine', 'myself'.", method: "Count of first-person pronouns (I, me, my, mine, myself) divided by total pronouns used", asdRelevance: 'Avoidance of self-referential language or use of name instead of "I" is an ASD feature.' },
  { name: 'pronoun_error_ratio', label: 'Pronoun Error Rate', description: "How often the child uses pronouns incorrectly — for example, using 'he' when they mean 'I', or mixing up 'you' and 'me'.", method: "Heuristic detection of pronoun usage in syntactically unexpected positions using spaCy dependency parsing", asdRelevance: 'Pronoun errors are strongly associated with ASD and reflect perspective-taking difficulties.' },
  { name: 'pronoun_reversal_count', label: 'Pronoun Reversals', description: "How many times the child reversed a pronoun — the most classic ASD pronoun error, where 'you' is used to mean 'I'.", method: "Pattern matching for 'you want/like/need/have' in child utterances in contexts where self-reference is expected", asdRelevance: 'Pronoun reversal is one of the most specific linguistic markers of ASD.' },
  { name: 'social_phrase_ratio', label: 'Social Language Rate', description: "How often the child uses socially oriented phrases — like greetings, thanks, or expressions of emotion.", method: "Proportion of utterances containing social phrase patterns: 'please', 'thank you', 'sorry', 'excuse me', emotional expressions", asdRelevance: 'Low social language use reflects reduced social motivation and pragmatic language knowledge.' },
  { name: 'greeting_count', label: 'Greetings Used', description: "How many times the child used a greeting (hi, hello, bye, goodbye, see you).", method: "Count of utterances matching greeting word patterns at turn boundaries", asdRelevance: 'Reduced greeting behaviour is a well-documented social communication deficit in ASD.' },
  { name: 'politeness_marker_count', label: 'Politeness Markers Used', description: "How many times the child used polite language — like 'please', 'thank you', or 'sorry'.", method: "Count of utterances containing politeness lexical items", asdRelevance: 'Difficulty with pragmatic politeness conventions is associated with ASD social language profiles.' },
  { name: 'appropriate_response_ratio', label: 'Appropriate Response Rate', description: "What fraction of the child's responses are judged to be appropriate given the preceding adult turn.", method: 'Proportion of child utterances that are semantically relevant and grammatically complete responses to adult prompts', asdRelevance: 'Overall measure of pragmatic competence; strongly differentiated between ASD and TD groups.' },
  { name: 'unintelligible_ratio', label: 'Unintelligible Speech Rate', description: "What fraction of the child's words or utterances were so unclear they couldn't be transcribed. In the CHAT format, these are marked as 'xxx'.", method: "Count of 'xxx' tokens in child utterances divided by total child tokens", asdRelevance: 'High unintelligibility is associated with ASD, particularly in younger children.' },
  { name: 'discourse_marker_ratio', label: 'Discourse Marker Rate', description: "How often the child uses words that connect ideas or signal their reasoning — like 'because', 'so', 'but', 'then', 'also'.", method: "Proportion of utterances containing discourse connectives and conjunctions from a predefined lexical list", asdRelevance: 'Low use of discourse markers indicates difficulty structuring logical, connected discourse.' },
  { name: 'continuation_marker_ratio', label: 'Continuation Marker Rate', description: "How often the child signals they want to keep talking or add more — using words like 'and', 'also', 'plus', 'and then'.", method: "Proportion of utterances beginning with continuation words", asdRelevance: 'Continuation markers reflect ability to build extended discourse — often reduced in ASD.' },
  { name: 'acknowledgment_ratio', label: 'Acknowledgment Rate', description: "How often the child uses backchannel words to show they are listening — like 'yes', 'okay', 'mm', 'right', 'uh-huh'.", method: "Proportion of child utterances consisting primarily of acknowledgment tokens", asdRelevance: 'Low acknowledgment rates indicate reduced active listening and social reciprocity.' },
  { name: 'nonverbal_behavior_ratio', label: 'Non-Verbal Behaviour Rate', description: "How often non-verbal behaviours are recorded in the transcript — things like laughing, pointing, clapping, or crying that are noted by the transcriber.", method: "Proportion of utterances containing CHAT action markers: &=laughs, &=cries, &=screams, &=claps, &=points, etc.", asdRelevance: 'Non-verbal communication is a key domain of ASD assessment alongside verbal language.' },
  { name: 'laughter_ratio', label: 'Laughter Rate', description: "How often laughter is recorded in the transcript relative to total utterances.", method: "Count of &=laughs CHAT markers divided by total utterance count", asdRelevance: 'Both unusually high and low laughter rates can be diagnostically relevant in ASD context.' },
  { name: 'vocal_behavior_diversity', label: 'Variety of Non-Verbal Sounds', description: "How many different types of non-verbal vocal or physical behaviours were recorded throughout the session.", method: "Count of distinct CHAT action marker types observed across the transcript", asdRelevance: 'Diverse non-verbal behaviour indicates richer communicative repertoire; limited variety is ASD-associated.' },
];

// ── AUDIO-DERIVED PRAGMATIC (30 features) ───────────────────────────────────
const AUDIO_PRAGMATIC_FEATURES: Feature[] = [
  { name: 'audio_pause_count', label: 'Number of Pauses (Audio)', description: 'How many pauses were detected directly from the audio signal using energy and silence detection.', method: 'Silence detection on audio energy envelope: frames with RMS energy below threshold for >0.15s', asdRelevance: 'Audio-detected pauses are independent of transcription — confirms transcript-based pause counts.' },
  { name: 'audio_pause_total_duration', label: 'Total Pause Duration (Audio)', description: 'The total seconds spent in silence across the entire recording.', method: 'Sum of all silence segment durations above 0.15s threshold from audio energy analysis', asdRelevance: 'High total silence duration confirms reduced verbal output.' },
  { name: 'audio_pause_mean_duration', label: 'Average Pause Length (Audio)', description: 'On average, how long each silence in the audio is.', method: 'Mean duration of all detected silence segments', asdRelevance: 'Longer average pauses indicate habitual speech planning difficulty.' },
  { name: 'audio_pause_median_duration', label: 'Typical Pause Length (Audio)', description: 'The middle value of all pause lengths — less affected by occasional very long silences.', method: 'Median of all silence segment durations', asdRelevance: 'More stable measure of typical pause behaviour.' },
  { name: 'audio_pause_std_duration', label: 'Variability in Pause Length (Audio)', description: 'How much pause lengths vary throughout the recording.', method: 'Standard deviation of silence segment durations', asdRelevance: 'High variability may indicate irregular processing or engagement.' },
  { name: 'audio_pause_max_duration', label: 'Longest Pause (Audio)', description: 'The single longest silence detected in the audio.', method: 'Maximum value across all silence segment durations', asdRelevance: 'Very long silences indicate severe disengagement or communication breakdown.' },
  { name: 'audio_pause_min_duration', label: 'Shortest Pause (Audio)', description: 'The shortest silence detected — the minimum gap between speech segments.', method: 'Minimum value above the silence threshold', asdRelevance: 'Contextualises the pause range.' },
  { name: 'audio_long_pause_count', label: 'Number of Long Pauses (Audio)', description: 'How many silences in the audio lasted more than 2 seconds.', method: 'Count of silence segments with duration above 2.0 seconds', asdRelevance: 'Confirms GMM-threshold long pause counts from transcript analysis.' },
  { name: 'audio_very_long_pause_count', label: 'Number of Very Long Pauses (Audio)', description: 'How many silences in the audio lasted more than 4 seconds.', method: 'Count of silence segments with duration above 4.0 seconds (disengaged cluster boundary)', asdRelevance: 'Very long pauses correspond to the "disengaged" cluster from GMM latency analysis.' },
  { name: 'audio_pause_ratio', label: 'Proportion of Audio That Is Silent', description: 'What fraction of the recording is silence rather than speech.', method: 'Total silence duration divided by total recording duration', asdRelevance: 'High ratio indicates the session is dominated by silence — a strong ASD signal.' },
  { name: 'audio_speaking_ratio', label: 'Proportion of Audio That Is Speech', description: 'What fraction of the recording contains active speech.', method: '1 minus audio pause ratio', asdRelevance: 'Complement of pause ratio; directly measures verbal engagement.' },
  { name: 'audio_pause_rate_per_minute', label: 'Pause Rate (per minute)', description: 'How many pauses happen per minute of recording.', method: 'Total pause count divided by recording duration in minutes', asdRelevance: 'Normalised measure of pause frequency independent of session length.' },
  { name: 'audio_filled_pause_count', label: 'Filled Pauses Detected from Audio', description: 'An estimate of how many filler sounds (um, uh) are present based on audio characteristics.', method: 'Count of short voiced segments with formant patterns consistent with schwa/nasal fillers', asdRelevance: 'Audio-based filler detection cross-validates transcript-based filler counts.' },
  { name: 'audio_unfilled_pause_count', label: 'Silent Pauses Detected from Audio', description: 'How many completely silent pauses were detected from the audio signal.', method: 'Count of silence segments between voiced speech frames', asdRelevance: 'Used to cross-validate CHAT pause marker counts.' },
  { name: 'audio_filled_pause_ratio', label: 'Rate of Filled Pauses (Audio)', description: 'What fraction of pause events are filled pauses (voiced) rather than silent.', method: 'Filled pause count divided by (filled + unfilled pause count)', asdRelevance: 'The mix of filled vs unfilled pauses reflects different aspects of speech planning difficulty.' },
  { name: 'audio_speaking_rate_wpm', label: 'Speaking Rate (Words Per Minute)', description: "How fast the child speaks in words per minute — counting only the time they are actively talking, not pauses.", method: 'Child word count divided by child active speaking time (silence segments excluded)', asdRelevance: 'Reduced speaking rate is associated with ASD; very high rates may indicate pressured speech.' },
  { name: 'audio_articulation_rate', label: 'Articulation Rate', description: "How fast the child's mouth is moving when they are speaking — the rate of syllable production.", method: 'Estimated syllable count divided by voiced segment duration (excluding pauses)', asdRelevance: 'Abnormal articulation rates (both high and low) are ASD-associated.' },
  { name: 'audio_speech_rate_variability', label: 'Speech Rate Variability', description: "How much the child's speaking speed varies throughout the session.", method: 'Standard deviation of per-utterance speaking rates in words per minute', asdRelevance: 'High variability may indicate uneven engagement; very flat rate may indicate scripted speech.' },
  { name: 'audio_segment_duration_mean', label: 'Average Speech Segment Length', description: "On average, how long each continuous stretch of speech is before a pause occurs.", method: 'Mean duration of voiced speech segments between silence intervals', asdRelevance: 'Short segments indicate fragmented, pause-interrupted speech production.' },
  { name: 'audio_segment_duration_std', label: 'Variability in Speech Segment Length', description: 'How much the length of continuous speech segments varies.', method: 'Standard deviation of voiced speech segment durations', asdRelevance: 'High variability may reflect intermittent engagement or topic-dependent fluency.' },
  { name: 'audio_segment_duration_max', label: 'Longest Continuous Speech Segment', description: "The longest stretch of uninterrupted speech in the recording.", method: 'Maximum voiced segment duration', asdRelevance: 'A long maximum segment may indicate monologuing on a restricted topic.' },
  { name: 'audio_segment_duration_min', label: 'Shortest Speech Segment', description: 'The shortest voiced speech segment detected.', method: 'Minimum voiced segment duration above noise floor', asdRelevance: 'Very short segments may indicate single-word responses or minimal engagement.' },
  { name: 'audio_response_latency_mean', label: 'Average Response Delay (Audio)', description: "On average, how long after someone stops speaking the next speaker begins — measured directly from the audio.", method: 'Mean of silence segments between alternating speaker segments identified by speaker diarisation', asdRelevance: 'Audio-derived latency is speaker-diarisation-based and independent of transcript timing.' },
  { name: 'audio_response_latency_std', label: 'Variability in Response Delays (Audio)', description: 'How much the response delays vary throughout the recording.', method: 'Standard deviation of audio-derived inter-speaker silence durations', asdRelevance: 'High variability indicates inconsistent response timing.' },
  { name: 'audio_response_latency_max', label: 'Longest Response Delay (Audio)', description: 'The longest silence between speakers detected in the audio.', method: 'Maximum inter-speaker silence duration from audio analysis', asdRelevance: 'Captures worst-case latency for cross-validation with transcript-based measures.' },
  { name: 'audio_total_duration', label: 'Total Recording Duration', description: 'The total length of the audio recording in seconds.', method: 'Audio file duration from librosa.get_duration()', asdRelevance: 'Used to normalise all time-based features for fair comparison across sessions.' },
  { name: 'audio_speech_duration', label: 'Total Speech Duration', description: 'The total amount of time spent in active speech throughout the recording.', method: 'Sum of all voiced segment durations identified by energy thresholding', asdRelevance: 'Core measure of verbal output; low values indicate reduced verbal participation.' },
  { name: 'audio_silence_duration', label: 'Total Silence Duration', description: 'The total amount of time spent in silence throughout the recording.', method: 'Total recording duration minus total speech duration', asdRelevance: 'High silence duration confirms reduced engagement.' },
  { name: 'audio_speech_to_silence_ratio', label: 'Speech-to-Silence Ratio (Audio)', description: 'How much speech there is compared to silence — a direct measure of verbal engagement in the session.', method: 'Total speech duration divided by total silence duration', asdRelevance: 'Low ratio is a strong ASD signal; high ratio indicates an active, verbally engaged child.' },
];
// ── ACOUSTIC FEATURE SETS ────────────────────────────────────────────────────

const PITCH_FEATURES: Feature[] = [
  { name: 'f0_mean', label: 'Average Pitch (F0)', description: 'The average fundamental frequency of the child\'s voice across the whole recording — essentially their average pitch in Hz.', method: 'pyin algorithm via librosa.pyin(); voiced frames extracted at 10ms hop length; mean computed over all voiced frames', asdRelevance: 'Atypical mean pitch (both higher and lower than typical) has been documented in ASD, potentially reflecting differences in laryngeal control or emotional prosody.' },
  { name: 'f0_std', label: 'Pitch Variability', description: 'How much the child\'s pitch varies over the course of the recording. High variability means the pitch goes up and down a lot; low variability means it stays at a similar level throughout.', method: 'Standard deviation of F0 values across all voiced frames', asdRelevance: 'Reduced pitch variability (monotone speech) is one of the most consistent prosodic ASD markers, reflecting limited use of intonation to convey meaning and emotion.' },
  { name: 'f0_min', label: 'Lowest Pitch Observed', description: 'The lowest pitch value recorded at any point during the child\'s speech.', method: 'Minimum F0 across all voiced frames with voicing confidence > 0.5', asdRelevance: 'Contrasted with maximum pitch to compute the pitch range; low minimum may indicate a compressed tonal range.' },
  { name: 'f0_max', label: 'Highest Pitch Observed', description: 'The highest pitch value at any point during the recording.', method: 'Maximum F0 across all voiced frames with voicing confidence > 0.5', asdRelevance: 'Children with ASD may show restricted high-pitch excursions, reflecting reduced use of question intonation and exclamatory contours.' },
  { name: 'f0_range', label: 'Pitch Range', description: 'The total span from lowest to highest pitch — a direct measure of how much the voice rises and falls overall.', method: 'f0_max minus f0_min across all voiced frames', asdRelevance: 'Narrowed pitch range is a hallmark prosodic feature of ASD; contrasts with typical children\'s broader intonational contours.' },
  { name: 'f0_median', label: 'Median Pitch', description: 'The middle value of all pitch measurements — more robust than the mean when extreme values are present.', method: 'Median F0 across all voiced frames', asdRelevance: 'Provides a stable estimate of habitual speaking pitch, less sensitive to isolated high or low excursions.' },
  { name: 'f0_iqr', label: 'Pitch Spread (IQR)', description: 'The range of the middle 50% of pitch values — capturing typical pitch movement without being distorted by extreme highs or lows.', method: 'Interquartile range (Q75 minus Q25) of F0 distribution', asdRelevance: 'A tighter IQR confirms monotone speech even when a few pitch outliers inflate the total range.' },
  { name: 'f0_skewness', label: 'Pitch Distribution Shape (Skew)', description: 'Whether pitch values cluster more towards high or low ends — whether the voice tends to stay flat and occasionally spike up, or stays high and occasionally dips.', method: 'Statistical skewness of the F0 distribution across voiced frames', asdRelevance: 'Asymmetric pitch distributions may reflect atypical intonation patterns, such as terminal rises on non-questions (common in ASD).' },
  { name: 'f0_kurtosis', label: 'Pitch Distribution Peakedness', description: 'Whether pitch values cluster tightly around a single level or spread broadly across a range.', method: 'Statistical kurtosis of the F0 distribution', asdRelevance: 'High kurtosis (peaked distribution) indicates flat, monotone prosody; low kurtosis indicates more varied pitch use.' },
  { name: 'f0_voiced_ratio', label: 'Proportion of Voiced Speech', description: 'What fraction of the speech signal has a detectable pitch — i.e., how much of the recording is voiced (vs. unvoiced consonants or silence).', method: 'Proportion of frames with pyin voicing confidence above 0.5 threshold', asdRelevance: 'Low voiced ratio may indicate atypical phonation, frequent whispering, or unusually high use of unvoiced speech sounds.' },
];

const MFCC_FEATURES: Feature[] = [
  { name: 'mfcc_1_mean', label: 'MFCC 1 — Average (Overall Energy)', description: 'The first MFCC coefficient, which broadly captures the overall energy or loudness of the speech spectrum. It is the most fundamental descriptor of the frequency envelope shape.', method: 'librosa.feature.mfcc() with n_mfcc=13, n_fft=2048, hop_length=512; mean of coefficient 1 across all frames', asdRelevance: 'MFCC 1 relates to overall vocal tract configuration; atypical values may reflect structural or habitual differences in phonation.' },
  { name: 'mfcc_2_mean', label: 'MFCC 2 — Average (Spectral Tilt)', description: 'The second MFCC, capturing the broad spectral tilt — whether the speech energy is concentrated in low or high frequencies.', method: 'Mean of MFCC coefficient 2 across all frames', asdRelevance: 'Spectral tilt differences have been documented in ASD, reflecting differences in resonance and articulation.' },
  { name: 'mfcc_3_mean', label: 'MFCC 3–13 — Spectral Shape Details', description: 'The higher-order MFCCs (3 through 13) capture progressively finer details of the spectral shape — the characteristic "texture" of the voice that distinguishes different vowels, consonants, and speaker identities.', method: 'Mean of each MFCC coefficient 3–13 across all frames', asdRelevance: 'The full MFCC vector encodes articulatory patterns; systematic differences between ASD and TD children have been found in several coefficients, particularly related to vowel formant patterns.' },
  { name: 'mfcc_1_std', label: 'MFCC 1 — Variability', description: 'How much the first MFCC coefficient varies over time — capturing temporal changes in overall spectral energy.', method: 'Standard deviation of MFCC coefficient 1 across all frames', asdRelevance: 'Low MFCC variability across all coefficients indicates less dynamic, more monotonous speech production.' },
  { name: 'mfcc_delta_mean', label: 'MFCC Delta — Average Rate of Change', description: 'The average rate at which the MFCCs change from frame to frame — capturing how dynamically the spectral characteristics of speech shift over time.', method: 'librosa.feature.delta() applied to MFCC matrix; mean of delta features across all frames', asdRelevance: 'Low MFCC delta values indicate slow-changing, less dynamic spectral transitions — associated with less varied articulation in ASD.' },
  { name: 'mfcc_delta_std', label: 'MFCC Delta — Variability', description: 'How much the rate of spectral change varies — whether transitions are consistently smooth or occasionally abrupt.', method: 'Standard deviation of MFCC delta features across all frames', asdRelevance: 'Captures inconsistency in articulatory transitions; ASD speech may show atypical variability patterns.' },
  { name: 'mfcc_delta2_mean', label: 'MFCC Delta-Delta — Acceleration', description: 'The second derivative of the MFCCs — how fast the rate of change itself changes. This captures the acceleration of spectral transitions.', method: 'librosa.feature.delta(order=2) applied to MFCC matrix; mean across all frames', asdRelevance: 'Delta-delta features encode articulatory acceleration; differences in ASD may reflect motor speech coordination differences.' },
  { name: 'mfcc_delta2_std', label: 'MFCC Delta-Delta — Variability', description: 'How variable the acceleration of spectral transitions is.', method: 'Standard deviation of second-order delta MFCC features', asdRelevance: 'Irregular articulatory acceleration may indicate motor speech differences in ASD.' },
  { name: 'mfcc_covariance_trace', label: 'MFCC Covariance (Overall Spectral Complexity)', description: 'A single number summarising how much the different MFCC coefficients move together — capturing overall spectral complexity and variability.', method: 'Trace (sum of diagonal) of the MFCC covariance matrix computed across all frames', asdRelevance: 'Children with ASD may show reduced overall spectral complexity, consistent with less varied articulation and more monotone speech.' },
];

const SPECTRAL_FEATURES: Feature[] = [
  { name: 'spectral_centroid_mean', label: 'Average Spectral Centroid (Brightness)', description: 'The "centre of mass" of the sound spectrum — essentially how bright or dark the voice sounds. A high centroid means the energy is concentrated in higher frequencies (brighter, sharper); a low centroid means energy is in lower frequencies (darker, fuller).', method: 'librosa.feature.spectral_centroid(); mean across all frames', asdRelevance: 'Spectral centroid reflects overall vocal quality and articulation; atypical brightness has been documented in ASD speech samples.' },
  { name: 'spectral_centroid_std', label: 'Spectral Centroid Variability', description: 'How much the brightness of the voice varies over time.', method: 'Standard deviation of spectral centroid across frames', asdRelevance: 'Low variability indicates monotone, less expressive speech.' },
  { name: 'spectral_rolloff_mean', label: 'Average Spectral Rolloff (Energy Distribution)', description: 'The frequency below which 85% of the signal energy falls — a measure of how energy is spread across high and low frequencies.', method: 'librosa.feature.spectral_rolloff(roll_percent=0.85); mean across all frames', asdRelevance: 'Rolloff captures the balance of low vs. high frequency energy, reflecting articulatory precision.' },
  { name: 'spectral_rolloff_std', label: 'Spectral Rolloff Variability', description: 'How much the high-to-low frequency energy balance varies over time.', method: 'Standard deviation of spectral rolloff across frames', asdRelevance: 'High variability may indicate inconsistent articulatory effort.' },
  { name: 'spectral_bandwidth_mean', label: 'Average Spectral Bandwidth', description: 'How wide the energy is spread across the frequency spectrum — a narrow bandwidth means energy is concentrated at a few frequencies; a wide bandwidth means energy is spread across many.', method: 'librosa.feature.spectral_bandwidth(); mean across all frames', asdRelevance: 'Unusual spectral bandwidth (very narrow or very wide) can indicate atypical vocal tract configuration.' },
  { name: 'spectral_bandwidth_std', label: 'Spectral Bandwidth Variability', description: 'How much the width of the energy distribution varies over time.', method: 'Standard deviation of spectral bandwidth across frames', asdRelevance: 'Low bandwidth variability is consistent with less dynamic articulation patterns.' },
  { name: 'spectral_contrast_mean', label: 'Average Spectral Contrast', description: 'The difference in energy between peaks and valleys across different frequency bands — a measure of how "clear" or "rich" the spectral structure of the voice is.', method: 'librosa.feature.spectral_contrast(); mean across frequency bands and time frames', asdRelevance: 'Reduced spectral contrast may indicate less clear formant structure, consistent with atypical articulation.' },
  { name: 'zero_crossing_rate_mean', label: 'Average Zero-Crossing Rate', description: 'How often the audio signal crosses from positive to negative (or vice versa) per second — a measure related to the noisiness or tonality of the speech signal.', method: 'librosa.feature.zero_crossing_rate(); mean across all frames', asdRelevance: 'High zero-crossing rates are associated with fricatives and noisy segments; unusual rates may reflect articulatory differences.' },
  { name: 'zero_crossing_rate_std', label: 'Zero-Crossing Rate Variability', description: 'How much the noisiness/tonality of the speech signal varies over time.', method: 'Standard deviation of zero-crossing rate across frames', asdRelevance: 'Captures variability in speech segment types; low variability may indicate restricted phonetic inventory.' },
  { name: 'spectral_flatness_mean', label: 'Average Spectral Flatness (Tonality)', description: 'How similar the spectral energy distribution is to white noise — a flat spectrum (score near 1) means the speech is noisy and unstructured; a peaked spectrum (score near 0) means it is tonal and voiced.', method: 'librosa.feature.spectral_flatness(); mean across all frames', asdRelevance: 'High spectral flatness may indicate breathy or noisy voice quality, which has been associated with certain ASD vocal profiles.' },
];

const VOICE_QUALITY_FEATURES: Feature[] = [
  { name: 'jitter_local', label: 'Jitter (Pitch Irregularity)', description: 'How much the pitch period (time between each vocal fold vibration) varies from one cycle to the next — a measure of voice stability and smoothness.', method: 'Computed from F0 contour: mean absolute difference between consecutive F0 periods, normalised by mean period length', asdRelevance: 'Elevated jitter indicates an unstable, rough voice quality; voice quality differences including jitter have been documented in ASD.' },
  { name: 'shimmer_local', label: 'Shimmer (Amplitude Irregularity)', description: 'How much the amplitude (loudness) of each vocal fold vibration cycle varies from the previous one — a measure of voice smoothness and consistency.', method: 'Computed from RMS energy contour: mean absolute difference between consecutive amplitude values, normalised by mean amplitude', asdRelevance: 'Elevated shimmer indicates an unstable, breathy voice quality; combined with jitter, captures overall voice perturbation.' },
  { name: 'hnr_mean', label: 'Harmonics-to-Noise Ratio (Voice Clarity)', description: 'How much of the voice signal is periodic (harmonic, tonal) vs. random noise — a high ratio means a clear, smooth voice; a low ratio means a breathy, rough, or hoarse voice.', method: 'Estimated via autocorrelation peak ratio: maximum autocorrelation value divided by zero-lag value, converted to dB', asdRelevance: 'Low HNR indicates breathiness or roughness; HNR differences have been found in ASD speech and may reflect hypotonia or laryngeal differences.' },
  { name: 'hnr_std', label: 'Voice Clarity Variability', description: 'How much the clarity of the voice changes over time.', method: 'Standard deviation of frame-level HNR estimates', asdRelevance: 'High variability in voice quality may indicate inconsistent phonation effort.' },
  { name: 'cpp_mean', label: 'Cepstral Peak Prominence (Voice Strength)', description: 'A measure of how strongly the periodic component of the voice stands out from the background noise — often considered the most robust single measure of voice quality.', method: 'Computed from real cepstrum: prominence of cepstral peak at F0 lag relative to regression line', asdRelevance: 'Low CPP is strongly associated with breathy/dysphonic voice quality; differences in ASD may reflect motor control of the larynx.' },
];
const FORMANT_FEATURES: Feature[] = [
  { name: 'f1_mean', label: 'First Formant (F1) — Average', description: 'The average frequency of the first resonance of the vocal tract — F1 is primarily controlled by jaw height and corresponds to vowel openness (low F1 = closed vowels like "ee"; high F1 = open vowels like "ah").', method: 'Estimated via LPC (Linear Predictive Coding) analysis using scipy.signal; peak detection in LPC spectral envelope; mean across all voiced frames', asdRelevance: 'Formant differences reflect differences in articulatory postures; systematic F1 shifts have been found in ASD, potentially reflecting reduced vowel space and less precise articulation.' },
  { name: 'f1_std', label: 'F1 Variability', description: 'How much the first formant varies over time — reflecting how dynamically the jaw opens and closes during speech.', method: 'Standard deviation of F1 estimates across voiced frames', asdRelevance: 'Low F1 variability indicates less dynamic jaw movement, consistent with reduced articulatory precision.' },
  { name: 'f2_mean', label: 'Second Formant (F2) — Average', description: 'The average frequency of the second vocal tract resonance — F2 is primarily controlled by tongue advancement (back vs. front) and is the key distinguisher between front vowels like "ee" and back vowels like "oo".', method: 'LPC peak detection; second resonance frequency; mean across voiced frames', asdRelevance: 'F2 is the most diagnostically sensitive formant for vowel space; reduced F2 range indicates centralised, less distinct vowel production — documented in ASD.' },
  { name: 'f2_std', label: 'F2 Variability', description: 'How much the second formant varies — a direct measure of tongue front-back movement range during speech.', method: 'Standard deviation of F2 estimates across voiced frames', asdRelevance: 'The combination of F1 and F2 variability defines the "vowel space" — reduced vowel space area is a robust ASD marker.' },
  { name: 'f3_mean', label: 'Third Formant (F3) — Average', description: 'The average frequency of the third vocal tract resonance — F3 is associated with tongue tip position and contributes to distinguishing sounds like "r" and "l".', method: 'LPC peak detection; third resonance frequency; mean across voiced frames', asdRelevance: 'F3 differences contribute to the overall acoustic profile; unusual F3 values may reflect atypical tongue-tip articulation patterns.' },
  { name: 'f3_std', label: 'F3 Variability', description: 'How much the third formant varies over time.', method: 'Standard deviation of F3 estimates across voiced frames', asdRelevance: 'Together with F1 and F2 variability, F3 variability characterises the full articulatory dynamism of the speech.' },
  { name: 'f1_f2_ratio', label: 'F1/F2 Ratio (Vowel Space Shape)', description: 'The ratio of the first to second formant — a simple single-number summary of the overall vowel space configuration.', method: 'Mean F1 divided by mean F2 across all voiced frames', asdRelevance: 'A compressed F1/F2 ratio indicates more centralised vowel production — a consistent finding in ASD articulatory research.' },
  { name: 'vowel_space_area', label: 'Vowel Space Area', description: 'A measure of how large and distinct the vowel space is — computed from the spread of F1 and F2 values. A larger area means more clearly differentiated vowels; a smaller area means vowels are produced more similarly to each other.', method: 'Area of convex hull of (F1, F2) data points across all voiced frames', asdRelevance: 'Reduced vowel space area is one of the most robust acoustic findings in ASD — children with ASD produce less distinct vowels, making speech less intelligible.' },
  { name: 'formant_dispersion', label: 'Formant Dispersion', description: 'How evenly spaced the formants are — a measure related to vocal tract length and configuration.', method: 'Mean distance between successive formant frequencies (F2-F1, F3-F2, F4-F3)', asdRelevance: 'Unusual formant dispersion may reflect atypical vocal tract configuration or resonance patterns.' },
  { name: 'f4_mean', label: 'Fourth Formant (F4) — Average', description: 'The fourth resonance of the vocal tract — higher formants carry subtler speaker-identity and vocal quality information.', method: 'LPC peak detection; fourth resonance frequency; mean across voiced frames', asdRelevance: 'F4 contributes to the overall vocal quality profile and speaker-specific characteristics.' },
  { name: 'formant_bandwidth_f1', label: 'F1 Bandwidth (Vowel Definition)', description: 'How broad or narrow the first formant resonance peak is — a narrow bandwidth means a well-defined, clear resonance; a wide bandwidth means a less distinct, more diffuse resonance.', method: 'Half-power bandwidth of F1 LPC peak', asdRelevance: 'Wide formant bandwidths indicate less precise resonance patterns, potentially related to reduced articulatory tension or coordination.' },
];

const ENERGY_FEATURES: Feature[] = [
  { name: 'rms_mean', label: 'Average Speech Energy (Loudness)', description: 'The average root-mean-square energy of the speech signal — the best single measure of how loud the speech is overall.', method: 'librosa.feature.rms(); mean of RMS energy across all frames', asdRelevance: 'Unusually quiet or unusually loud speech are both documented in ASD; RMS energy provides the baseline loudness measure.' },
  { name: 'rms_std', label: 'Loudness Variability', description: 'How much the loudness of the speech varies over time.', method: 'Standard deviation of RMS energy across all frames', asdRelevance: 'Low loudness variability indicates flat, monotone speech delivery with reduced prosodic emphasis.' },
  { name: 'rms_max', label: 'Peak Loudness', description: 'The loudest moment in the recording.', method: 'Maximum RMS energy value across all frames', asdRelevance: 'Combined with mean and minimum, characterises the dynamic range of the child\'s speech.' },
  { name: 'rms_min', label: 'Minimum Loudness', description: 'The quietest voiced moment in the recording.', method: 'Minimum RMS energy across voiced frames (above silence threshold)', asdRelevance: 'Very quiet speech may indicate reduced confidence or engagement.' },
  { name: 'dynamic_range', label: 'Dynamic Range', description: 'The span from quietest to loudest — how much the voice varies in intensity throughout the recording.', method: 'rms_max minus rms_min (in dB)', asdRelevance: 'Reduced dynamic range indicates less use of loudness variation for emphasis, consistent with monotone speech in ASD.' },
  { name: 'energy_skewness', label: 'Energy Distribution Shape', description: 'Whether energy levels are skewed — whether most speech is at a consistent level with rare loud moments, or the other way around.', method: 'Statistical skewness of the RMS energy distribution across frames', asdRelevance: 'Asymmetric energy distributions may reflect atypical prosodic emphasis patterns.' },
];

const RHYTHM_FEATURES: Feature[] = [
  { name: 'tempo', label: 'Speech Tempo (BPM)', description: 'The estimated tempo of the speech in beats per minute — capturing the overall rhythmic rate of speech production.', method: 'librosa.beat.tempo() applied to onset strength envelope', asdRelevance: 'Atypical speech tempo (both faster and slower than typical) has been documented in ASD; rhythm differences are a hallmark of ASD prosody.' },
  { name: 'onset_rate', label: 'Speech Onset Rate', description: 'How many new speech sounds begin per second — a measure of the rate at which the mouth starts new acoustic events.', method: 'librosa.onset.onset_detect() applied to audio; onset count divided by total duration', asdRelevance: 'Reduced onset rate may indicate slower, more deliberate speech; high rates may indicate rapid, pressured speech.' },
  { name: 'onset_strength_mean', label: 'Average Onset Strength', description: 'On average, how strongly each new speech sound begins — whether articulation starts with clear, strong onsets or soft, gradual ones.', method: 'librosa.onset.onset_strength(); mean across all frames', asdRelevance: 'Weak onset strength may indicate reduced articulatory precision and effort.' },
  { name: 'onset_strength_std', label: 'Onset Strength Variability', description: 'How much the strength of speech onsets varies throughout the recording.', method: 'Standard deviation of onset strength across frames', asdRelevance: 'Low variability indicates consistent but potentially monotone articulation.' },
  { name: 'silence_proportion', label: 'Proportion of Silence in Audio', description: 'What fraction of the total audio file is silent — not just inter-turn silence, but all silence including within-utterance pauses.', method: 'Proportion of frames with RMS energy below 10% of mean RMS energy', asdRelevance: 'High silence proportion confirms reduced verbal participation and frequent mid-speech pausing.' },
  { name: 'speech_rate_local', label: 'Local Speech Rate Variability', description: 'How much the speed of speech varies within segments — whether the child speaks at a consistent pace or accelerates and decelerates within utterances.', method: 'Standard deviation of per-utterance speech rates (syllable count / duration)', asdRelevance: 'Unusual speech rate variability — either very regular (robotic) or very irregular — is associated with atypical prosody in ASD.' },
];

const CHROMA_FEATURES: Feature[] = [
  { name: 'chroma_mean', label: 'Average Chroma Features', description: 'The average distribution of energy across the 12 musical pitch classes (C, C#, D, ...) — capturing harmonic characteristics of the speech.', method: 'librosa.feature.chroma_stft() with 12 bins; mean of each chroma bin across all frames', asdRelevance: 'While chroma features are primarily designed for music, they capture harmonic regularities in speech that differ between typical and atypical vocal production.' },
  { name: 'chroma_std', label: 'Chroma Variability', description: 'How much the harmonic content of the speech varies over time.', method: 'Standard deviation of chroma features across frames', asdRelevance: 'Low chroma variability indicates less harmonic variation in the voice, consistent with less dynamic prosody.' },
  { name: 'chroma_cqt_mean', label: 'Constant-Q Chroma (Harmonic Precision)', description: 'A more precise version of chroma analysis using a constant-Q transform — capturing harmonic structure with better frequency resolution.', method: 'librosa.feature.chroma_cqt(); mean across all frames', asdRelevance: 'CQT chroma captures finer harmonic regularities in the voice, providing a more sensitive measure of tonal variation.' },
  { name: 'chroma_cens_mean', label: 'CENS Chroma (Normalised Harmonic Summary)', description: 'A smoothed, normalised version of chroma features that removes short-term variations — capturing the stable harmonic character of the voice.', method: 'librosa.feature.chroma_cens(); mean across all frames', asdRelevance: 'CENS provides a robust summary of long-term harmonic patterns, complementing the more transient standard chroma.' },
  { name: 'tonnetz_mean', label: 'Tonal Centroid (Harmonic Space Position)', description: 'A measure of where the speech falls in tonal space — capturing the balance of harmonic relationships in the voice.', method: 'librosa.feature.tonnetz(); mean across 6 tonal dimensions across all frames', asdRelevance: 'Tonal centroid features capture harmonic regularities that may differ systematically between ASD and TD speech.' },
  { name: 'tonnetz_std', label: 'Tonal Centroid Variability', description: 'How much the harmonic space position varies over time.', method: 'Standard deviation of tonnetz features across frames', asdRelevance: 'Captures dynamic harmonic variation — low variability reflects less tonal movement in speech.' },
];

// ── CATEGORY METADATA ────────────────────────────────────────────────────────
interface FeatureCategory {
  id: string;
  label: string;
  count: number;
  icon: string;
  color: string;
  bgColor: string;
  borderColor: string;
  summary: string;
  method: string;
  features: Feature[];
}

const PRAGMATIC_CATEGORIES: FeatureCategory[] = [
  {
    id: 'turn_taking',
    label: 'Turn-Taking',
    count: 44,
    icon: '',
    color: 'text-emerald-700',
    bgColor: 'bg-emerald-50',
    borderColor: 'border-emerald-200',
    summary: 'Measures the back-and-forth rhythm of conversation — who speaks when, how long, how often, and how quickly speakers respond to each other.',
    method: 'CHAT transcript speaker-tier parsing with optional audio timing from %snd annotations or Whisper timestamps.',
    features: TURN_TAKING_FEATURES,
  },
  {
    id: 'topic_coherence',
    label: 'Topic Coherence',
    count: 31,
    icon: '',
    color: 'text-sky-700',
    bgColor: 'bg-sky-50',
    borderColor: 'border-sky-200',
    summary: 'Measures how well the conversation stays on topic — whether the child follows the adult\'s lead, how often topics shift, and how varied or restricted the topics are.',
    method: 'spaCy word vector cosine similarity between consecutive utterances + LDA topic modelling (5 topics) via scikit-learn.',
    features: TOPIC_COHERENCE_FEATURES,
  },
  {
    id: 'pause_latency',
    label: 'Pause & Latency',
    count: 47,
    icon: '',
    color: 'text-amber-700',
    bgColor: 'bg-amber-50',
    borderColor: 'border-amber-200',
    summary: 'Measures how timing, silence, and hesitation patterns play out — how long it takes to respond, how many fillers are used, and how fluent the speech is overall.',
    method: 'GMM clustering on ASDBank corpus identified 3 latency clusters: Rapid (~0.2s), Processing (~1.25s), Disengaged (~4.32s). Thresholds: 0.45s (normal), 2.0s (long), 4.0s (very long). CHAT pause markers (.),(..),(...).(pause) decoded to durations.',
    features: PAUSE_LATENCY_FEATURES,
  },
  {
    id: 'repair_detection',
    label: 'Conversational Repair',
    count: 38,
    icon: '',
    color: 'text-rose-700',
    bgColor: 'bg-rose-50',
    borderColor: 'border-rose-200',
    summary: 'Measures how communication breakdowns are handled — how often things go wrong, who initiates fixing them, and whether they get resolved.',
    method: 'CHAT retrace markers [/] [//] [///] [?] + pattern matching for repair-initiating phrases + sequence analysis to classify repair type and outcome.',
    features: REPAIR_DETECTION_FEATURES,
  },
  {
    id: 'pragmatic_linguistic',
    label: 'Pragmatic Linguistic',
    count: 35,
    icon: '',
    color: 'text-violet-700',
    bgColor: 'bg-violet-50',
    borderColor: 'border-violet-200',
    summary: 'Measures the language the child produces — vocabulary diversity, echolalia, pronoun use, question patterns, and social language markers.',
    method: 'CHAT transcript text analysis with spaCy POS tagging, MLU counting from raw token and %mor tier, echolalia detection via lexical overlap, pronoun pattern matching.',
    features: PRAGMATIC_LINGUISTIC_FEATURES,
  },
  {
    id: 'audio_pragmatic',
    label: 'Audio-Derived Timing',
    count: 30,
    icon: '',
    color: 'text-orange-700',
    bgColor: 'bg-orange-50',
    borderColor: 'border-orange-200',
    summary: 'Measures pause and timing patterns extracted directly from the audio signal — independent of the transcript, these confirm and complement the text-based measures.',
    method: 'Librosa energy envelope analysis with RMS silence thresholding. Speech/silence segmentation at frame level. Speaking rate from word count divided by active speech duration.',
    features: AUDIO_PRAGMATIC_FEATURES,
  },
];
// ── ACOUSTIC CATEGORY METADATA ───────────────────────────────────────────────
interface FeatureCategory {
  id: string;
  label: string;
  count: number;
  color: string;
  bgColor: string;
  borderColor: string;
  summary: string;
  method: string;
  features: Feature[];
}

const ACOUSTIC_CATEGORIES: FeatureCategory[] = [
  {
    id: 'pitch',
    label: 'Pitch (Fundamental Frequency)',
    count: 10,
    color: 'text-sky-700',
    bgColor: 'bg-sky-50',
    borderColor: 'border-sky-200',
    summary: 'Measures the fundamental frequency (F0) of the voice — how high or low the voice is, how much it varies, and the overall pitch range. Pitch is the primary carrier of prosodic information.',
    method: 'pyin probabilistic YIN algorithm via librosa.pyin() — the most accurate open-source pitch estimator, using probabilistic thresholding to distinguish voiced from unvoiced frames at 10ms resolution.',
    features: PITCH_FEATURES,
  },
  {
    id: 'mfcc',
    label: 'MFCCs (Spectral Envelope)',
    count: 26,
    color: 'text-indigo-700',
    bgColor: 'bg-indigo-50',
    borderColor: 'border-indigo-200',
    summary: 'Mel-frequency cepstral coefficients — the standard representation of the overall spectral shape (vocal tract filter) of speech. MFCCs capture the "fingerprint" of articulation and vocal quality in a compact set of numbers.',
    method: 'librosa.feature.mfcc() with n_mfcc=13, n_fft=2048, hop_length=512, mel filterbank. Delta and delta-delta features computed via librosa.feature.delta(). Mean and standard deviation computed for each coefficient and its derivatives.',
    features: MFCC_FEATURES,
  },
  {
    id: 'spectral',
    label: 'Spectral Features',
    count: 10,
    color: 'text-teal-700',
    bgColor: 'bg-teal-50',
    borderColor: 'border-teal-200',
    summary: 'Measures the frequency-domain properties of the speech signal — where the energy sits in the spectrum, how spread out it is, and how tonal vs. noisy the voice sounds.',
    method: 'librosa spectral feature functions: spectral_centroid(), spectral_rolloff(), spectral_bandwidth(), spectral_contrast(), spectral_flatness(), zero_crossing_rate(). All computed per frame with mean and std aggregation.',
    features: SPECTRAL_FEATURES,
  },
  {
    id: 'voice_quality',
    label: 'Voice Quality',
    count: 5,
    color: 'text-rose-700',
    bgColor: 'bg-rose-50',
    borderColor: 'border-rose-200',
    summary: 'Measures the smoothness and regularity of vocal fold vibration — capturing breathiness, roughness, and overall phonation quality that are independent of pitch and loudness.',
    method: 'Jitter and shimmer computed from F0 contour and RMS energy contour respectively. HNR estimated via autocorrelation peak ratio. CPP from cepstral analysis of the speech signal.',
    features: VOICE_QUALITY_FEATURES,
  },
  {
    id: 'formants',
    label: 'Formants (Vowel Space)',
    count: 11,
    color: 'text-amber-700',
    bgColor: 'bg-amber-50',
    borderColor: 'border-amber-200',
    summary: 'Measures the resonance frequencies of the vocal tract — the characteristic peaks in the spectrum that define vowel quality and articulatory precision. The vowel space defined by F1 and F2 is one of the most diagnostically sensitive acoustic measures.',
    method: 'Linear Predictive Coding (LPC) analysis via scipy.signal.lpc() with order 2 + sample_rate/1000. Peak detection on LPC spectral envelope to identify formant frequencies. Computed on voiced frames only.',
    features: FORMANT_FEATURES,
  },
  {
    id: 'energy',
    label: 'Energy & Intensity',
    count: 6,
    color: 'text-orange-700',
    bgColor: 'bg-orange-50',
    borderColor: 'border-orange-200',
    summary: 'Measures the loudness and dynamic intensity of the speech signal — how loud the overall speech is, how much it varies, and what the dynamic range is.',
    method: 'librosa.feature.rms() computed with frame_length=2048, hop_length=512. Statistical aggregation (mean, std, max, min) across all frames. Dynamic range computed from max to min in dB scale.',
    features: ENERGY_FEATURES,
  },
  {
    id: 'rhythm',
    label: 'Rhythm & Timing',
    count: 6,
    color: 'text-green-700',
    bgColor: 'bg-green-50',
    borderColor: 'border-green-200',
    summary: 'Measures the temporal structure of the speech — how fast speech is produced, how often new sounds begin, and what proportion of the recording is silence.',
    method: 'librosa.beat.tempo() on onset strength envelope. librosa.onset.onset_detect() for onset events. Silence detection via RMS energy thresholding at 10% of mean energy. Per-utterance speech rate from syllable count estimates.',
    features: RHYTHM_FEATURES,
  },
  {
    id: 'chroma',
    label: 'Chroma & Tonal Features',
    count: 6,
    color: 'text-purple-700',
    bgColor: 'bg-purple-50',
    borderColor: 'border-purple-200',
    summary: 'Measures the harmonic and tonal structure of the speech signal — capturing which pitch classes are prominent and how the harmonic content moves over time.',
    method: 'librosa.feature.chroma_stft(), chroma_cqt(), chroma_cens(), and tonnetz(). All computed per frame using the short-time Fourier transform, constant-Q transform, and energy normalised variants respectively.',
    features: CHROMA_FEATURES,
  },
];
// ── SECTION NAV CONFIG ───────────────────────────────────────────────────────
const NAV_SECTIONS = [
  { id: 'overview', label: 'System Overview', icon: '◆' },
  { id: 'pipeline', label: 'The Pipeline', icon: '→' },
  { id: 'input', label: '1 · Input Layer', icon: '↑' },
  { id: 'extraction', label: '2 · Feature Extraction', icon: '⊙' },
  { id: 'pragmatic', label: '   Pragmatic Detail', icon: '◦' },
  { id: 'training', label: '3 · Model Training', icon: '▣' },
  { id: 'fusion', label: '4 · Prediction & Fusion', icon: '◈' },
  { id: 'interpretability', label: '5 · Interpretability', icon: '◉' },
];

// ── FEATURE TABLE MODAL ──────────────────────────────────────────────────────
function FeatureTableModal({ category, onClose }: { category: FeatureCategory; onClose: () => void }) {
  const [search, setSearch] = useState('');
  const filtered = category.features.filter(f =>
    f.label.toLowerCase().includes(search.toLowerCase()) ||
    f.name.toLowerCase().includes(search.toLowerCase()) ||
    f.description.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center bg-black/60 backdrop-blur-sm p-4 overflow-y-auto" onClick={onClose}>
      <div
        className="bg-white rounded-2xl shadow-2xl w-full max-w-6xl my-8"
        onClick={e => e.stopPropagation()}
      >
        {/* Modal header */}
        <div className={`${category.bgColor} ${category.borderColor} border-b rounded-t-2xl px-8 py-6 flex items-start justify-between`}>
          <div>
            <div className="flex items-center gap-3 mb-2">
              <span className="text-2xl">{category.icon}</span>
              <h2 className="text-2xl font-bold text-gray-900">{category.label} Features</h2>
              <span className={`text-sm font-semibold px-3 py-1 rounded-full ${category.bgColor} ${category.color} border ${category.borderColor}`}>
                {category.count} features
              </span>
            </div>
            <p className="text-sm text-gray-600 max-w-3xl">{category.summary}</p>
          </div>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-700 text-3xl leading-none ml-4 mt-1">×</button>
        </div>

        {/* Search */}
        <div className="px-8 py-4 border-b border-gray-100">
          <input
            type="text"
            placeholder="Search features..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            className="w-full px-4 py-2 rounded-lg border border-gray-200 text-sm focus:outline-none focus:border-gray-400 bg-gray-50"
          />
          {search && (
            <p className="text-xs text-gray-400 mt-2">{filtered.length} of {category.features.length} features shown</p>
          )}
        </div>

        {/* Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-50 border-b border-gray-200">
                <th className="text-left px-6 py-3 font-semibold text-gray-600 w-48">Feature Name</th>
                <th className="text-left px-6 py-3 font-semibold text-gray-600 w-48">Plain English Name</th>
                <th className="text-left px-6 py-3 font-semibold text-gray-600">What It Measures</th>
                <th className="text-left px-6 py-3 font-semibold text-gray-600 w-64">How We Detect It</th>
                <th className="text-left px-6 py-3 font-semibold text-gray-600 w-64">Why It Matters for ASD</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((f, i) => (
                <tr key={f.name} className={`border-b border-gray-100 ${i % 2 === 0 ? 'bg-white' : 'bg-gray-50/50'} hover:bg-blue-50/40 transition-colors`}>
                  <td className="px-6 py-3 font-mono text-xs text-gray-500 align-top">{f.name}</td>
                  <td className="px-6 py-3 font-semibold text-gray-900 align-top">{f.label}</td>
                  <td className="px-6 py-3 text-gray-700 align-top leading-relaxed">{f.description}</td>
                  <td className="px-6 py-3 text-gray-600 align-top text-xs leading-relaxed italic">{f.method}</td>
                  <td className="px-6 py-3 text-gray-700 align-top leading-relaxed">{f.asdRelevance}</td>
                </tr>
              ))}
              {filtered.length === 0 && (
                <tr><td colSpan={5} className="text-center py-12 text-gray-400">No features match your search.</td></tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Modal footer */}
        <div className="px-8 py-4 bg-gray-50 rounded-b-2xl border-t border-gray-200 flex justify-between items-center">
          <p className="text-xs text-gray-400">Source: <code className="bg-gray-200 px-1 rounded">src/features/pragmatic_conversational/</code></p>
          <button onClick={onClose} className="px-5 py-2 bg-gray-900 text-white rounded-lg text-sm hover:bg-gray-700 transition-colors">Close</button>
        </div>
      </div>
    </div>
  );
}

// ── MAIN PAGE ────────────────────────────────────────────────────────────────
export default function HowItWorksPage() {
  const router = useRouter();
  const [activeSection, setActiveSection] = useState('overview');
  const [openCategory, setOpenCategory] = useState<FeatureCategory | null>(null);
  const sectionRefs = useRef<Record<string, HTMLElement | null>>({});

  useEffect(() => {
    const observer = new IntersectionObserver(
      entries => {
        entries.forEach(entry => {
          if (entry.isIntersecting) setActiveSection(entry.target.id);
        });
      },
      { rootMargin: '-20% 0px -70% 0px' }
    );
    Object.values(sectionRefs.current).forEach(el => { if (el) observer.observe(el); });
    return () => observer.disconnect();
  }, []);

  const scrollTo = (id: string) => {
    sectionRefs.current[id]?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  const setRef = (id: string) => (el: HTMLElement | null) => { sectionRefs.current[id] = el; };

  return (
    <div className="bg-white min-h-screen">
      {/* ── HEADER ── */}
      <header className="sticky top-0 z-40 bg-white/90 backdrop-blur-md border-b border-gray-200">
        <div className="max-w-screen-xl mx-auto px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button onClick={() => router.push('/')} className="text-gray-400 hover:text-gray-700 transition-colors text-sm flex items-center gap-1">
              ← Back
            </button>
            <div className="h-5 w-px bg-gray-200" />
            <span className="text-lg font-semibold text-gray-900 tracking-tight">Artistic.</span>
            <div className="h-5 w-px bg-gray-200" />
            <span className="text-sm text-gray-500">How It Works</span>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-xs text-gray-400 hidden md:block">ASD Detection System — Full Technical Documentation</span>
            <button onClick={() => router.push('/guideline')} className="px-4 py-1.5 text-xs bg-gray-100 text-gray-600 rounded-full hover:bg-gray-200 transition-colors">
              Feature Guide
            </button>
          </div>
        </div>
      </header>

      <div className="max-w-screen-xl mx-auto flex">
        {/* ── STICKY SIDEBAR NAV ── */}
        <aside className="hidden lg:block w-56 flex-shrink-0 sticky top-16 h-[calc(100vh-4rem)] overflow-y-auto py-8 pr-4">
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-widest mb-4 px-3">Contents</p>
          <nav className="space-y-0.5">
            {NAV_SECTIONS.map(s => (
              <button
                key={s.id}
                onClick={() => scrollTo(s.id)}
                className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-all flex items-center gap-2 ${
                  activeSection === s.id
                    ? 'bg-gray-900 text-white font-medium'
                    : 'text-gray-500 hover:text-gray-900 hover:bg-gray-100'
                }`}
              >
                <span className="text-xs opacity-60 w-4">{s.icon}</span>
                <span className={s.id === 'acoustic' || s.id === 'pragmatic' ? 'text-xs' : ''}>{s.label}</span>
              </button>
            ))}
          </nav>
        </aside>

        {/* ── MAIN CONTENT ── */}
        <main className="flex-1 min-w-0 py-12 px-6 lg:px-12 space-y-24">

          {/* ── SECTION: OVERVIEW ── */}
          <section id="overview" ref={setRef('overview')}>
            <div className="max-w-3xl">
              <div className="inline-flex items-center gap-2 bg-gray-100 text-gray-600 text-xs font-semibold px-3 py-1 rounded-full mb-6 uppercase tracking-widest">
                System Overview
              </div>
              <h1 className="text-5xl font-bold text-gray-900 tracking-tight leading-tight mb-6">
                How Artistic<br />detects ASD
              </h1>
              <p className="text-xl text-gray-500 leading-relaxed mb-8">
                Artistic is an AI-powered speech analysis platform that analyses recorded conversations and transcripts to surface patterns associated with Autism Spectrum Disorder (ASD). It extracts over 400 measurable features from speech, then uses machine learning to produce a transparent, explainable prediction.
              </p>

            </div>
          </section>

          {/* ── SECTION: PIPELINE ── */}
          <section id="pipeline" ref={setRef('pipeline')}>
            <div className="inline-flex items-center gap-2 bg-gray-100 text-gray-600 text-xs font-semibold px-3 py-1 rounded-full mb-6 uppercase tracking-widest">
              The Pipeline
            </div>
            <h2 className="text-3xl font-bold text-gray-900 mb-3">End-to-end flow</h2>
            <p className="text-gray-500 mb-10 max-w-2xl">
              Every analysis follows the same five-stage pipeline. Each stage builds on the previous one — from raw files all the way to an explainable prediction.
            </p>
            <div className="flex flex-col md:flex-row items-stretch gap-1">
              {[
                { step: '01', title: 'Input', desc: 'Audio or CHAT transcript file uploaded', color: 'bg-gray-900', text: 'text-white' },
                { step: '02', title: 'Extraction', desc: '400+ features from 3 signal types', color: 'bg-emerald-800', text: 'text-white' },
                { step: '03', title: 'Training', desc: 'ML models learn ASD vs TD patterns', color: 'bg-sky-800', text: 'text-white' },
                { step: '04', title: 'Fusion', desc: '3 component scores weighted & combined', color: 'bg-violet-800', text: 'text-white' },
                { step: '05', title: 'Explain', desc: 'SHAP + counterfactuals show why', color: 'bg-rose-800', text: 'text-white' },
              ].map((p, i, arr) => (
                <div key={p.step} className="flex md:flex-col items-center md:items-stretch flex-1">
                  <div className={`${p.color} ${p.text} rounded-2xl p-5`}>
                    <div className="text-xs font-mono opacity-60 mb-2">{p.step}</div>
                    <div className="text-lg font-bold mb-1">{p.title}</div>
                    <div className="text-sm opacity-80">{p.desc}</div>
                  </div>
                  {i < arr.length - 1 && (
                    <div className="flex-shrink-0 flex items-center justify-center md:rotate-90 w-8 h-8 md:w-full md:h-8 text-gray-300 text-xl">→</div>
                  )}
                </div>
              ))}
            </div>
            <div className="mt-8 bg-gray-50 border border-gray-200 rounded-2xl p-6">
              <p className="text-sm font-semibold text-gray-700 mb-3">Two input paths into the same pipeline</p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-white rounded-xl border border-gray-200 p-4">
                  <div className="flex items-center gap-3 mb-2">
                    <div className="w-8 h-8 bg-sky-100 rounded-lg flex items-center justify-center text-sky-700 font-bold text-sm">
                      <IconMic className="w-4 h-4" />
                    </div>
                    <span className="font-semibold text-gray-900 text-sm">Audio file (WAV / MP3 / FLAC)</span>
                  </div>
                  <p className="text-sm text-gray-500">Whisper AI transcribes the speech → transcript used for pragmatic + syntactic features. Audio signal used directly for acoustic + audio-pragmatic features.</p>
                </div>
                <div className="bg-white rounded-xl border border-gray-200 p-4">
                  <div className="flex items-center gap-3 mb-2">
                    <div className="w-8 h-8 bg-emerald-100 rounded-lg flex items-center justify-center text-emerald-700 font-bold text-sm">
                      <IconTranscript className="w-4 h-4" />
                    </div>
                    <span className="font-semibold text-gray-900 text-sm">CHAT transcript (.cha file)</span>
                  </div>
                  <p className="text-sm text-gray-500">CLAN CHAT format with speaker tiers, timing (%snd), and morphology (%mor) — parsed directly for pragmatic and syntactic features.</p>
                </div>
              </div>
            </div>
          </section>

          {/* ── SECTION: INPUT ── */}
          <section id="input" ref={setRef('input')}>
            <div className="inline-flex items-center gap-2 bg-gray-900 text-white text-xs font-semibold px-3 py-1 rounded-full mb-6 uppercase tracking-widest">
              Stage 1 · Input Layer
            </div>
            <h2 className="text-3xl font-bold text-gray-900 mb-3">What you upload — and what happens to it</h2>
            <p className="text-gray-500 mb-10 max-w-2xl">
              Artistic accepts two types of files. Everything flows through a common input handler (<code className="bg-gray-100 px-1.5 py-0.5 rounded text-xs">src/pipeline/input_handler.py</code>) that routes each file type to the right processing path.
            </p>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-10">
              {/* Audio card */}
              <div className="border border-gray-200 rounded-2xl overflow-hidden">
                <div className="bg-sky-600 text-white px-6 py-4">
                  <div className="text-2xl mb-2">
                    <IconMic className="w-7 h-7" />
                  </div>
                  <h3 className="text-lg font-bold">Audio Files</h3>
                  <p className="text-sm text-sky-100">WAV · MP3 · FLAC</p>
                </div>
                <div className="p-6 space-y-4">
                  <div>
                    <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2">Step 1 — Transcription</p>
                    <p className="text-sm text-gray-700">The audio is passed to <strong>OpenAI Whisper</strong> (<code className="bg-gray-100 px-1 rounded text-xs">src/audio/transcriber.py</code>), which transcribes the speech into text with word-level timestamps for each segment.</p>
                  </div>
                  <div>
                    <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2">Step 3 — Child audio isolation</p>
                    <p className="text-sm text-gray-700"><code className="bg-gray-100 px-1 rounded text-xs">ChildAudioExtractor</code> uses timing data to segment out just the child&apos;s speech segments before acoustic analysis, ensuring acoustic features reflect only the child&apos;s voice.</p>
                  </div>
                  <div>
                    <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2">Step 4 — Dual feature paths</p>
                    <p className="text-sm text-gray-700">The <strong>audio signal</strong> goes to acoustic feature extraction. The <strong>transcript</strong> goes to pragmatic and syntactic extraction — both happen in parallel.</p>
                  </div>
                </div>
              </div>

              {/* CHAT card */}
              <div className="border border-gray-200 rounded-2xl overflow-hidden">
                <div className="bg-emerald-700 text-white px-6 py-4">
                  <div className="text-2xl mb-2">
                    <IconTranscript className="w-7 h-7" />
                  </div>
                  <h3 className="text-lg font-bold">CHAT Transcripts</h3>
                  <p className="text-sm text-emerald-100">.cha format from CLAN / ASDBank</p>
                </div>
                <div className="p-6 space-y-4">
                  <div>
                    <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2">What is CHAT format?</p>
                    <p className="text-sm text-gray-700">CHAT (Codes for the Human Analysis of Transcripts) is the standard transcription format used by researchers worldwide. It encodes speaker turns, timing, pauses, disfluencies, and morphology in a structured way that computers can read.</p>
                  </div>
                  <div>
                    <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2">What the parser extracts</p>
                    <p className="text-sm text-gray-700"><code className="bg-gray-100 px-1 rounded text-xs">CHATParser</code> (<code className="bg-gray-100 px-1 rounded text-xs">src/parsers/chat_parser.py</code>) reads: speaker codes (CHI, MOT, INV), utterance text, <code className="bg-gray-100 px-1 rounded text-xs">%mor</code> morphology tier, <code className="bg-gray-100 px-1 rounded text-xs">%snd</code> timing, pause markers <code className="bg-gray-100 px-1 rounded text-xs">(.) (..) (...)</code>, and retrace markers <code className="bg-gray-100 px-1 rounded text-xs">[/] [//] [///]</code>.</p>
                  </div>
                  <div>
                    <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2">Dataset — ASDBank</p>
                    <p className="text-sm text-gray-700">Training data comes from <strong>6 ASDBank corpora</strong>: Eigsti, Flusberg, Nadig, Quigley-McNalley, Rollins, and AAC — all real clinical recordings of children with ASD and typically developing (TD) controls.</p>
                  </div>
                </div>
              </div>
            </div>

            {/* CHAT example */}
            <div className="bg-gray-950 rounded-2xl p-6 overflow-x-auto">
              <p className="text-xs text-gray-400 mb-4 font-semibold uppercase tracking-wide">Example CHAT file snippet</p>
              <pre className="text-sm text-green-300 leading-relaxed font-mono">{`@Begin
@Participants: CHI Target_Child Child, MOT Mother Adult
@ID: eng|asdbank|CHI|4;2.||ASD|Child||
*MOT: what are you playing with today ?
%snd: "session.wav" 1234 2456
*CHI: um (...) the [/] the blue car .
%mor: co|um pause:long det|the det|the adj|blue n|car .
*MOT: the blue car, nice ! can you tell me more ?
*CHI: &-uh [//] it goes vroom [/] vroom .
%mor: co|uh v|go-3S adv|vroom n|vroom .
@End`}</pre>
              <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-3">
                {[
                  { code: '*CHI:', desc: 'Child speaker tier' },
                  { code: '%mor:', desc: 'Morphology annotation' },
                  { code: '(...)', desc: 'Long pause (~1.5s)' },
                  { code: '[/]', desc: 'Retrace / false start' },
                ].map(item => (
                  <div key={item.code} className="bg-gray-800 rounded-lg px-3 py-2">
                    <code className="text-green-400 text-xs block mb-0.5">{item.code}</code>
                    <p className="text-xs text-gray-400">{item.desc}</p>
                  </div>
                ))}
              </div>
            </div>
          </section>

          {/* ── SECTION: FEATURE EXTRACTION ── */}
          <section id="extraction" ref={setRef('extraction')}>
            <div className="inline-flex items-center gap-2 bg-emerald-600 text-white text-xs font-semibold px-3 py-1 rounded-full mb-6 uppercase tracking-widest">
              Stage 2 · Feature Extraction
            </div>
            <h2 className="text-3xl font-bold text-gray-900 mb-3">Turning speech into numbers</h2>
            <p className="text-gray-500 mb-10 max-w-2xl">
              All extractors inherit from a common <code className="bg-gray-100 px-1.5 py-0.5 rounded text-xs">BaseFeatureExtractor</code> and return a <code className="bg-gray-100 px-1.5 py-0.5 rounded text-xs">FeatureResult</code> object. Three parallel extraction components each analyse a different dimension of the speech signal.
            </p>

            <div className="space-y-6">
            {/* Component 1 — Acoustic */}
              <div className="border-2 border-sky-300 bg-sky-50/40 rounded-2xl p-6">
                <div className="flex items-start justify-between flex-wrap gap-4">
                  <div>
                    <div className="flex items-center gap-3 mb-2">
                      <div className="w-8 h-8 bg-sky-600 text-white rounded-lg flex items-center justify-center font-bold text-sm">1</div>
                      <h3 className="text-xl font-bold text-gray-900">Acoustic / Prosodic Component</h3>
                      <span className="text-xs bg-sky-100 text-sky-700 border border-sky-200 px-2 py-0.5 rounded-full font-semibold">153 features</span>
                      <span className="text-xs bg-gray-900 text-white px-2 py-0.5 rounded-full font-semibold">fully documented below ↓</span>
                    </div>
                    <p className="text-sm text-gray-600 max-w-2xl">Analyses <em>how</em> speech sounds — pitch, rhythm, voice quality, spectral characteristics, and timing. All features are extracted from the isolated child audio segments using the <strong>librosa</strong> library. Features are computed as <strong>global statistical summaries</strong> (mean, standard deviation, min, max) over the entire recording — not time-segmented.</p>
                  </div>
                  <span className="text-xs text-gray-400 font-mono">src/features/acoustic_prosodic/</span>
                </div>
                <div className="mt-5 grid grid-cols-2 md:grid-cols-4 gap-3">
                  {ACOUSTIC_CATEGORIES.map(c => (
                    <div key={c.id} className={`bg-white rounded-xl border ${c.borderColor} p-3 flex items-center gap-3`}>
                      <span className={`inline-flex items-center justify-center w-7 h-7 rounded-lg ${c.bgColor} ${c.color} border ${c.borderColor} flex-shrink-0`}>
                        <AcousticIcon id={c.id} className="w-4 h-4" />
                      </span>
                      <div>
                        <div className={`text-sm font-semibold ${c.color}`}>{c.label}</div>
                        <div className="text-xs font-mono text-gray-500 mt-0.5">{c.count} features</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
{/* Component 3 — Pragmatic */}
<div className="border-2 border-emerald-300 bg-emerald-50/40 rounded-2xl p-6">
                <div className="flex items-start justify-between flex-wrap gap-4">
                  <div>
                    <div className="flex items-center gap-3 mb-2">
                      <div className="w-8 h-8 bg-emerald-600 text-white rounded-lg flex items-center justify-center font-bold text-sm">3</div>
                      <h3 className="text-xl font-bold text-gray-900">Pragmatic / Conversational Component</h3>
                      <span className="text-xs bg-emerald-100 text-emerald-700 border border-emerald-200 px-2 py-0.5 rounded-full font-semibold">207 features</span>
                      <span className="text-xs bg-gray-900 text-white px-2 py-0.5 rounded-full font-semibold">fully documented below ↓</span>
                    </div>
                    <p className="text-sm text-gray-600 max-w-2xl">The most comprehensive component — analyses <em>how</em> the child uses language in conversation. Covers turn-taking rhythm, topic coherence, pause patterns, repair strategies, vocabulary, and more. Trained on the ASDBank corpus.</p>
                  </div>
                  <span className="text-xs text-gray-400 font-mono">src/features/pragmatic_conversational/</span>
                </div>
                <div className="mt-5 grid grid-cols-2 md:grid-cols-3 gap-3">
                  {PRAGMATIC_CATEGORIES.map(c => (
                    <div key={c.id} className={`bg-white rounded-xl border ${c.borderColor} p-3 flex items-start justify-between`}>
                      <div className="flex items-center gap-3">
                        <span className={`inline-flex items-center justify-center w-7 h-7 rounded-lg ${c.bgColor} ${c.color} border ${c.borderColor}`}>
                          <PragmaticIcon id={c.id} className="w-4 h-4" />
                        </span>
                        <div>
                          <div className={`text-sm font-semibold ${c.color}`}>{c.label}</div>
                          <div className="text-xs font-mono text-gray-500 mt-0.5">{c.count} features</div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Component 2 — Syntactic */}
              <div className="border-2 border-dashed border-violet-200 bg-violet-50/40 rounded-2xl p-6">
                <div className="flex items-start justify-between flex-wrap gap-4">
                  <div>
                    <div className="flex items-center gap-3 mb-2">
                      <div className="w-8 h-8 bg-violet-600 text-white rounded-lg flex items-center justify-center font-bold text-sm">2</div>
                      <h3 className="text-xl font-bold text-gray-900">Syntactic / Semantic Component</h3>
                      <span className="text-xs bg-violet-100 text-violet-700 border border-violet-200 px-2 py-0.5 rounded-full font-semibold">26 features</span>
                    </div>
                    <p className="text-sm text-gray-600 max-w-2xl">Analyses sentence grammar and word meaning — how complex the sentence structures are, how grammatically accurate the speech is, and how semantically coherent the language is. Uses <strong>spaCy</strong> for parsing and <strong>NLTK WordNet</strong> for semantic analysis.</p>
                  </div>
                  <span className="text-xs text-gray-400 font-mono">src/features/syntactic_semantic/</span>
                </div>
                <div className="mt-5 grid grid-cols-2 md:grid-cols-3 gap-3">
                  {[
                    { group: 'Syntactic Complexity', count: 6, detail: 'dependency tree depth, clause complexity' },
                    { group: 'Grammatical Accuracy', count: 5, detail: 'tense consistency, structure diversity' },
                    { group: 'Sentence Structure', count: 4, detail: 'parse tree height, NP/VP complexity' },
                    { group: 'Semantic Features', count: 4, detail: 'spaCy embedding similarity' },
                    { group: 'Vocabulary Semantic', count: 4, detail: 'WordNet hypernym depth, synset diversity' },
                    { group: 'Advanced Semantic', count: 3, detail: 'entity density, verb argument structure' },
                  ].map(g => (
                    <div key={g.group} className="bg-white rounded-xl border border-violet-200 p-3">
                      <div className="text-sm font-semibold text-gray-800">{g.group}</div>
                      <div className="text-xs text-violet-700 font-mono mt-0.5">{g.count} features</div>
                      <div className="text-xs text-gray-400 mt-0.5">{g.detail}</div>
                    </div>
                  ))}
                </div>

              </div>


            </div>
          </section>
{/* ── ACOUSTIC DETAIL (NEW) ── */}
          <section id="acoustic" ref={setRef('acoustic')}>
            <div className="inline-flex items-center gap-2 bg-sky-600 text-white text-xs font-semibold px-3 py-1 rounded-full mb-6 uppercase tracking-widest">
              Acoustic Component — Full Detail
            </div>
            <h2 className="text-3xl font-bold text-gray-900 mb-3">153 acoustic features, explained</h2>
            <p className="text-gray-500 mb-4 max-w-2xl">
              The acoustic component is organised into 8 sub-modules, each targeting a distinct aspect of the speech signal. All features are computed as <strong>global statistical summaries</strong> (mean, standard deviation, min, max, etc.) over the entire child audio recording — not time-segmented windows. Click <strong>View All Features</strong> on any category to open a full searchable table.
            </p>

            {/* Key design callout */}
            <div className="bg-sky-50 border border-sky-200 rounded-2xl p-5 mb-10 flex items-start gap-4">
              <div className="w-9 h-9 bg-sky-100 rounded-xl flex items-center justify-center flex-shrink-0">
                <IconWaveform className="w-5 h-5 text-sky-700" />
              </div>
              <div>
                <p className="text-sm font-semibold text-sky-900 mb-1">Design principle: global statistical summaries</p>
                <p className="text-sm text-sky-800">
                  Unlike the pragmatic component which analyses conversation turn-by-turn, the acoustic component extracts features across the <strong>entire speech signal</strong>. For example, &quot;pitch variability&quot; is the standard deviation of F0 across all voiced frames — not an analysis of individual words or sentences. This approach is robust to varying recording lengths and avoids overfitting to specific conversation segments.
                </p>
              </div>
            </div>

            {/* Acoustic preprocessing callout */}
            <div className="bg-gray-900 text-white rounded-2xl p-6 mb-10">
              <p className="text-xs text-gray-400 mb-4 font-semibold uppercase tracking-wide">Acoustic Preprocessing Pipeline</p>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-sm">
                {[
                  { step: '01', title: 'Load & Resample', desc: 'librosa.load() at 22050Hz; mono conversion if stereo', code: 'sr=22050, mono=True' },
                  { step: '02', title: 'Pre-emphasis', desc: 'High-frequency boost filter to compensate for spectral roll-off of speech', code: 'y = np.append(y[0], y[1:] - 0.97*y[:-1])' },
                  { step: '03', title: 'Child isolation', desc: 'Segment audio to child speaker regions only using Whisper timing', code: 'ChildAudioExtractor' },
                  { step: '04', title: 'Feature extraction', desc: 'Per-module extraction with global statistical aggregation', code: 'mean, std, min, max' },
                ].map(s => (
                  <div key={s.step} className="bg-gray-800 rounded-xl p-4">
                    <div className="text-xs font-mono text-gray-400 mb-1">{s.step}</div>
                    <div className="font-semibold text-white mb-1">{s.title}</div>
                    <div className="text-xs text-gray-400 mb-2">{s.desc}</div>
                    <code className="text-xs text-green-400 bg-gray-900 px-2 py-1 rounded block">{s.code}</code>
                  </div>
                ))}
              </div>
            </div>

            <div className="space-y-8">
              {ACOUSTIC_CATEGORIES.map((cat, idx) => (
                <div key={cat.id} className={`border ${cat.borderColor} rounded-2xl overflow-hidden`}>
                  <div className={`${cat.bgColor} px-6 py-5 flex items-start justify-between flex-wrap gap-4`}>
                    <div className="flex items-start gap-4">
                      <div className={`w-10 h-10 bg-white rounded-xl border ${cat.borderColor} flex items-center justify-center`}>
                        <AcousticIcon id={cat.id} className={`w-6 h-6 ${cat.color}`} />
                      </div>
                      <div>
                        <div className="flex items-center gap-3 flex-wrap">
                          <h3 className={`text-lg font-bold ${cat.color}`}>{cat.label}</h3>
                          <span className={`text-xs font-semibold px-2.5 py-0.5 rounded-full bg-white border ${cat.borderColor} ${cat.color}`}>
                            {cat.count} features
                          </span>
                          <span className="text-xs text-gray-400 font-mono">Sub-module {idx + 1} of 8</span>
                        </div>
                        <p className="text-sm text-gray-600 mt-1 max-w-2xl">{cat.summary}</p>
                      </div>
                    </div>
                    <button
                      onClick={() => setOpenCategory(cat)}
                      className={`flex-shrink-0 px-5 py-2.5 rounded-xl text-sm font-semibold transition-all border ${cat.borderColor} bg-white ${cat.color} hover:bg-gray-900 hover:text-white hover:border-gray-900`}
                    >
                      View All {cat.count} Features →
                    </button>
                  </div>

                  <div className="bg-white px-6 py-5">
                    <div className="mb-4">
                      <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2">Extraction Method</p>
                      <p className="text-sm text-gray-600">{cat.method}</p>
                    </div>
                    <div>
                      <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-3">Sample Features</p>
                      <div className="flex flex-wrap gap-2">
                        {cat.features.slice(0, 6).map(f => (
                          <span key={f.name} className={`text-xs font-mono px-2.5 py-1 rounded-lg ${cat.bgColor} ${cat.color} border ${cat.borderColor}`}>
                            {f.name}
                          </span>
                        ))}
                        {cat.features.length > 6 && (
                          <button onClick={() => setOpenCategory(cat)} className="text-xs px-2.5 py-1 rounded-lg text-gray-400 bg-gray-100 hover:bg-gray-200 transition-colors">
                            +{cat.features.length - 6} more
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* ASD relevance summary box */}
            <div className="mt-10 bg-gray-50 border border-gray-200 rounded-2xl p-6">
              <h3 className="font-bold text-gray-900 mb-4">Acoustic ASD Markers — Summary of Evidence</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                {[
                  { marker: 'Reduced pitch variability', evidence: 'One of the most replicated findings — monotone speech reflecting limited prosodic use for social/emotional communication', strength: 'Strong' },
                  { marker: 'Narrowed vowel space (F1/F2)', evidence: 'Systematic reduction in vowel space area documented across multiple ASD studies, indicating less precise articulation', strength: 'Strong' },
                  { marker: 'Elevated jitter & shimmer', evidence: 'Voice perturbation measures differ between ASD and TD groups, potentially reflecting laryngeal motor differences', strength: 'Moderate' },
                  { marker: 'Atypical speech rate', evidence: 'Both unusually slow and unusually fast speech documented in ASD subgroups; rate variability is diagnostically relevant', strength: 'Moderate' },
                  { marker: 'Unusual spectral characteristics', evidence: 'MFCC profiles and spectral centroid differences found in multiple automatic speech analysis studies', strength: 'Moderate' },
                  { marker: 'Reduced dynamic range', evidence: 'Flat loudness profile consistent with reduced prosodic marking of emphasis and contrast', strength: 'Moderate' },
                ].map(m => (
                  <div key={m.marker} className="bg-white rounded-xl border border-gray-200 p-4">
                    <div className="flex items-start justify-between mb-2">
                      <span className="font-semibold text-gray-900 text-sm">{m.marker}</span>
                      <span className={`text-xs px-2 py-0.5 rounded-full font-medium flex-shrink-0 ml-2 ${m.strength === 'Strong' ? 'bg-emerald-100 text-emerald-700' : 'bg-amber-100 text-amber-700'}`}>
                        {m.strength}
                      </span>
                    </div>
                    <p className="text-xs text-gray-500 leading-relaxed">{m.evidence}</p>
                  </div>
                ))}
              </div>
            </div>
          </section>
          {/* ── SECTION: PRAGMATIC DETAIL ── */}
          <section id="pragmatic" ref={setRef('pragmatic')}>
            <div className="inline-flex items-center gap-2 bg-emerald-600 text-white text-xs font-semibold px-3 py-1 rounded-full mb-6 uppercase tracking-widest">
              Pragmatic Component — Full Detail
            </div>
            <h2 className="text-3xl font-bold text-gray-900 mb-3">207 pragmatic features, explained</h2>
            <p className="text-gray-500 mb-4 max-w-2xl">
              The pragmatic component is organised into 6 sub-modules, each targeting a distinct aspect of conversational behaviour. Click <strong>View All Features</strong> on any category to open a full searchable table with plain-English descriptions, extraction methods, and ASD relevance for every feature.
            </p>
            <p className="text-sm text-gray-400 mb-10">
              All features are parsed from CLAN CHAT format transcripts (<code className="bg-gray-100 px-1 rounded text-xs">.cha</code> files) using the custom <code className="bg-gray-100 px-1 rounded text-xs">CHATParser</code>.
            </p>

            <div className="space-y-8">
              {PRAGMATIC_CATEGORIES.map((cat, idx) => (
                <div key={cat.id} className={`border ${cat.borderColor} rounded-2xl overflow-hidden`}>
                  <div className={`${cat.bgColor} px-6 py-5 flex items-start justify-between flex-wrap gap-4`}>
                    <div className="flex items-start gap-4">
                      <div className={`w-10 h-10 bg-white rounded-xl border ${cat.borderColor} flex items-center justify-center`}>
                        <PragmaticIcon id={cat.id} className={`${cat.color.replace('text-', 'fill-').replace('fill-current', '')} w-6 h-6`} />
                      </div>
                      <div>
                        <div className="flex items-center gap-3 flex-wrap">
                          <h3 className={`text-lg font-bold ${cat.color}`}>{cat.label}</h3>
                          <span className={`text-xs font-semibold px-2.5 py-0.5 rounded-full bg-white border ${cat.borderColor} ${cat.color}`}>
                            {cat.count} features
                          </span>
                          <span className="text-xs text-gray-400 font-mono">Sub-module {idx + 1} of 6</span>
                        </div>
                        <p className="text-sm text-gray-600 mt-1 max-w-2xl">{cat.summary}</p>
                      </div>
                    </div>
                    <button
                      onClick={() => setOpenCategory(cat)}
                      className={`flex-shrink-0 px-5 py-2.5 rounded-xl text-sm font-semibold transition-all border ${cat.borderColor} bg-white ${cat.color} hover:bg-gray-900 hover:text-white hover:border-gray-900`}
                    >
                      View All {cat.count} Features →
                    </button>
                  </div>

                  <div className="bg-white px-6 py-5">
                    <div className="mb-4">
                      <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2">Extraction Method</p>
                      <p className="text-sm text-gray-600">{cat.method}</p>
                    </div>

                    {/* Feature preview chips */}
                    <div>
                      <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-3">Sample Features</p>
                      <div className="flex flex-wrap gap-2">
                        {cat.features.slice(0, 8).map(f => (
                          <span key={f.name} className={`text-xs font-mono px-2.5 py-1 rounded-lg ${cat.bgColor} ${cat.color} border ${cat.borderColor}`}>
                            {f.name}
                          </span>
                        ))}
                        {cat.features.length > 8 && (
                          <button
                            onClick={() => setOpenCategory(cat)}
                            className="text-xs px-2.5 py-1 rounded-lg text-gray-400 bg-gray-100 hover:bg-gray-200 transition-colors"
                          >
                            +{cat.features.length - 8} more
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* ── SECTION: TRAINING ── */}
          <section id="training" ref={setRef('training')}>
            <div className="inline-flex items-center gap-2 bg-sky-600 text-white text-xs font-semibold px-3 py-1 rounded-full mb-6 uppercase tracking-widest">
              Stage 3 · Model Training
            </div>
            <h2 className="text-3xl font-bold text-gray-900 mb-3">How the models learn</h2>
            <p className="text-gray-500 mb-10 max-w-2xl">
              Each of the 3 feature components has its own ML model trained independently. A unified orchestrator (<code className="bg-gray-100 px-1.5 py-0.5 rounded text-xs">src/models/model_trainer.py</code>) coordinates all three.
            </p>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-10">
              {/* Dataset */}
              <div className="bg-gray-50 border border-gray-200 rounded-2xl p-6">
                <h3 className="font-bold text-gray-900 mb-3 flex items-center gap-2">
                  <span className="w-6 h-6 bg-gray-900 text-white rounded-md flex items-center justify-center text-xs">
                    <IconChart className="w-4 h-4" />
                  </span>
                  Training Data — ASDBank
                </h3>
                <p className="text-sm text-gray-600 mb-4">All models are trained on real clinical recordings from the <strong>ASDBank</strong> collection, hosted on TalkBank. Six corpora of child–adult conversations:</p>
                <div className="space-y-2">
                  {['Eigsti', 'Flusberg', 'Nadig', 'Quigley-McNalley', 'Rollins', 'AAC'].map(c => (
                    <div key={c} className="flex items-center gap-3 text-sm">
                      <div className="w-2 h-2 rounded-full bg-emerald-500 flex-shrink-0" />
                      <span className="text-gray-700">ASDBank — {c} corpus</span>
                    </div>
                  ))}
                </div>
                <p className="text-xs text-gray-400 mt-4">Each recording is labelled as <strong>ASD</strong> (autism spectrum disorder) or <strong>TD</strong> (typically developing) based on clinical diagnosis metadata in the CHAT header.</p>
              </div>

              {/* Preprocessing */}
              <div className="bg-gray-50 border border-gray-200 rounded-2xl p-6">
                <h3 className="font-bold text-gray-900 mb-3 flex items-center gap-2">
                  <span className="w-6 h-6 bg-gray-900 text-white rounded-md flex items-center justify-center text-xs">⚙</span>
                  Preprocessing Pipeline
                </h3>
                <div className="space-y-3 text-sm text-gray-600">
                  <div className="bg-white rounded-lg border border-gray-200 p-3">
                    <p className="font-semibold text-gray-800 text-xs uppercase tracking-wide mb-1">1 — Imputation</p>
                    <p>Missing feature values filled with the median of the training set. Ensures no NaN values reach the model.</p>
                  </div>
                  <div className="bg-white rounded-lg border border-gray-200 p-3">
                    <p className="font-semibold text-gray-800 text-xs uppercase tracking-wide mb-1">2 — Scaling</p>
                    <p><code className="bg-gray-100 px-1 rounded text-xs">StandardScaler</code> normalises all features to zero mean and unit variance, preventing large-valued features from dominating.</p>
                  </div>
                  <div className="bg-white rounded-lg border border-gray-200 p-3">
                    <p className="font-semibold text-gray-800 text-xs uppercase tracking-wide mb-1">3 — Class balancing (SMOTE)</p>
                    <p>Synthetic Minority Over-sampling Technique (SMOTE) generates synthetic ASD examples to balance the dataset — because clinical datasets often have more TD than ASD samples.</p>
                  </div>
                  <div className="bg-white rounded-lg border border-gray-200 p-3">
                    <p className="font-semibold text-gray-800 text-xs uppercase tracking-wide mb-1">4 — Feature selection</p>
                    <p><code className="bg-gray-100 px-1 rounded text-xs">SelectKBest</code> or mutual information selects the top 30 most informative features per component, reducing noise and overfitting risk.</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Algorithms */}
            <div className="border border-gray-200 rounded-2xl overflow-hidden mb-8">
              <div className="bg-gray-50 px-6 py-4 border-b border-gray-200">
                <h3 className="font-bold text-gray-900">Supported ML Algorithms</h3>
                <p className="text-sm text-gray-500 mt-0.5">Any of these can be selected for each component via the Training UI or API</p>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 divide-x divide-y divide-gray-100">
                {[
                  { name: 'Random Forest', type: 'Ensemble', desc: 'Many decision trees voting together', color: 'text-emerald-700' },
                  { name: 'XGBoost', type: 'Gradient Boosting', desc: 'Iteratively corrects previous tree errors', color: 'text-sky-700' },
                  { name: 'LightGBM', type: 'Gradient Boosting', desc: 'Faster tree boosting for large feature sets', color: 'text-sky-700' },
                  { name: 'SVM', type: 'Support Vector', desc: 'Finds optimal boundary between classes', color: 'text-violet-700' },
                  { name: 'Logistic Regression', type: 'Linear', desc: 'Probabilistic linear classification', color: 'text-violet-700' },
                  { name: 'MLP', type: 'Neural Network', desc: 'Multi-layer perceptron neural network', color: 'text-rose-700' },
                  { name: 'Gradient Boosting', type: 'Ensemble', desc: 'Sklearn gradient boosting classifier', color: 'text-amber-700' },
                  { name: 'AdaBoost', type: 'Ensemble', desc: 'Adaptive boosting on weak learners', color: 'text-amber-700' },
                ].map(a => (
                  <div key={a.name} className="px-5 py-4">
                    <div className={`text-sm font-bold ${a.color}`}>{a.name}</div>
                    <div className="text-xs text-gray-400 mt-0.5">{a.type}</div>
                    <div className="text-xs text-gray-500 mt-1">{a.desc}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Maximum Simplicity Mode */}
            <div className="bg-amber-50 border border-amber-200 rounded-2xl p-6">
              <div className="flex items-start gap-4">
                <div className="w-10 h-10 bg-amber-100 rounded-xl flex items-center justify-center text-xl flex-shrink-0">
                  <IconWarning className="w-6 h-6" />
                </div>
                <div>
                  <h3 className="font-bold text-gray-900 mb-2">Design Choice: Maximum Simplicity Mode</h3>
                  <p className="text-sm text-gray-700 mb-3">
                    Because ASDBank contains a relatively small number of recordings, the models are intentionally <strong>hyper-regularised</strong> to avoid overfitting. The default training configuration targets <strong>75–80% accuracy</strong> on the validation set — deliberately conservative to ensure the model generalises to new children.
                  </p>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-xs font-mono">
                    {[
                      { alg: 'Random Forest', params: 'n_estimators=1, max_depth=2, max_features=1' },
                      { alg: 'XGBoost', params: 'n_estimators=1, max_depth=1, lr=0.0001, subsample=0.1' },
                      { alg: 'SVM / LR', params: 'C=0.000001 (extreme regularisation)' },
                      { alg: 'MLP', params: 'hidden=(2,), alpha=10.0 (heavy L2)' },
                      { alg: 'LightGBM', params: 'n_estimators=1, reg_alpha=100, reg_lambda=200' },
                    ].map(p => (
                      <div key={p.alg} className="bg-white rounded-lg border border-amber-200 px-3 py-2">
                        <div className="font-semibold text-amber-800 text-xs mb-0.5">{p.alg}</div>
                        <div className="text-gray-500 break-all text-xs">{p.params}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* ── SECTION: FUSION ── */}
          <section id="fusion" ref={setRef('fusion')}>
            <div className="inline-flex items-center gap-2 bg-violet-600 text-white text-xs font-semibold px-3 py-1 rounded-full mb-6 uppercase tracking-widest">
              Stage 4 · Prediction &amp; Fusion
            </div>
            <h2 className="text-3xl font-bold text-gray-900 mb-3">Combining three signals into one prediction</h2>
            <p className="text-gray-500 mb-10 max-w-2xl">
              Each of the 3 component models produces an independent probability score. The <code className="bg-gray-100 px-1.5 py-0.5 rounded text-xs">ModelFusion</code> class (<code className="bg-gray-100 px-1.5 py-0.5 rounded text-xs">src/pipeline/model_fusion.py</code>) combines them into a single final prediction.
            </p>

            {/* Default fusion */}
            <div className="border border-violet-200 bg-violet-50/30 rounded-2xl p-6 mb-8">
              <h3 className="font-bold text-gray-900 mb-4">Default Method: Weighted Averaging</h3>
              <p className="text-sm text-gray-600 mb-6">Each component&apos;s probability is multiplied by its weight, and the results are summed. The pragmatic component carries the most weight because it has the most features, the best-validated extraction methodology, and the highest discriminative power on ASDBank data.</p>

              <div className="flex flex-col md:flex-row items-center gap-4">
                {[
                  { name: 'Pragmatic / Conversational', weight: '50%', bar: 'w-1/2', color: 'bg-emerald-500', textColor: 'text-emerald-700' },
                  { name: 'Acoustic / Prosodic', weight: '25%', bar: 'w-1/4', color: 'bg-sky-500', textColor: 'text-sky-700' },
                  { name: 'Syntactic / Semantic', weight: '25%', bar: 'w-1/4', color: 'bg-violet-500', textColor: 'text-violet-700' },
                ].map(c => (
                  <div key={c.name} className="flex-1 w-full">
                    <div className={`text-sm font-semibold ${c.textColor} mb-1`}>{c.name}</div>
                    <div className="bg-gray-200 rounded-full h-4 w-full overflow-hidden">
                      <div className={`${c.color} h-full ${c.bar} rounded-full`} />
                    </div>
                    <div className="text-lg font-bold text-gray-900 mt-1">{c.weight}</div>
                  </div>
                ))}
              </div>

              <div className="mt-6 bg-white border border-violet-200 rounded-xl p-4">
                <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2">Formula</p>
                <code className="text-sm text-gray-700">
                  P(ASD) = 0.50 × P_pragmatic + 0.25 × P_acoustic + 0.25 × P_syntactic
                </code>
              </div>
            </div>

            {/* Other fusion methods */}
            <div className="border border-gray-200 rounded-2xl overflow-hidden">
              <div className="bg-gray-50 px-6 py-4 border-b border-gray-200">
                <h3 className="font-bold text-gray-900">All Fusion Strategies Available</h3>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-5 divide-y md:divide-y-0 md:divide-x divide-gray-100">
                {[
                  { name: 'Voting', desc: 'Each model votes ASD/TD; majority wins' },
                  { name: 'Averaging', desc: 'Equal-weight mean of all 3 probabilities' },
                  { name: 'Weighted', desc: 'Weighted sum (default: 0.5 / 0.25 / 0.25)', highlight: true },
                  { name: 'Max Confidence', desc: 'Uses whichever model is most confident' },
                  { name: 'Stacking', desc: 'Logistic Regression meta-learner on component scores' },
                ].map(m => (
                  <div key={m.name} className={`px-5 py-4 ${m.highlight ? 'bg-violet-50' : ''}`}>
                    <div className={`text-sm font-bold ${m.highlight ? 'text-violet-800' : 'text-gray-800'}`}>{m.name} {m.highlight && '★'}</div>
                    <div className="text-xs text-gray-500 mt-1">{m.desc}</div>
                  </div>
                ))}
              </div>
            </div>
          </section>

          {/* ── SECTION: INTERPRETABILITY ── */}
          <section id="interpretability" ref={setRef('interpretability')}>
            <div className="inline-flex items-center gap-2 bg-rose-600 text-white text-xs font-semibold px-3 py-1 rounded-full mb-6 uppercase tracking-widest">
              Stage 5 · Interpretability
            </div>
            <h2 className="text-3xl font-bold text-gray-900 mb-3">Understanding the prediction</h2>
            <p className="text-gray-500 mb-10 max-w-2xl">
              A prediction without explanation is not useful for clinicians. Artistic generates two types of explanation for every prediction — one showing <em>why</em> this prediction was made, and one showing <em>what would need to change</em> to get a different result.
            </p>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-10">
              {/* SHAP */}
              <div className="border border-rose-200 rounded-2xl overflow-hidden">
                <div className="bg-rose-600 text-white px-6 py-5">
                  <div className="text-2xl mb-2">
                    <IconChart className="w-7 h-7" />
                  </div>
                  <h3 className="text-xl font-bold">SHAP Explanations</h3>
                  <p className="text-sm text-rose-100 mt-1">SHapley Additive exPlanations</p>
                </div>
                <div className="bg-white p-6 space-y-4">
                  <div>
                    <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2">What is SHAP?</p>
                    <p className="text-sm text-gray-700">SHAP assigns each feature a score that shows how much it <em>pushed the prediction towards ASD or away from ASD</em> for this specific child. It&apos;s based on game theory — treating each feature as a &quot;player&quot; and calculating their fair contribution to the outcome.</p>
                  </div>
                  <div>
                    <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2">Waterfall Plot (local)</p>
                    <p className="text-sm text-gray-700">Shows how each feature pushed the prediction up or down from the baseline for a single prediction. Red features pushed towards ASD, blue features pushed away. Helps clinicians understand exactly why this child got this result.</p>
                  </div>
                  <div>
                    <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2">Global SHAP (population)</p>
                    <p className="text-sm text-gray-700">Bar and beeswarm plots show which features matter most across the whole training population — these are available in the Feature Guide page.</p>
                  </div>
                  <div className="bg-gray-50 rounded-lg px-4 py-3 text-xs text-gray-500">
                    <strong>Implementation:</strong> <code className="bg-gray-200 px-1 rounded">src/interpretability/explainability/shap_manager.py</code> — uses TreeExplainer for tree models, LinearExplainer for linear models.
                  </div>
                </div>
              </div>

              {/* Counterfactuals */}
              <div className="border border-orange-200 rounded-2xl overflow-hidden">
                <div className="bg-orange-600 text-white px-6 py-5">
                  <div className="text-2xl mb-2">
                    <IconCounterfactual className="w-7 h-7" />
                  </div>
                  <h3 className="text-xl font-bold">Counterfactual Analysis</h3>
                  <p className="text-sm text-orange-100 mt-1">What-If Explanations</p>
                </div>
                <div className="bg-white p-6 space-y-4">
                  <div>
                    <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2">What are counterfactuals?</p>
                    <p className="text-sm text-gray-700">Counterfactuals answer the question: &quot;<em>What is the minimum change to this child&apos;s features that would flip the prediction from ASD to TD (or vice versa)?</em>&quot; This helps clinicians identify which specific behaviours, if they changed, would most meaningfully affect the outcome.</p>
                  </div>
                  <div>
                    <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2">How it works — VAE-based generation</p>
                    <p className="text-sm text-gray-700">A <strong>Variational Autoencoder (VAE)</strong> learns the realistic space of feature values from the training data. The counterfactual generator then searches this space for the closest realistic point to the current features that would produce the opposite prediction.</p>
                  </div>
                  <div>
                    <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2">Why this matters clinically</p>
                    <p className="text-sm text-gray-700">Instead of just saying &quot;this looks like ASD&quot;, Artistic can say &quot;if the child&apos;s average response latency decreased from 3.2s to 0.8s, the prediction would change&quot; — directly pointing to an actionable intervention target.</p>
                  </div>
                  <div className="bg-gray-50 rounded-lg px-4 py-3 text-xs text-gray-500">
                    <strong>Implementation:</strong> <code className="bg-gray-200 px-1 rounded">src/interpretability/counterfactuals/</code> — PyTorch VAE + L2-distance minimisation.
                  </div>
                </div>
              </div>
            </div>

            {/* Annotated transcript */}
            <div className="border border-gray-200 rounded-2xl p-6">
              <h3 className="font-bold text-gray-900 mb-2 flex items-center gap-3">
                <span className="inline-flex items-center justify-center w-7 h-7 rounded-md bg-gray-900 text-white">
                  <IconNote className="w-4 h-4" />
                </span>
                Annotated Transcript
              </h3>
              <p className="text-sm text-gray-600 mb-4">
                Every prediction also produces an <strong>annotated transcript</strong> — a colour-coded HTML view of the conversation that highlights the specific turns, pauses, latencies, topic shifts, and repairs that the model detected. This is generated by <code className="bg-gray-100 px-1 rounded text-xs">TranscriptAnnotator</code> in <code className="bg-gray-100 px-1 rounded text-xs">src/pipeline/annotated_transcript.py</code>.
              </p>
              <div className="flex flex-wrap gap-3">
                {[
                  { label: '[TURN]', color: 'bg-blue-100 text-blue-700 border-blue-200' },
                  { label: '[PAUSE]', color: 'bg-yellow-100 text-yellow-700 border-yellow-200' },
                  { label: '[LATENCY]', color: 'bg-orange-100 text-orange-700 border-orange-200' },
                  { label: '[REPAIR]', color: 'bg-red-100 text-red-700 border-red-200' },
                  { label: '[TOPIC]', color: 'bg-purple-100 text-purple-700 border-purple-200' },
                  { label: '[MARKER]', color: 'bg-green-100 text-green-700 border-green-200' },
                ].map(tag => (
                  <span key={tag.label} className={`text-xs font-mono font-semibold px-3 py-1 rounded-full border ${tag.color}`}>{tag.label}</span>
                ))}
              </div>
            </div>
          </section>

          {/* ── FOOTER PADDING ── */}
          <div className="h-16" />

        </main>
      </div>

      {/* ── MODAL ── */}
      {openCategory && (
        <FeatureTableModal category={openCategory} onClose={() => setOpenCategory(null)} />
      )}
    </div>
  );
}
