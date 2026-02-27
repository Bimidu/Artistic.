'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';

export default function Home() {
  const router = useRouter();

  useEffect(() => {
    const runInit = () => {
      window.__artisticInitHomeUi?.();
    };

    // If the script already ran in this session, just re-bind to the new DOM.
    if (typeof window.__artisticInitHomeUi === 'function') {
      runInit();
      return;
    }

    // Otherwise load it once, then initialize.
    const existing = document.getElementById('artistic-app-js') as HTMLScriptElement | null;
    if (existing) {
      existing.addEventListener('load', runInit, { once: true });
      runInit();
      return;
    }

    const script = document.createElement('script');
    script.id = 'artistic-app-js';
    script.src = '/app.js';
    script.async = true;
    script.addEventListener('load', runInit, { once: true });
    document.body.appendChild(script);
  }, []);

  return (
    <>
      {/* Header */}
      <header className="bg-lime-950">
        <div className="max-w-7xl mx-auto px-12 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-6">
              <div className="text-4xl text-white">Artistic</div>
              <div className="text-lg text-primary-400 hidden sm:block">ASD Detection System</div>
            </div>

            <div className="flex items-center gap-8">
              <div className="toggle-switch" id="modeToggle">
                <div className="toggle-option active" data-mode="user">User Mode</div>
                <div className="toggle-option" data-mode="training">Training Mode</div>
                <div className="toggle-slider" id="toggleSlider"></div>
              </div>

              <div className="flex items-center gap-3 text-base text-primary-600">
                <span className="w-2.5 h-2.5 rounded-full bg-red-400" id="statusDot"></span>
                <span id="statusText">Disconnected</span>
              </div>
              <div>
                <button
                  onClick={() => router.push('/guideline')}
                  className="px-5 py-2 bg-white text-lime-900 rounded-xl hover:bg-lime-100"
                >
                  Feature Guidelines
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* API Configuration Bar */}
        <div className="bg-white hidden" id="apiConfigBar">
          <div className="max-w-7xl mx-auto px-12 py-2">
            <div className="flex items-center gap-6 justify-between">
              <div className="flex items-center gap-4 flex-1">
                <label className="text-sm text-primary-600 whitespace-nowrap">API URL</label>
                <input type="text" className="flex-1 px-4 py-2 bg-primary-50 rounded-full text-sm focus:outline-none focus:bg-primary-100 transition-all" id="apiUrl" defaultValue="http://localhost:8000" />
              </div>
              <button className="px-6 py-2 bg-primary-900 text-white rounded-full text-sm hover:bg-primary-800 transition-all" onClick={() => window.testConnection?.()}>Test Connection</button>
            </div>
          </div>
        </div>
      </header>

      {/* Landing Section */}
      <div id="landingSection" className="bg-gradient-to-b from-lime-900/20 to-white">
        <div className="max-w-7xl mx-auto px-12 py-20">
          {/* Hero */}
          <div className="text-center mb-20">
            <h1 className="text-6xl font-normal text-lime-950 mb-6">
              ASD Detection Through Speech Analysis
            </h1>
            <p className="text-xl text-lime-800 max-w-3xl mx-auto leading-relaxed">
              Advanced machine learning system for analyzing speech patterns to support
              autism spectrum disorder detection using multi-modal feature extraction for children
            </p>
          </div>

          {/* Components Grid */}
          <div className="mb-16">
            <h2 className="text-4xl font-light text-lime-950 text-center mb-12">
              Four-Component Analysis Framework
            </h2>
            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
              {/* Component 1: Pragmatic & Conversational */}
              <div className="bg-white rounded-2xl p-8 transition-shadow">
                <div className="w-16 h-16 bg-lime-100 rounded-2xl flex items-center justify-center mb-6">
                  <svg className="w-10 h-10 text-lime-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                  </svg>
                </div>
                <h3 className="text-xl font-medium text-lime-950 mb-3">Pragmatic &amp; Conversational</h3>
                <p className="text-sm text-lime-700 leading-relaxed">
                  Analyzes turn-taking, topic maintenance, conversational repairs, and social communication patterns in dialogue
                </p>
              </div>
              {/* Component 2: Acoustic & Prosodic */}
              <div className="bg-white rounded-2xl p-8 transition-shadow">
                <div className="w-16 h-16 bg-lime-100 rounded-2xl flex items-center justify-center mb-6">
                  <svg className="w-10 h-10 text-lime-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
                  </svg>
                </div>
                <h3 className="text-xl font-medium text-lime-950 mb-3">Acoustic &amp; Prosodic</h3>
                <p className="text-sm text-lime-700 leading-relaxed">
                  Examines pitch variation, speech rhythm, intonation patterns, and vocal quality characteristics
                </p>
              </div>
              {/* Component 3: Syntactic & Semantic */}
              <div className="bg-white rounded-2xl p-8 transition-shadow">
                <div className="w-16 h-16 bg-lime-100 rounded-2xl flex items-center justify-center mb-6">
                  <svg className="w-10 h-10 text-lime-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                  </svg>
                </div>
                <h3 className="text-xl font-medium text-lime-950 mb-3">Syntactic &amp; Semantic</h3>
                <p className="text-sm text-lime-700 leading-relaxed">
                  Evaluates sentence structure complexity, grammatical patterns, and semantic coherence in language use
                </p>
              </div>
              {/* Component 4: Multi-Modal Fusion */}
              <div className="bg-white rounded-2xl p-8 transition-shadow">
                <div className="w-16 h-16 bg-lime-100 rounded-2xl flex items-center justify-center mb-6">
                  <svg className="w-10 h-10 text-lime-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M4 5a1 1 0 011-1h4a1 1 0 011 1v7a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM14 5a1 1 0 011-1h4a1 1 0 011 1v7a1 1 0 01-1 1h-4a1 1 0 01-1-1V5zM4 16a1 1 0 011-1h4a1 1 0 011 1v3a1 1 0 01-1 1H5a1 1 0 01-1-1v-3zM14 16a1 1 0 011-1h4a1 1 0 011 1v3a1 1 0 01-1 1h-4a1 1 0 01-1-1v-3z" />
                  </svg>
                </div>
                <h3 className="text-xl font-medium text-lime-950 mb-3">Multi-Modal Fusion</h3>
                <p className="text-sm text-lime-700 leading-relaxed">
                  Integrates insights from all components using ensemble learning for comprehensive analysis
                </p>
              </div>
            </div>
          </div>

          {/* Key Features */}
          <div className="grid md:grid-cols-3 gap-6 max-w-5xl mx-auto">
            <div className="text-center">
              <div className="w-12 h-12 bg-lime-100 rounded-xl flex items-center justify-center mx-auto mb-4">
                <svg className="w-7 h-7 text-lime-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <h4 className="text-lg font-medium text-lime-950 mb-2">Real-time Analysis</h4>
              <p className="text-sm text-lime-700">Instant processing of audio files and text transcripts</p>
            </div>
            <div className="text-center">
              <div className="w-12 h-12 bg-lime-100 rounded-xl flex items-center justify-center mx-auto mb-4">
                <svg className="w-7 h-7 text-lime-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                </svg>
              </div>
              <h4 className="text-lg font-medium text-lime-950 mb-2">Explainable AI</h4>
              <p className="text-sm text-lime-700">SHAP values and counterfactual explanations</p>
            </div>
            <div className="text-center">
              <div className="w-12 h-12 bg-lime-100 rounded-xl flex items-center justify-center mx-auto mb-4">
                <svg className="w-7 h-7 text-lime-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
                </svg>
              </div>
              <h4 className="text-lg font-medium text-lime-950 mb-2">Flexible Training</h4>
              <p className="text-sm text-lime-700">Custom model training with multiple ML algorithms</p>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-12 py-12">
        {/* User Mode */}
        <div className="mode-content" id="userMode">
          <div className="grid lg:grid-cols-2 gap-12">
            {/* Input Section */}
            <div className="bg-primary-50 border-t-2 border-black rounded-b-3xl overflow-hidden">
              <div className="px-10 py-8 bg-white">
                <h2 className="text-5xl text-primary-900">Analyze Speech</h2>
              </div>
              <div className="bg-primary-100">
                <div className="flex px-10 gap-4">
                  <button className="tab px-8 py-5 text-lg border-b-2 border-primary-900 text-primary-900" data-input="audio">Audio Upload</button>
                  <button className="tab px-8 py-5 text-lg border-b-2 border-transparent text-primary-500 hover:text-primary-900 transition-colors" data-input="file">CHAT File</button>
                </div>
              </div>
              <div className="p-10">
                {/* Audio Upload */}
                <div className="input-panel" id="audioPanel">
                  <div className="upload-area bg-white rounded-3xl p-16 text-center cursor-pointer" id="audioUploadArea">
                    <svg className="mx-auto h-20 w-20 text-primary-400" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                      <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                    <p className="mt-6 text-2xl text-primary-900">Drop audio file here</p>
                    <p className="mt-3 text-base text-primary-500">Supports WAV, MP3, FLAC, OGG</p>
                  </div>
                  <input type="file" className="hidden" id="audioFileInput" accept=".wav,.mp3,.flac" />
                  <div id="selectedAudioFile" className="mt-4 text-base text-accent-600"></div>

                  {/* In-browser Recording Controls */}
                  <div id="audioRecordSection" className="mt-8 bg-white rounded-2xl p-6 border border-primary-200">
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <h3 className="text-lg font-medium text-primary-900">Record Audio</h3>
                      </div>
                      <div className="flex items-center gap-3">
                        <button
                          id="audioRecordButton"
                          type="button"
                          className="px-4 py-2 rounded-full bg-primary-900 text-white text-sm hover:bg-primary-800 transition-colors"
                        >
                          Start recording
                        </button>
                        <button
                          id="audioStopButton"
                          type="button"
                          className="px-4 py-2 rounded-full bg-red-100 text-red-700 text-sm hover:bg-red-200 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                          disabled
                        >
                          Stop &amp; analyze
                        </button>
                      </div>
                    </div>
                    <div className="flex items-center justify-between text-sm text-primary-600">
                      <div id="audioRecordStatus" className="flex items-center gap-2">
                        <span className="w-2.5 h-2.5 rounded-full bg-gray-300" id="audioRecordIndicator"></span>
                        <span id="audioRecordStatusText">Microphone idle</span>
                      </div>
                      <div id="audioRecordTimer" className="font-mono text-primary-700 hidden">00:00</div>
                    </div>
                    <p id="audioRecordError" className="mt-3 text-sm text-red-600 hidden"></p>
                  </div>

                  <button className="mt-8 w-full px-8 py-5 bg-primary-900 text-white rounded-2xl text-xl hover:bg-primary-800 transition-all disabled:opacity-40 disabled:cursor-not-allowed" onClick={() => window.predictFromAudio?.()} id="predictAudioBtn" disabled>
                    Analyze Audio
                  </button>
                </div>

                {/* CHAT File Upload */}
                <div className="input-panel hidden" id="filePanel">
                  <div className="upload-area bg-white rounded-3xl p-16 text-center cursor-pointer" id="chaUploadArea">
                    <svg className="mx-auto h-20 w-20 text-primary-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    <p className="mt-6 text-2xl text-primary-900">Drop CHAT file here</p>
                    <p className="mt-3 text-base text-primary-500">Supports .cha files</p>
                  </div>
                  <input type="file" className="hidden" id="chaFileInput" accept=".cha,.CHA,text/plain" />
                  <div id="selectedChaFile" className="mt-4 text-base text-accent-600"></div>
                  <button className="mt-8 w-full px-8 py-5 bg-primary-900 text-white rounded-2xl text-xl hover:bg-primary-800 transition-all disabled:opacity-40 disabled:cursor-not-allowed" onClick={() => window.predictFromChatFile?.()} id="predictChaBtn" disabled>
                    Analyze File
                  </button>
                </div>
              </div>
            </div>

            {/* Results Section */}
            <div className="bg-primary-50 border-t-2 border-black rounded-b-3xl overflow-hidden">
              <div className="px-10 py-8 bg-white">
                <h2 className="text-5xl text-primary-900">Analysis Results</h2>
              </div>
              <div className="p-10" id="resultsArea">
                <div className="text-center py-24 text-primary-400 text-xl">
                  Upload an audio file or CHAT transcript to see results
                </div>
              </div>
            </div>
          </div>

          {/* Waveform Display */}
          <div id="waveformSectionResults" className="hidden mt-12 bg-primary-50 rounded-3xl overflow-hidden">
            <div className="px-10 py-6 bg-white">
              <h3 className="text-2xl text-primary-900 mb-1">Child Speech Waveform</h3>
              <p className="text-sm text-primary-500">Visual representation of the child&apos;s speech audio</p>
            </div>
            <div className="px-10 pb-10 bg-primary-50">
              <div className="bg-white rounded-2xl p-6 border border-primary-200">
                <div className="mb-3 p-3 bg-lime-50 border-l-4 border-lime-600 rounded">
                  <p className="text-sm text-lime-900">
                    <strong>Note:</strong> Acoustic and prosodic features are extracted as global statistical summaries from the entire speech signal, not from specific time segments.
                  </p>
                </div>
                <canvas id="waveformCanvasResults" className="w-full" style={{ height: '150px' }}></canvas>
                <div className="mt-3 px-2">
                  <div className="flex flex-wrap items-center gap-4 text-xs">
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-3 bg-blue-500 rounded-sm"></div>
                      <span className="text-primary-700">Raw speech signal (amplitude over time)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-3 bg-gray-300 rounded-sm"></div>
                      <span className="text-primary-700">Relative speech energy (loudness distribution)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-3 bg-orange-400 rounded-sm"></div>
                      <span className="text-primary-700">Detected speech-silence boundaries</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-3 bg-teal-500 rounded-sm"></div>
                      <span className="text-primary-700">Detected speech activity timeline (used for pause and rhythm analysis)</span>
                    </div>
                  </div>
                  <p className="mt-2 text-xs text-primary-600 italic">
                    These visual elements represent signal-level properties of the audio.
                    Acoustic and prosodic features are computed as global statistical summaries.
                  </p>
                </div>
                <div className="mt-4 p-4 bg-primary-50 rounded-lg border border-primary-200">
                  <p className="text-sm font-medium text-primary-800 mb-2">Observed Speech Characteristics:</p>
                  <div id="waveformSummaryResults" className="text-sm text-primary-700 leading-relaxed">
                    <p>Analyzing speech characteristics...</p>
                  </div>
                </div>
                <div className="mt-4 flex items-center justify-between text-sm text-primary-600">
                  <span id="waveformInfoResults">Loading waveform...</span>
                  <audio id="waveformAudioResults" controls className="max-w-xs"></audio>
                </div>
              </div>
            </div>
          </div>

          {/* Annotated Transcript */}
          <div className="mt-12 border-t-2 border-black bg-primary-50 rounded-b-s3xl overflow-hidden hidden" id="annotationCard">
            <div className="px-10 py-8 bg-white flex items-center justify-between">
              <h2 className="text-5xl text-primary-900">Annotated Pragmatic Transcript</h2>
              <div className="flex items-center gap-3">
                <span id="annotationCount" className="px-6 py-3 bg-accent-100 text-accent-700 text-base rounded-full">Features Marked</span>
                <button id="toggleTranscriptView" className="px-4 py-2 bg-primary-100 text-primary-700 rounded-lg hover:bg-primary-200 transition-colors text-sm">
                  <span id="viewToggleText">Compact View</span>
                </button>
              </div>
            </div>
            <div className="p-10">
              {/* Feature Summary Panel */}
              <div id="featureSummaryPanel" className="mb-6 bg-white rounded-2xl p-6 border border-primary-200">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-xl font-medium text-primary-900">Detected Features</h3>
                  <button id="toggleFeatureSummary" className="text-primary-600 hover:text-primary-900 transition-colors text-sm">
                    <span id="summaryToggleText">Hide</span>
                  </button>
                </div>
                <div id="featureSummaryContent" className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-3"></div>
              </div>

              {/* Controls Panel */}
              <div className="mb-6 bg-white rounded-2xl p-6 border border-primary-200">
                <div className="flex flex-wrap items-center gap-4">
                  <div className="flex-1 min-w-[200px]">
                    <input type="text" id="transcriptSearch" placeholder="Search transcript..." className="w-full px-4 py-2 bg-primary-50 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:bg-white transition-all" />
                  </div>
                  <div className="flex items-center gap-2">
                    <label className="text-sm text-primary-700">Filter by feature:</label>
                    <select id="featureFilter" className="px-4 py-2 bg-primary-50 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:bg-white transition-all">
                      <option value="all">All Features</option>
                    </select>
                  </div>
                  <button id="clearFilters" className="px-4 py-2 bg-primary-100 text-primary-700 rounded-lg hover:bg-primary-200 transition-colors text-sm">Clear Filters</button>
                  <div className="flex items-center gap-2">
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input type="checkbox" id="semanticCoherenceToggle" className="w-5 h-5 text-primary-600 rounded" />
                      <span className="text-sm text-primary-700">Semantic Coherence</span>
                    </label>
                  </div>
                </div>
              </div>

              {/* Transcript Display */}
              <div id="annotatedTranscript" className="bg-white rounded-2xl p-8 border border-primary-200 max-h-[600px] overflow-y-auto transcript-container"></div>

              {/* Statistics Panel */}
              <div id="transcriptStats" className="mt-6 bg-white rounded-2xl p-6 border border-primary-200 hidden">
                <h3 className="text-lg font-medium text-primary-900 mb-4">Transcript Statistics</h3>
                <div id="statsContent" className="grid grid-cols-2 md:grid-cols-4 gap-4"></div>
              </div>
            </div>
          </div>

          {/* Local SHAP Explanation */}
          <div id="localShapSection" className="mt-12 hidden">
            <h3 className="text-3xl font-medium text-primary-900 mb-4">Why this prediction was made</h3>
            <p className="text-sm text-primary-600 mb-6">
              This waterfall plot explains how each conversational feature contributed to the final ASD / TD prediction for this specific transcript.
            </p>
            <div className="bg-white rounded-2xl p-6 border border-primary-200">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img id="localShapWaterfall" className="w-full rounded-xl border border-primary-100" alt="Local SHAP Waterfall Explanation" />
            </div>
          </div>

          {/* Counterfactual Explanation */}
          <div id="counterfactualSection" className="mt-12 hidden">
            <h3 className="text-3xl font-medium text-primary-900 mb-4">What would change this prediction?</h3>
            <p className="text-sm text-primary-600 mb-6">
              This analysis shows the smallest realistic changes required to flip the model&apos;s prediction to the opposite class.
            </p>
            <div id="whatIfBox" className="bg-primary-50 border border-primary-200 rounded-2xl p-6 mb-6 text-primary-900"></div>
            <div className="grid md:grid-cols-3 gap-4 mb-6">
              <div className="bg-white rounded-2xl p-5 border border-primary-200">
                <p className="text-sm text-primary-600">Prediction flipped</p>
                <p id="cfFlipped" className="text-xl font-bold"></p>
              </div>
              <div className="bg-white rounded-2xl p-5 border border-primary-200">
                <p className="text-sm text-primary-600">Overall change (L2)</p>
                <p id="cfL2" className="text-xl font-bold"></p>
              </div>
              <div className="bg-white rounded-2xl p-5 border border-primary-200">
                <p className="text-sm text-primary-600">Features changed</p>
                <p id="cfTotal" className="text-xl font-bold"></p>
              </div>
            </div>
            <div className="bg-white rounded-2xl p-6 border border-primary-200">
              <h4 className="text-xl font-medium text-primary-900 mb-4">Most influential feature changes</h4>
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b text-left">
                    <th className="py-2">Feature</th>
                    <th className="py-2">Original</th>
                    <th className="py-2">Counterfactual</th>
                    <th className="py-2">Change</th>
                  </tr>
                </thead>
                <tbody id="cfTableBody"></tbody>
              </table>
            </div>
          </div>

          {/* Interactive Counterfactual Chat */}
          <div id="cfChatSection" className="mt-10 bg-primary-50 border border-primary-200 rounded-2xl p-6 hidden">
            <h3 className="text-2xl font-medium text-primary-900 mb-3">Explore a What-If Scenario</h3>
            <p className="text-sm text-primary-600 mb-5">
              Ask a hypothetical question about a conversational behavior to explore how it might influence the model&apos;s decision.
              <span className="italic"> (Simulated response – future extension)</span>
            </p>
            <div className="flex flex-col md:flex-row gap-4 mb-4">
              <select id="cfQuestion" className="flex-1 px-4 py-2 rounded-xl border border-primary-300 bg-white text-sm focus:outline-none">
                <option>What if the number of &quot;uhm&quot; continuation markers increased by 0.3?</option>
                <option>What if average turn length decreased slightly?</option>
                <option>What if semantic coherence improved?</option>
              </select>
              <button onClick={() => window.simulateCounterfactualChat?.()} className="px-6 py-2 rounded-xl bg-primary-600 text-white text-sm hover:bg-primary-700 transition">Ask</button>
            </div>
            <div id="cfChatResponse" className="hidden bg-white border border-primary-200 rounded-xl p-5 text-sm text-primary-900"></div>
          </div>
        </div>

        {/* Training Mode */}
        <div className="mode-content hidden" id="trainingMode">
          {/* Training mode tabs row */}
          <div className="bg-primary-100 border-b-2 border-primary-200">
            <div className="max-w-7xl mx-auto">
              <div className="flex gap-4">
                <button type="button" className="training-tab px-8 py-5 text-lg border-b-2 border-primary-900 text-primary-900" data-training-tab="feature-extraction">Feature Extraction</button>
                <button type="button" className="training-tab px-8 py-5 text-lg border-b-2 border-transparent text-primary-500 hover:text-primary-900 transition-colors" data-training-tab="training">Training</button>
                <button type="button" className="training-tab px-8 py-5 text-lg border-b-2 border-transparent text-primary-500 hover:text-primary-900 transition-colors" data-training-tab="trained-models">Trained Models</button>
              </div>
            </div>
          </div>

          {/* Feature Extraction Section */}
          <div className="training-tab-panel py-12" id="trainingTabFeatureExtraction" data-training-tab="feature-extraction">
            <div className="mb-12 bg-primary-50 border-t-2 border-black rounded-b-3xl overflow-hidden">
              <div className="px-10 py-8 bg-white flex items-center justify-between">
                <h2 className="text-5xl text-primary-900">Feature Extraction</h2>
                <button className="px-6 py-3 bg-primary-900 text-white rounded-2xl text-base hover:bg-primary-800 transition-all" onClick={() => window.loadDatasets?.()}>Refresh</button>
              </div>
              <div className="p-10">
                <p className="text-lg text-primary-600 mb-6">
                  Select datasets from your file system to extract features. Extracted features will be saved to CSV files.
                </p>
                <div className="grid lg:grid-cols-2 gap-8">
                  <div className="bg-white rounded-2xl p-6" style={{ maxHeight: '500px', overflowY: 'auto' }}>
                    <h3 className="text-2xl text-primary-900 mb-4">Select Datasets to Extract</h3>
                    <div id="extractionDatasetList">
                      <div className="text-center py-16 text-primary-400 text-xl">Click Refresh to load datasets</div>
                    </div>
                  </div>
                  <div className="bg-white rounded-2xl p-6">
                    <h3 className="text-2xl text-primary-900 mb-4">Extraction Settings</h3>
                    <div className="mb-6">
                      <label className="block text-lg text-primary-900 mb-3">Component</label>
                      <select className="w-full px-6 py-4 bg-primary-50 rounded-2xl text-base focus:outline-none focus:bg-primary-100 transition-all" id="extractionComponent">
                        <option value="pragmatic_conversational">Pragmatic &amp; Conversational</option>
                        <option value="acoustic_prosodic">Acoustic &amp; Prosodic</option>
                        <option value="syntactic_semantic">Syntactic &amp; Semantic</option>
                      </select>
                      <p className="text-sm text-primary-500 mt-2">Select which component&apos;s features to extract</p>
                    </div>
                    <div className="mb-6">
                      <label className="block text-lg text-primary-900 mb-3">Max Samples per Dataset</label>
                      <input type="number" id="maxSamplesExtraction" min="1" className="w-full px-6 py-4 bg-primary-50 rounded-2xl text-base focus:outline-none focus:bg-primary-100 transition-all" placeholder="Leave empty for all samples" />
                      <p className="text-sm text-primary-500 mt-2">Limit samples for large datasets (e.g., TD). Leave empty to extract all.</p>
                    </div>
                    <button className="w-full px-8 py-5 bg-primary-900 text-white rounded-2xl text-xl hover:bg-primary-800 transition-all" onClick={() => window.extractFeatures?.()}>
                      Extract Features
                    </button>
                    <div className="mt-6 bg-primary-50 rounded-xl p-4 hidden" id="extractionStatus">
                      <div id="extractionStatusContent"></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Training Section */}
          <div className="training-tab-panel hidden py-12" id="trainingTabTraining" data-training-tab="training">
            <div className="mb-12 bg-primary-50 border-t-2 border-black rounded-b-3xl overflow-hidden">
              <div className="px-10 py-8 bg-white flex items-center justify-between">
                <h2 className="text-5xl text-primary-900">Training</h2>
                <button className="px-6 py-3 bg-primary-900 text-white rounded-2xl text-base hover:bg-primary-800 transition-all" onClick={() => window.loadAvailableDatasetsForTraining?.()}>Refresh</button>
              </div>
              <div className="p-10">
                <div className="grid lg:grid-cols-2 gap-8 items-stretch">
                  {/* Dataset Selection for Training */}
                  <div className="bg-white rounded-2xl p-6 flex flex-col">
                    <h3 className="text-2xl text-primary-900 mb-4 flex-shrink-0">Available Datasets (from CSV)</h3>
                    <div id="datasetList" className="flex-1 overflow-y-auto min-h-0">
                      <div className="text-center py-16 text-primary-400 text-xl">Click Refresh to load datasets</div>
                    </div>
                  </div>
                  {/* Training Controls */}
                  <div className="bg-white rounded-2xl p-6">
                    <h3 className="text-2xl text-primary-900 mb-4">Training Controls</h3>
                    <div className="mb-8">
                      <label className="block text-xl text-primary-900 mb-4">Component</label>
                      <select className="w-full px-6 py-4 bg-white rounded-2xl text-base focus:outline-none focus:bg-primary-100 transition-all" id="trainingComponent">
                        <option value="pragmatic_conversational">Pragmatic &amp; Conversational</option>
                        <option value="acoustic_prosodic">Acoustic &amp; Prosodic</option>
                        <option value="syntactic_semantic">Syntactic &amp; Semantic (Dummy Features)</option>
                      </select>
                      <p className="text-sm text-primary-500 mt-2">Note: Acoustic &amp; Syntactic use placeholder features for testing</p>
                    </div>
                    <div className="mb-8">
                      <label className="block text-xl text-primary-900 mb-4">Feature Selection</label>
                      <div className="space-y-4">
                        <label className="flex items-center cursor-pointer p-5 bg-white rounded-2xl hover:bg-primary-100 transition-colors">
                          <input type="checkbox" id="featureSelectionEnabled" defaultChecked className="w-5 h-5 text-primary-600 rounded" />
                          <span className="ml-4 text-lg text-primary-900">Enable feature selection</span>
                        </label>
                        <div id="featureCountSection">
                          <label className="block text-base text-primary-700 mb-2">Number of features to select</label>
                          <input type="number" id="nFeatures" defaultValue="30" min="1" max="218" className="w-full px-6 py-4 bg-white rounded-2xl text-base focus:outline-none focus:bg-primary-100 transition-all" />
                          <p className="text-sm text-primary-500 mt-2">Default: 30 (max: 218 for pragmatic, 20 for others)</p>
                        </div>
                      </div>
                    </div>
                    <div className="mb-8">
                      <label className="block text-xl text-primary-900 mb-4">Counterfactual Explanations</label>
                      <div className="space-y-4">
                        <label className="flex items-center cursor-pointer p-5 bg-white rounded-2xl hover:bg-primary-100 transition-colors">
                          <input type="checkbox" id="enableAutoencoder" className="w-5 h-5 text-primary-600 rounded" />
                          <div className="ml-4 flex-1">
                            <span className="text-lg text-primary-900">Enable counterfactual autoencoder</span>
                            <p className="text-sm text-primary-500 mt-1">Train autoencoder for counterfactual explanations (may crash on macOS - disabled by default)</p>
                          </div>
                        </label>
                      </div>
                    </div>
                    <div className="mb-8">
                      <label className="block text-xl text-primary-900 mb-4">Model Types</label>
                      <p className="text-sm text-primary-500 mb-3">Available models change based on selected component</p>
                      <div id="modelTypesContainer" className="grid grid-cols-2 gap-3">
                        <label className="flex items-center cursor-pointer p-4 bg-white rounded-2xl hover:bg-primary-100 transition-colors">
                          <input type="checkbox" value="svm" defaultChecked className="w-5 h-5 text-primary-600 rounded" />
                          <span className="ml-3 text-base text-primary-900">SVM</span>
                        </label>
                        <label className="flex items-center cursor-pointer p-4 bg-white rounded-2xl hover:bg-primary-100 transition-colors">
                          <input type="checkbox" value="logistic" defaultChecked className="w-5 h-5 text-primary-600 rounded" />
                          <span className="ml-3 text-base text-primary-900">Logistic Regression</span>
                        </label>
                      </div>
                    </div>
                    <div className="mb-8">
                      <label className="block text-xl text-primary-900 mb-4">Training Parameters</label>
                      <div className="space-y-4">
                        <div className="bg-white rounded-2xl p-4">
                          <label className="block text-sm text-primary-700 mb-2">Test Set Size (%)</label>
                          <input type="number" id="testSize" defaultValue="20" min="10" max="40" step="5" className="w-full px-4 py-3 bg-primary-50 rounded-xl text-base focus:outline-none focus:bg-primary-100 transition-all" />
                          <p className="text-xs text-primary-500 mt-1">Percentage of data reserved for testing</p>
                        </div>
                        <div className="bg-white rounded-2xl p-4">
                          <label className="block text-sm text-primary-700 mb-2">Random Seed</label>
                          <input type="number" id="randomState" defaultValue="42" min="0" max="999" className="w-full px-4 py-3 bg-primary-50 rounded-xl text-base focus:outline-none focus:bg-primary-100 transition-all" />
                          <p className="text-xs text-primary-500 mt-1">For reproducible results</p>
                        </div>
                      </div>
                    </div>
                    {/* Advanced Options */}
                    <div className="mb-8">
                      <button onClick={() => window.toggleHyperparameters?.()} className="w-full flex items-center justify-between p-4 bg-white rounded-2xl hover:bg-primary-100 transition-colors">
                        <span className="text-xl text-primary-900">Advanced Hyperparameters</span>
                        <svg id="hyperparamChevron" className="w-6 h-6 text-primary-600 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
                        </svg>
                      </button>
                      <div id="hyperparamSection" className="hidden mt-4 space-y-4">
                        <div className="bg-white rounded-2xl p-4">
                          <p className="text-sm text-primary-600 mb-4">Customize hyperparameters for each selected model type. Leave default for recommended values.</p>
                          <div id="hyperparamControls" className="space-y-6"></div>
                        </div>
                      </div>
                    </div>
                    <div>
                      <button className="w-full px-8 py-5 bg-primary-900 text-white rounded-2xl text-xl hover:bg-primary-800 transition-all" onClick={() => window.startTraining?.()}>Start Training</button>
                    </div>
                    <div className="mt-8 bg-primary-50 rounded-xl p-4 hidden" id="trainingStatus">
                      <div id="trainingStatusContent"></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Trained Models Section */}
          <div className="training-tab-panel hidden py-12" id="trainingTabTrainedModels" data-training-tab="trained-models">
            <div className="mt-12 bg-primary-50 border-t-2 border-black rounded-b-3xl overflow-hidden">
              <div className="px-10 py-8 bg-white flex items-center justify-between">
                <h2 className="text-5xl text-primary-900">Trained Models</h2>
                <button className="px-6 py-3 bg-primary-900 text-white rounded-2xl text-base hover:bg-primary-800 transition-all" onClick={() => window.loadAvailableModels?.()}>Refresh</button>
              </div>
              <div className="p-10">
                <div id="availableModelsContainer">
                  <div className="text-center py-16 text-primary-400 text-xl">Click Refresh to load trained models</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Model Details Modal */}
      <div id="modelDetailsModal" className="fixed inset-0 bg-black bg-opacity-50 hidden items-center justify-center z-50" onClick={(e) => window.closeModelDetails?.(e)}>
        <div className="bg-white rounded-3xl max-w-5xl w-full mx-4 max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
          <div className="sticky top-0 bg-white px-8 py-6 border-b border-primary-200 flex items-center justify-between">
            <h2 className="text-3xl font-medium text-primary-900">Model Performance Details</h2>
            <button onClick={() => window.closeModelDetails?.()} className="text-primary-600 hover:text-primary-900 transition-colors">
              <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          <div className="p-8" id="modalContent"></div>
        </div>
      </div>

    </>
  );
}
