'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';

export default function Home() {
  const router = useRouter();

  const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

  useEffect(() => {
    (window as unknown as { __ARTISTIC_API_URL?: string }).__ARTISTIC_API_URL = apiUrl;
  }, [apiUrl]);

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
      <header className="bg-primary-900 border-b border-primary-800">
        <div className="max-w-7xl mx-auto px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="text-xl font-medium text-white tracking-tight">Artistic</div>
              <div className="hidden sm:block h-4 w-px bg-primary-700"></div>
              <div className="hidden sm:block text-sm text-primary-500">Speech Analysis Platform</div>
            </div>

            <div className="flex items-center gap-6">
              <div className="toggle-switch" id="modeToggle">
                <div className="toggle-option active" data-mode="user">User Mode</div>
                <div className="toggle-option" data-mode="training">Training Mode</div>
                <div className="toggle-slider" id="toggleSlider"></div>
              </div>

              <div className="flex items-center gap-2 text-sm text-primary-500">
                <span className="w-2 h-2 rounded-full bg-red-400" id="statusDot"></span>
                <span id="statusText">Disconnected</span>
              </div>

              <button
                onClick={() => router.push('/guideline')}
                className="px-4 py-2 text-sm border border-primary-700 text-primary-400 rounded-lg hover:border-primary-500 hover:text-white transition-all"
              >
                Feature Guide
              </button>
            </div>
          </div>
        </div>

        {/* API Configuration Bar */}
        <div className="bg-white hidden" id="apiConfigBar">
          <div className="max-w-7xl mx-auto px-8 py-2">
            <div className="flex items-center gap-6 justify-between">
              <div className="flex items-center gap-4 flex-1">
                <label className="text-sm text-primary-600 whitespace-nowrap">API URL</label>
                <input type="text" className="flex-1 px-4 py-2 bg-primary-50 rounded-lg text-sm focus:outline-none focus:bg-primary-100 transition-all" id="apiUrl" defaultValue={process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'} />
              </div>
              <button className="px-5 py-2 bg-primary-900 text-white rounded-lg text-sm hover:bg-primary-800 transition-all" onClick={() => window.testConnection?.()}>Test Connection</button>
            </div>
          </div>
        </div>
      </header>

      {/* Landing Section */}
      <div id="landingSection" className="bg-white">

        {/* Hero */}
        <div className="max-w-7xl mx-auto px-8 pt-24 pb-20">
          <div className="max-w-3xl mx-auto text-center">
            <div className="inline-flex items-center gap-2 px-3 py-1.5 bg-primary-100 rounded-full text-xs font-medium text-primary-600 mb-8 tracking-wide uppercase">
              <span className="w-1.5 h-1.5 rounded-full bg-lime-600"></span>
              AI-powered speech screening
            </div>
            <h1 className="text-6xl font-normal text-primary-900 mb-6 leading-tight" style={{ letterSpacing: '-0.03em' }}>
              Smarter speech<br />screening starts here.
            </h1>
            <p className="text-lg text-primary-600 max-w-2xl mx-auto leading-relaxed mb-10">
              Artistic analyzes speech recordings and conversation transcripts to surface early markers of autism spectrum disorder — with transparent, explainable AI built for clinicians and researchers.
            </p>
            <div className="flex items-center justify-center gap-3 text-sm text-primary-500">
              <span className="flex items-center gap-1.5">
                <svg className="w-4 h-4 text-lime-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                </svg>
                Upload audio or CHAT files
              </span>
              <span className="text-primary-300">·</span>
              <span className="flex items-center gap-1.5">
                <svg className="w-4 h-4 text-lime-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                </svg>
                Get an AI-powered assessment
              </span>
              <span className="text-primary-300">·</span>
              <span className="flex items-center gap-1.5">
                <svg className="w-4 h-4 text-lime-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                </svg>
                Understand every prediction
              </span>
            </div>
          </div>
        </div>

        {/* Divider */}
        <div className="max-w-7xl mx-auto px-8">
          <div className="border-t border-primary-100"></div>
        </div>

        {/* How it works */}
        <div className="max-w-7xl mx-auto px-8 py-20">
          <div className="mb-12 text-center">
            <p className="text-xs font-medium text-primary-500 uppercase tracking-widest mb-2">How it works</p>
            <h2 className="text-3xl font-normal text-primary-900" style={{ letterSpacing: '-0.02em' }}>
              From upload to insight in three steps
            </h2>
          </div>

          <div className="grid md:grid-cols-3 gap-6 relative">
            {/* Connector line - desktop only */}
            <div className="hidden md:block absolute top-10 left-[calc(33.33%+1rem)] right-[calc(33.33%+1rem)] h-px bg-primary-200" style={{ zIndex: 0 }}></div>

            {/* Step 1 */}
            <div className="relative flex flex-col items-start">
              <div className="flex items-center gap-4 mb-5 w-full">
                <div className="w-10 h-10 rounded-full bg-primary-900 text-white flex items-center justify-center text-sm font-medium flex-shrink-0" style={{ zIndex: 1 }}>
                  01
                </div>
                <div className="flex-1 h-px bg-primary-200 md:hidden"></div>
              </div>
              <div className="bg-primary-50 border border-primary-200 rounded-2xl p-7 w-full hover:border-primary-300 transition-colors">
                <div className="w-9 h-9 bg-white border border-primary-200 rounded-xl flex items-center justify-center mb-4">
                  <svg className="w-4 h-4 text-primary-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                  </svg>
                </div>
                <h3 className="text-base font-medium text-primary-900 mb-2">Upload your file</h3>
                <p className="text-sm text-primary-600 leading-relaxed">
                  Drop an audio recording (WAV, MP3, FLAC) or a CHAT transcript file from your session directly into the tool — no conversion or formatting needed.
                </p>
              </div>
            </div>

            {/* Step 2 */}
            <div className="relative flex flex-col items-start">
              <div className="flex items-center gap-4 mb-5 w-full">
                <div className="w-10 h-10 rounded-full bg-primary-900 text-white flex items-center justify-center text-sm font-medium flex-shrink-0" style={{ zIndex: 1 }}>
                  02
                </div>
                <div className="flex-1 h-px bg-primary-200 md:hidden"></div>
              </div>
              <div className="bg-primary-50 border border-primary-200 rounded-2xl p-7 w-full hover:border-primary-300 transition-colors">
                <div className="w-9 h-9 bg-white border border-primary-200 rounded-xl flex items-center justify-center mb-4">
                  <svg className="w-4 h-4 text-primary-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                  </svg>
                </div>
                <h3 className="text-base font-medium text-primary-900 mb-2">Artistic runs the analysis</h3>
                <p className="text-sm text-primary-600 leading-relaxed">
                  Our AI examines conversation flow, voice patterns, and language structure — then combines all signals into a single, calibrated ASD screening result.
                </p>
              </div>
            </div>

            {/* Step 3 */}
            <div className="relative flex flex-col items-start">
              <div className="flex items-center gap-4 mb-5 w-full">
                <div className="w-10 h-10 rounded-full bg-primary-900 text-white flex items-center justify-center text-sm font-medium flex-shrink-0" style={{ zIndex: 1 }}>
                  03
                </div>
                <div className="flex-1 h-px bg-primary-200 md:hidden"></div>
              </div>
              <div className="bg-primary-50 border border-primary-200 rounded-2xl p-7 w-full hover:border-primary-300 transition-colors">
                <div className="w-9 h-9 bg-white border border-primary-200 rounded-xl flex items-center justify-center mb-4">
                  <svg className="w-4 h-4 text-primary-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                </div>
                <h3 className="text-base font-medium text-primary-900 mb-2">Review the full picture</h3>
                <p className="text-sm text-primary-600 leading-relaxed">
                  See an annotated transcript, a plain-language explanation of what drove the prediction, and explore what-if scenarios to understand the result in depth.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Divider */}
        <div className="max-w-7xl mx-auto px-8">
          <div className="border-t border-primary-100"></div>
        </div>

        {/* What Artistic analyzes */}
        <div className="max-w-7xl mx-auto px-8 py-16">
          <div className="mb-10">
            <p className="text-xs font-medium text-primary-500 uppercase tracking-widest mb-2">What we analyze</p>
            <h2 className="text-3xl font-normal text-primary-900" style={{ letterSpacing: '-0.02em' }}>
              Every dimension of your child&apos;s speech
            </h2>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-5">
            {/* Card 1 */}
            <div className="bg-white border border-primary-200 rounded-2xl p-7 hover:border-primary-400 hover:shadow-sm transition-all">
              <div className="w-10 h-10 bg-primary-100 rounded-xl flex items-center justify-center mb-5">
                <svg className="w-5 h-5 text-primary-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
              </div>
              <h3 className="text-base font-medium text-primary-900 mb-2">Conversation patterns</h3>
              <p className="text-sm text-primary-600 leading-relaxed">
                How your child takes turns, stays on topic, and responds in back-and-forth dialogue
              </p>
            </div>
            {/* Card 2 */}
            <div className="bg-white border border-primary-200 rounded-2xl p-7 hover:border-primary-400 hover:shadow-sm transition-all">
              <div className="w-10 h-10 bg-primary-100 rounded-xl flex items-center justify-center mb-5">
                <svg className="w-5 h-5 text-primary-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
                </svg>
              </div>
              <h3 className="text-base font-medium text-primary-900 mb-2">Voice &amp; rhythm</h3>
              <p className="text-sm text-primary-600 leading-relaxed">
                Pitch variation, speech pacing, and fluency — the vocal nuances that clinical screening values
              </p>
            </div>
            {/* Card 3 */}
            <div className="bg-white border border-primary-200 rounded-2xl p-7 hover:border-primary-400 hover:shadow-sm transition-all">
              <div className="w-10 h-10 bg-primary-100 rounded-xl flex items-center justify-center mb-5">
                <svg className="w-5 h-5 text-primary-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                </svg>
              </div>
              <h3 className="text-base font-medium text-primary-900 mb-2">Language structure</h3>
              <p className="text-sm text-primary-600 leading-relaxed">
                Sentence complexity, grammatical patterns, and coherence across the conversation
              </p>
            </div>
            {/* Card 4 */}
            <div className="bg-white border border-primary-200 rounded-2xl p-7 hover:border-primary-400 hover:shadow-sm transition-all">
              <div className="w-10 h-10 bg-primary-100 rounded-xl flex items-center justify-center mb-5">
                <svg className="w-5 h-5 text-primary-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M4 5a1 1 0 011-1h4a1 1 0 011 1v7a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM14 5a1 1 0 011-1h4a1 1 0 011 1v7a1 1 0 01-1 1h-4a1 1 0 01-1-1V5zM4 16a1 1 0 011-1h4a1 1 0 011 1v3a1 1 0 01-1 1H5a1 1 0 01-1-1v-3zM14 16a1 1 0 011-1h4a1 1 0 011 1v3a1 1 0 01-1 1h-4a1 1 0 01-1-1v-3z" />
                </svg>
              </div>
              <h3 className="text-base font-medium text-primary-900 mb-2">Unified assessment</h3>
              <p className="text-sm text-primary-600 leading-relaxed">
                All signals combined into a single, evidence-based prediction — with no black-box guesswork
              </p>
            </div>
          </div>
        </div>

        {/* Feature Highlights */}
        <div className="bg-primary-50 border-t border-primary-100">
          <div className="max-w-7xl mx-auto px-8 py-12">
            <div className="grid md:grid-cols-3 divide-y md:divide-y-0 md:divide-x divide-primary-200">
              <div className="py-6 md:py-0 md:px-12 first:pl-0 last:pr-0">
                <div className="flex items-start gap-4">
                  <div className="w-9 h-9 bg-white border border-primary-200 rounded-xl flex items-center justify-center flex-shrink-0">
                    <svg className="w-4 h-4 text-primary-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-primary-900 mb-1">Results in seconds</h4>
                    <p className="text-sm text-primary-600 leading-relaxed">Upload a file and get a full AI-powered analysis immediately — no waiting, no setup</p>
                  </div>
                </div>
              </div>
              <div className="py-6 md:py-0 md:px-12">
                <div className="flex items-start gap-4">
                  <div className="w-9 h-9 bg-white border border-primary-200 rounded-xl flex items-center justify-center flex-shrink-0">
                    <svg className="w-4 h-4 text-primary-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                    </svg>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-primary-900 mb-1">Transparent predictions</h4>
                    <p className="text-sm text-primary-600 leading-relaxed">Every result comes with SHAP explanations and counterfactual analysis — so you always know why</p>
                  </div>
                </div>
              </div>
              <div className="py-6 md:py-0 md:px-12">
                <div className="flex items-start gap-4">
                  <div className="w-9 h-9 bg-white border border-primary-200 rounded-xl flex items-center justify-center flex-shrink-0">
                    <svg className="w-4 h-4 text-primary-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
                    </svg>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-primary-900 mb-1">Built for research</h4>
                    <p className="text-sm text-primary-600 leading-relaxed">Train custom models on your own datasets with full control over features and algorithms</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-8 py-12">
        {/* User Mode */}
        <div className="mode-content" id="userMode">
          <div className="grid lg:grid-cols-2 gap-8">
            {/* Input Section */}
            <div className="bg-white border border-primary-200 rounded-2xl overflow-hidden">
              <div className="px-8 py-6 border-b border-primary-100">
                <h2 className="text-2xl font-normal text-primary-900" style={{ letterSpacing: '-0.02em' }}>Analyze Speech</h2>
              </div>
              <div className="bg-primary-50 border-b border-primary-100">
                <div className="flex px-8">
                  <button className="tab px-6 py-4 text-sm font-medium border-b-2 border-primary-900 text-primary-900" data-input="audio">Audio Upload</button>
                  <button className="tab px-6 py-4 text-sm font-medium border-b-2 border-transparent text-primary-500 hover:text-primary-700 transition-colors" data-input="file">CHAT File</button>
                </div>
              </div>
              <div className="p-8">
                {/* Audio Upload */}
                <div className="input-panel" id="audioPanel">
                  <div className="upload-area bg-primary-50 border border-dashed border-primary-300 rounded-2xl p-12 text-center cursor-pointer hover:bg-primary-100 hover:border-primary-400 transition-all" id="audioUploadArea">
                    <svg className="mx-auto h-10 w-10 text-primary-400" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                      <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                    <p className="mt-4 text-base text-primary-700">Drop audio file here</p>
                    <p className="mt-1 text-sm text-primary-500">WAV, MP3, FLAC, OGG</p>
                  </div>
                  <input type="file" className="hidden" id="audioFileInput" accept=".wav,.mp3,.flac" />
                  <div id="selectedAudioFile" className="mt-3 text-sm text-accent-600"></div>

                  {/* In-browser Recording Controls */}
                  <div id="audioRecordSection" className="mt-6 bg-primary-50 rounded-xl p-5 border border-primary-200">
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="text-sm font-medium text-primary-900">Record Audio</h3>
                      <div className="flex items-center gap-2">
                        <button
                          id="audioRecordButton"
                          type="button"
                          className="px-4 py-1.5 rounded-full bg-primary-900 text-white text-xs font-medium hover:bg-primary-800 transition-colors"
                        >
                          Start recording
                        </button>
                        <button
                          id="audioStopButton"
                          type="button"
                          className="px-4 py-1.5 rounded-full bg-red-100 text-red-700 text-xs font-medium hover:bg-red-200 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                          disabled
                        >
                          Stop &amp; analyze
                        </button>
                      </div>
                    </div>
                    <div className="flex items-center justify-between text-xs text-primary-500">
                      <div id="audioRecordStatus" className="flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-gray-300" id="audioRecordIndicator"></span>
                        <span id="audioRecordStatusText">Microphone idle</span>
                      </div>
                      <div id="audioRecordTimer" className="font-mono text-primary-600 hidden">00:00</div>
                    </div>
                    <p id="audioRecordError" className="mt-2 text-xs text-red-600 hidden"></p>
                  </div>

                  <button className="mt-6 w-full px-6 py-4 bg-primary-900 text-white rounded-xl text-sm font-medium hover:bg-primary-800 transition-all disabled:opacity-40 disabled:cursor-not-allowed" onClick={() => window.predictFromAudio?.()} id="predictAudioBtn" disabled>
                    Analyze Audio
                  </button>
                </div>

                {/* CHAT File Upload */}
                <div className="input-panel hidden" id="filePanel">
                  <div className="upload-area bg-primary-50 border border-dashed border-primary-300 rounded-2xl p-12 text-center cursor-pointer hover:bg-primary-100 hover:border-primary-400 transition-all" id="chaUploadArea">
                    <svg className="mx-auto h-10 w-10 text-primary-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    <p className="mt-4 text-base text-primary-700">Drop CHAT file here</p>
                    <p className="mt-1 text-sm text-primary-500">Supports .cha files</p>
                  </div>
                  <input type="file" className="hidden" id="chaFileInput" accept=".cha,.CHA,text/plain" />
                  <div id="selectedChaFile" className="mt-3 text-sm text-accent-600"></div>
                  <button className="mt-6 w-full px-6 py-4 bg-primary-900 text-white rounded-xl text-sm font-medium hover:bg-primary-800 transition-all disabled:opacity-40 disabled:cursor-not-allowed" onClick={() => window.predictFromChatFile?.()} id="predictChaBtn" disabled>
                    Analyze File
                  </button>
                </div>
              </div>
            </div>

            {/* Results Section */}
            <div className="bg-white border border-primary-200 rounded-2xl overflow-hidden">
              <div className="px-8 py-6 border-b border-primary-100">
                <h2 className="text-2xl font-normal text-primary-900" style={{ letterSpacing: '-0.02em' }}>Analysis Results</h2>
              </div>
              <div className="p-8" id="resultsArea">
                <div className="text-center py-20 text-primary-400 text-sm">
                  Upload an audio file or CHAT transcript to see results
                </div>
              </div>
            </div>
          </div>

          {/* Waveform Display */}
          <div id="waveformSectionResults" className="hidden mt-8 bg-white border border-primary-200 rounded-2xl overflow-hidden">
            <div className="px-8 py-6 border-b border-primary-100">
              <h3 className="text-lg font-normal text-primary-900 mb-0.5" style={{ letterSpacing: '-0.02em' }}>Child Speech Waveform</h3>
              <p className="text-sm text-primary-500">Visual representation of the child&apos;s speech audio</p>
            </div>
            <div className="p-8">
              <div className="bg-white rounded-xl p-5 border border-primary-200">
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
          <div className="mt-8 hidden bg-white border border-primary-200 rounded-2xl overflow-hidden" id="annotationCard">

            {/* Title row — same style as Analyze Speech header */}
            <div className="px-8 py-6 border-b border-primary-100 flex items-center justify-between">
              <h2 className="text-2xl font-normal text-primary-900" style={{ letterSpacing: '-0.02em' }}>Annotated Transcript</h2>
              <span id="annotationCount" className="px-3 py-1 bg-primary-100 text-primary-600 text-xs font-medium rounded-full">Features Marked</span>
            </div>

            {/* Controls strip — same style as the tab bar */}
            <div className="bg-primary-50 border-b border-primary-100 px-8 py-3">
              <div className="flex flex-wrap items-center gap-3">
                <div className="flex-1 min-w-[180px]">
                  <input type="text" id="transcriptSearch" placeholder="Search transcript..." className="w-full px-3 py-1.5 bg-white border border-primary-200 rounded-lg text-sm focus:outline-none focus:border-primary-400 transition-all" />
                </div>
                <div className="flex items-center gap-2">
                  <label className="text-xs text-primary-500 whitespace-nowrap">Filter:</label>
                  <select id="featureFilter" className="px-3 py-1.5 bg-white border border-primary-200 rounded-lg text-sm focus:outline-none focus:border-primary-400 transition-all">
                    <option value="all">All Features</option>
                  </select>
                </div>
                <button id="clearFilters" className="px-3 py-1.5 text-xs border border-primary-200 text-primary-500 rounded-lg hover:border-primary-400 hover:text-primary-800 transition-colors bg-white">Clear</button>
                <label className="flex items-center gap-1.5 cursor-pointer ml-auto">
                  <input type="checkbox" id="semanticCoherenceToggle" className="w-3.5 h-3.5 text-primary-600 rounded" />
                  <span className="text-xs text-primary-500 whitespace-nowrap">Semantic Coherence</span>
                </label>
              </div>
            </div>

            {/* Detected Features section */}
            <div id="featureSummaryPanel" className="px-8 py-5 border-b border-primary-100">
              <p className="text-xs font-medium text-primary-400 uppercase tracking-widest mb-3">Detected Features</p>
              <div id="featureSummaryContent" className="flex flex-wrap gap-2 items-start"></div>
            </div>

            {/* Transcript body */}
            <div className="p-8">
              <div id="annotatedTranscript" className="transcript-container max-h-[560px] overflow-y-auto"></div>

            </div>
          </div>

          {/* Local SHAP Explanation */}
          <div id="localShapSection" className="mt-8 hidden">
            <h3 className="text-xl font-medium text-primary-900 mb-2">Why this prediction was made</h3>
            <p className="text-sm text-primary-600 mb-5">
              This waterfall plot explains how each conversational feature contributed to the final ASD / TD prediction for this specific transcript.
            </p>
            <div className="bg-white rounded-xl p-5 border border-primary-200">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img id="localShapWaterfall" className="w-full rounded-xl border border-primary-100" alt="Local SHAP Waterfall Explanation" />
            </div>
          </div>

          {/* Counterfactual Explanation */}
          <div id="counterfactualSection" className="mt-8 hidden">
            <h3 className="text-xl font-medium text-primary-900 mb-2">What would change this prediction?</h3>
            <p className="text-sm text-primary-600 mb-5">
              This analysis shows the smallest realistic changes required to flip the model&apos;s prediction to the opposite class.
            </p>
            <div id="whatIfBox" className="bg-primary-50 border border-primary-200 rounded-xl p-5 mb-5 text-primary-900"></div>
            <div className="grid md:grid-cols-3 gap-4 mb-5">
              <div className="bg-white rounded-xl p-5 border border-primary-200">
                <p className="text-sm text-primary-600">Prediction flipped</p>
                <p id="cfFlipped" className="text-xl font-bold"></p>
              </div>
              <div className="bg-white rounded-xl p-5 border border-primary-200">
                <p className="text-sm text-primary-600">Overall change (L2)</p>
                <p id="cfL2" className="text-xl font-bold"></p>
              </div>
              <div className="bg-white rounded-xl p-5 border border-primary-200">
                <p className="text-sm text-primary-600">Features changed</p>
                <p id="cfTotal" className="text-xl font-bold"></p>
              </div>
            </div>
            <div className="bg-white rounded-xl p-5 border border-primary-200">
              <h4 className="text-base font-medium text-primary-900 mb-4">Most influential feature changes</h4>
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
          <div id="cfChatSection" className="mt-8 bg-primary-50 border border-primary-200 rounded-xl p-6 hidden">
            <h3 className="text-lg font-medium text-primary-900 mb-2">Explore a What-If Scenario</h3>
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
              <button onClick={() => window.simulateCounterfactualChat?.()} className="px-6 py-2 rounded-xl bg-primary-900 text-white text-sm hover:bg-primary-800 transition">Ask</button>
            </div>
            <div id="cfChatResponse" className="hidden bg-white border border-primary-200 rounded-xl p-5 text-sm text-primary-900"></div>
          </div>
        </div>

        {/* Training Mode */}
        <div className="mode-content hidden" id="trainingMode">
          {/* Training mode tabs row */}
          <div className="bg-primary-50 border-b border-primary-200 rounded-t-xl">
            <div className="flex gap-1 px-2 pt-2">
              <button type="button" className="training-tab px-6 py-3 text-sm font-medium border-b-2 border-primary-900 text-primary-900 rounded-t-lg" data-training-tab="feature-extraction">Feature Extraction</button>
              <button type="button" className="training-tab px-6 py-3 text-sm font-medium border-b-2 border-transparent text-primary-500 hover:text-primary-700 transition-colors rounded-t-lg" data-training-tab="training">Training</button>
              <button type="button" className="training-tab px-6 py-3 text-sm font-medium border-b-2 border-transparent text-primary-500 hover:text-primary-700 transition-colors rounded-t-lg" data-training-tab="trained-models">Trained Models</button>
            </div>
          </div>

          {/* Feature Extraction Section */}
          <div className="training-tab-panel py-8" id="trainingTabFeatureExtraction" data-training-tab="feature-extraction">
            <div className="bg-white border border-primary-200 rounded-2xl overflow-hidden">
              <div className="px-8 py-6 border-b border-primary-100 flex items-center justify-between">
                <h2 className="text-2xl font-normal text-primary-900" style={{ letterSpacing: '-0.02em' }}>Feature Extraction</h2>
                <button className="px-5 py-2 bg-primary-900 text-white rounded-lg text-sm font-medium hover:bg-primary-800 transition-all" onClick={() => window.loadDatasets?.()}>Refresh</button>
              </div>
              <div className="p-8">
                <p className="text-sm text-primary-600 mb-6">
                  Select datasets from your file system to extract features. Extracted features will be saved to CSV files.
                </p>
                <div className="grid lg:grid-cols-2 gap-6">
                  <div className="bg-primary-50 border border-primary-200 rounded-xl p-6" style={{ maxHeight: '500px', overflowY: 'auto' }}>
                    <h3 className="text-base font-medium text-primary-900 mb-4">Select Datasets to Extract</h3>
                    <div id="extractionDatasetList">
                      <div className="text-center py-12 text-primary-400 text-sm">Click Refresh to load datasets</div>
                    </div>
                  </div>
                  <div className="bg-primary-50 border border-primary-200 rounded-xl p-6">
                    <h3 className="text-base font-medium text-primary-900 mb-5">Extraction Settings</h3>
                    <div className="mb-5">
                      <label className="block text-sm font-medium text-primary-700 mb-2">Component</label>
                      <select className="w-full px-4 py-3 bg-white border border-primary-200 rounded-xl text-sm focus:outline-none focus:border-primary-400 transition-all" id="extractionComponent">
                        <option value="pragmatic_conversational">Pragmatic &amp; Conversational</option>
                        <option value="acoustic_prosodic">Acoustic &amp; Prosodic</option>
                        <option value="syntactic_semantic">Syntactic &amp; Semantic</option>
                      </select>
                      <p className="text-xs text-primary-500 mt-2">Select which component&apos;s features to extract</p>
                    </div>
                    <div className="mb-5">
                      <label className="block text-sm font-medium text-primary-700 mb-2">Max Samples per Dataset</label>
                      <input type="number" id="maxSamplesExtraction" min="1" className="w-full px-4 py-3 bg-white border border-primary-200 rounded-xl text-sm focus:outline-none focus:border-primary-400 transition-all" placeholder="Leave empty for all samples" />
                      <p className="text-xs text-primary-500 mt-2">Limit samples for large datasets. Leave empty to extract all.</p>
                    </div>
                    <button className="w-full px-6 py-4 bg-primary-900 text-white rounded-xl text-sm font-medium hover:bg-primary-800 transition-all" onClick={() => window.extractFeatures?.()}>
                      Extract Features
                    </button>
                    <div className="mt-4 bg-white border border-primary-200 rounded-xl p-4 hidden" id="extractionStatus">
                      <div id="extractionStatusContent"></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Training Section */}
          <div className="training-tab-panel hidden py-8" id="trainingTabTraining" data-training-tab="training">
            <div className="bg-white border border-primary-200 rounded-2xl overflow-hidden">
              <div className="px-8 py-6 border-b border-primary-100 flex items-center justify-between">
                <h2 className="text-2xl font-normal text-primary-900" style={{ letterSpacing: '-0.02em' }}>Training</h2>
                <button className="px-5 py-2 bg-primary-900 text-white rounded-lg text-sm font-medium hover:bg-primary-800 transition-all" onClick={() => window.loadAvailableDatasetsForTraining?.()}>Refresh</button>
              </div>
              <div className="p-8">
                <div className="grid lg:grid-cols-2 gap-6 items-stretch">
                  {/* Dataset Selection for Training */}
                  <div className="bg-primary-50 border border-primary-200 rounded-xl p-6 flex flex-col">
                    <h3 className="text-base font-medium text-primary-900 mb-4 flex-shrink-0">Available Datasets (from CSV)</h3>
                    <div id="datasetList" className="flex-1 overflow-y-auto min-h-0">
                      <div className="text-center py-12 text-primary-400 text-sm">Click Refresh to load datasets</div>
                    </div>
                  </div>
                  {/* Training Controls */}
                  <div className="bg-primary-50 border border-primary-200 rounded-xl p-6">
                    <h3 className="text-base font-medium text-primary-900 mb-5">Training Controls</h3>
                    <div className="mb-6">
                      <label className="block text-sm font-medium text-primary-700 mb-2">Component</label>
                      <select className="w-full px-4 py-3 bg-white border border-primary-200 rounded-xl text-sm focus:outline-none focus:border-primary-400 transition-all" id="trainingComponent">
                        <option value="pragmatic_conversational">Pragmatic &amp; Conversational</option>
                        <option value="acoustic_prosodic">Acoustic &amp; Prosodic</option>
                        <option value="syntactic_semantic">Syntactic &amp; Semantic (Dummy Features)</option>
                      </select>
                      <p className="text-xs text-primary-500 mt-2">Note: Acoustic &amp; Syntactic use placeholder features for testing</p>
                    </div>
                    <div className="mb-6">
                      <label className="block text-sm font-medium text-primary-700 mb-3">Feature Selection</label>
                      <div className="space-y-3">
                        <label className="flex items-center cursor-pointer p-4 bg-white border border-primary-200 rounded-xl hover:border-primary-300 transition-colors">
                          <input type="checkbox" id="featureSelectionEnabled" defaultChecked className="w-4 h-4 text-primary-600 rounded" />
                          <span className="ml-3 text-sm text-primary-900">Enable feature selection</span>
                        </label>
                        <div id="featureCountSection">
                          <label className="block text-xs text-primary-600 mb-2">Number of features to select</label>
                          <input type="number" id="nFeatures" defaultValue="30" min="1" max="218" className="w-full px-4 py-3 bg-white border border-primary-200 rounded-xl text-sm focus:outline-none focus:border-primary-400 transition-all" />
                          <p className="text-xs text-primary-500 mt-2">Default: 30 (max: 218 for pragmatic, 20 for others)</p>
                        </div>
                      </div>
                    </div>
                    <div className="mb-6">
                      <label className="block text-sm font-medium text-primary-700 mb-3">Counterfactual Explanations</label>
                      <label className="flex items-center cursor-pointer p-4 bg-white border border-primary-200 rounded-xl hover:border-primary-300 transition-colors">
                        <input type="checkbox" id="enableAutoencoder" className="w-4 h-4 text-primary-600 rounded" />
                        <div className="ml-3 flex-1">
                          <span className="text-sm text-primary-900">Enable counterfactual autoencoder</span>
                          <p className="text-xs text-primary-500 mt-0.5">Train autoencoder for counterfactual explanations (may crash on macOS)</p>
                        </div>
                      </label>
                    </div>
                    <div className="mb-6">
                      <label className="block text-sm font-medium text-primary-700 mb-3">Model Types</label>
                      <p className="text-xs text-primary-500 mb-3">Available models change based on selected component</p>
                      <div id="modelTypesContainer" className="grid grid-cols-2 gap-3">
                        <label className="flex items-center cursor-pointer p-4 bg-white border border-primary-200 rounded-xl hover:border-primary-300 transition-colors">
                          <input type="checkbox" value="svm" defaultChecked className="w-4 h-4 text-primary-600 rounded" />
                          <span className="ml-3 text-sm text-primary-900">SVM</span>
                        </label>
                        <label className="flex items-center cursor-pointer p-4 bg-white border border-primary-200 rounded-xl hover:border-primary-300 transition-colors">
                          <input type="checkbox" value="logistic" defaultChecked className="w-4 h-4 text-primary-600 rounded" />
                          <span className="ml-3 text-sm text-primary-900">Logistic Regression</span>
                        </label>
                      </div>
                    </div>
                    <div className="mb-6">
                      <label className="block text-sm font-medium text-primary-700 mb-3">Training Parameters</label>
                      <div className="space-y-3">
                        <div className="bg-white border border-primary-200 rounded-xl p-4">
                          <label className="block text-xs text-primary-600 mb-2">Test Set Size (%)</label>
                          <input type="number" id="testSize" defaultValue="20" min="10" max="40" step="5" className="w-full px-4 py-2 bg-primary-50 rounded-lg text-sm focus:outline-none border border-primary-200 focus:border-primary-400 transition-all" />
                          <p className="text-xs text-primary-500 mt-1">Percentage of data reserved for testing</p>
                        </div>
                        <div className="bg-white border border-primary-200 rounded-xl p-4">
                          <label className="block text-xs text-primary-600 mb-2">Random Seed</label>
                          <input type="number" id="randomState" defaultValue="42" min="0" max="999" className="w-full px-4 py-2 bg-primary-50 rounded-lg text-sm focus:outline-none border border-primary-200 focus:border-primary-400 transition-all" />
                          <p className="text-xs text-primary-500 mt-1">For reproducible results</p>
                        </div>
                      </div>
                    </div>
                    {/* Advanced Options */}
                    <div className="mb-6">
                      <button onClick={() => window.toggleHyperparameters?.()} className="w-full flex items-center justify-between p-4 bg-white border border-primary-200 rounded-xl hover:border-primary-300 transition-colors">
                        <span className="text-sm font-medium text-primary-900">Advanced Hyperparameters</span>
                        <svg id="hyperparamChevron" className="w-4 h-4 text-primary-500 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
                        </svg>
                      </button>
                      <div id="hyperparamSection" className="hidden mt-3 space-y-3">
                        <div className="bg-white border border-primary-200 rounded-xl p-4">
                          <p className="text-xs text-primary-600 mb-4">Customize hyperparameters for each selected model type. Leave default for recommended values.</p>
                          <div id="hyperparamControls" className="space-y-4"></div>
                        </div>
                      </div>
                    </div>
                    <button className="w-full px-6 py-4 bg-primary-900 text-white rounded-xl text-sm font-medium hover:bg-primary-800 transition-all" onClick={() => window.startTraining?.()}>Start Training</button>
                    <div className="mt-4 bg-white border border-primary-200 rounded-xl p-4 hidden" id="trainingStatus">
                      <div id="trainingStatusContent"></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Trained Models Section */}
          <div className="training-tab-panel hidden py-8" id="trainingTabTrainedModels" data-training-tab="trained-models">
            <div className="bg-white border border-primary-200 rounded-2xl overflow-hidden">
              <div className="px-8 py-6 border-b border-primary-100 flex items-center justify-between">
                <h2 className="text-2xl font-normal text-primary-900" style={{ letterSpacing: '-0.02em' }}>Trained Models</h2>
                <button className="px-5 py-2 bg-primary-900 text-white rounded-lg text-sm font-medium hover:bg-primary-800 transition-all" onClick={() => window.loadAvailableModels?.()}>Refresh</button>
              </div>
              <div className="p-8">
                <div id="availableModelsContainer">
                  <div className="text-center py-16 text-primary-400 text-sm">Click Refresh to load trained models</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Model Details Modal */}
      <div id="modelDetailsModal" className="fixed inset-0 bg-black bg-opacity-50 hidden items-center justify-center z-50" onClick={(e) => window.closeModelDetails?.(e)}>
        <div className="bg-white rounded-2xl max-w-5xl w-full mx-4 max-h-[90vh] overflow-y-auto shadow-2xl" onClick={(e) => e.stopPropagation()}>
          <div className="sticky top-0 bg-white px-8 py-5 border-b border-primary-200 flex items-center justify-between rounded-t-2xl">
            <h2 className="text-xl font-medium text-primary-900">Model Performance Details</h2>
            <button onClick={() => window.closeModelDetails?.()} className="text-primary-400 hover:text-primary-900 transition-colors p-1">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
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
