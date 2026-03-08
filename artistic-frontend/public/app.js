// API URL
function getApiUrl() {
    return document.getElementById('apiUrl').value.replace(/\/$/, '');
}

function updateToggleSlider() {
    const activeOption = document.querySelector('.toggle-option.active');
    const toggleSwitch = document.getElementById('modeToggle');
    const toggleSlider = document.getElementById('toggleSlider');

    if (!activeOption || !toggleSwitch || !toggleSlider) return;

    const optionRect = activeOption.getBoundingClientRect();
    const switchRect = toggleSwitch.getBoundingClientRect();
    const relativeLeft = optionRect.left - switchRect.left;

    toggleSlider.style.width = `${optionRect.width}px`;
    toggleSlider.style.transform = `translateX(${relativeLeft - 4}px)`;
}

function initHomeUiBindings() {
    // Mode switching with toggle (User / Training)
    const modeToggle = document.getElementById('modeToggle');
    if (modeToggle && modeToggle.dataset.artisticBound !== '1') {
        modeToggle.dataset.artisticBound = '1';

        modeToggle.addEventListener('click', (e) => {
            const option = e.target.closest('.toggle-option');
            if (!option) return;

            document.querySelectorAll('.toggle-option').forEach(opt => opt.classList.remove('active'));
            option.classList.add('active');

            const mode = option.dataset.mode;
            document.querySelectorAll('.mode-content').forEach(c => c.classList.add('hidden'));
            const modeEl = document.getElementById(mode + 'Mode');
            if (modeEl) modeEl.classList.remove('hidden');

            // Toggle API config bar visibility
            const apiConfigBar = document.getElementById('apiConfigBar');
            const landingSection = document.getElementById('landingSection');
            if (apiConfigBar) {
                if (mode === 'training') {
                    apiConfigBar.classList.remove('hidden');
                    if (landingSection) landingSection.classList.add('hidden');
                    // Auto-load models when entering training mode
                    setTimeout(() => {
                        try {
                            loadAvailableModels();
                        } catch (err) {
                            console.error('Failed to auto-load models:', err);
                        }
                    }, 100);
                } else {
                    apiConfigBar.classList.add('hidden');
                    if (landingSection) landingSection.classList.remove('hidden');
                }
            }

            updateToggleSlider();
        });
    }

    // Initialize toggle slider position
    setTimeout(updateToggleSlider, 100);
    if (!window.__artisticToggleResizeBound) {
        window.__artisticToggleResizeBound = true;
        window.addEventListener('resize', updateToggleSlider);
    }

    // Tab switching (Audio Upload / CHAT File)
    document.querySelectorAll('.tab').forEach(tab => {
        if (tab.dataset.artisticBound === '1') return;
        tab.dataset.artisticBound = '1';

        tab.addEventListener('click', () => {
            document.querySelectorAll('.tab').forEach(t => {
                t.classList.remove('border-primary-900', 'text-primary-900');
                t.classList.add('border-transparent', 'text-primary-500');
            });
            tab.classList.add('border-primary-900', 'text-primary-900');
            tab.classList.remove('border-transparent', 'text-primary-500');

            const inputType = tab.dataset.input;
            document.querySelectorAll('.input-panel').forEach(p => p.classList.add('hidden'));
            const panel = document.getElementById(inputType + 'Panel');
            if (panel) panel.classList.remove('hidden');

            // Load models when switching to a prediction tab
            try {
                loadModelsForPrediction();
            } catch (err) {
                console.error('Failed to load models for prediction:', err);
            }
        });
    });

    // Training mode tab switching (Feature Extraction, Training, Trained Models)
    document.querySelectorAll('.training-tab').forEach(tab => {
        if (tab.dataset.artisticBound === '1') return;
        tab.dataset.artisticBound = '1';

        tab.addEventListener('click', () => {
            document.querySelectorAll('.training-tab').forEach(t => {
                t.classList.remove('border-primary-900', 'text-primary-900');
                t.classList.add('border-transparent', 'text-primary-500');
            });
            tab.classList.add('border-primary-900', 'text-primary-900');
            tab.classList.remove('border-transparent', 'text-primary-500');

            const tabId = tab.getAttribute('data-training-tab');
            document.querySelectorAll('.training-tab-panel').forEach(p => p.classList.add('hidden'));
            const panel = document.querySelector(`.training-tab-panel[data-training-tab="${tabId}"]`);
            if (panel) panel.classList.remove('hidden');
        });
    });

    // File upload area bindings (idempotent)
    setupUploadArea('audioUploadArea', 'audioFileInput', 'selectedAudioFile', ['.wav', '.mp3', '.flac', '.ogg', '.m4a']);
    setupUploadArea('chaUploadArea', 'chaFileInput', 'selectedChaFile', ['.cha']);

    // Initialize audio recording (idempotent)
    const recordSection = document.getElementById('audioRecordSection');
    if (recordSection && recordSection.dataset.artisticBound !== '1') {
        recordSection.dataset.artisticBound = '1';
        setupAudioRecording();
    }

    // Bind analyze buttons directly to prediction functions as a fallback
    const analyzeAudioBtn = document.getElementById('predictAudioBtn');
    if (analyzeAudioBtn && analyzeAudioBtn.dataset.artisticBound !== '1') {
        analyzeAudioBtn.dataset.artisticBound = '1';
        analyzeAudioBtn.addEventListener('click', () => {
            try {
                predictFromAudio();
            } catch (err) {
                console.error('predictFromAudio failed:', err);
            }
        });
    }

    const analyzeChatBtn = document.getElementById('predictChaBtn');
    if (analyzeChatBtn && analyzeChatBtn.dataset.artisticBound !== '1') {
        analyzeChatBtn.dataset.artisticBound = '1';
        analyzeChatBtn.addEventListener('click', () => {
            try {
                predictFromChatFile();
            } catch (err) {
                console.error('predictFromChatFile failed:', err);
            }
        });
    }
}

// Expose an explicit initializer so the React page can re-bind after route changes.
window.__artisticInitHomeUi = initHomeUiBindings;

// Expose functions used by React `onClick` handlers.
// (When this file is loaded via a dynamically-created script tag, relying on implicit globals can be brittle.)
window.testConnection = testConnection;
window.predictFromAudio = predictFromAudio;
window.predictFromChatFile = predictFromChatFile;
window.loadDatasets = loadDatasets;
window.extractFeatures = extractFeatures;
window.loadAvailableDatasetsForTraining = loadAvailableDatasetsForTraining;
window.startTraining = startTraining;
window.loadAvailableModels = loadAvailableModels;
window.toggleHyperparameters = toggleHyperparameters;
window.simulateCounterfactualChat = simulateCounterfactualChat;
window.closeModelDetails = closeModelDetails;
window.askCounterfactualGPT = askCounterfactualGPT;

// Run once on initial page load (Home mounts this script after DOM is ready)
initHomeUiBindings();

// File upload handling
function setupUploadArea(areaId, inputId, selectedId, allowedExtensions) {
    const area = document.getElementById(areaId);
    const input = document.getElementById(inputId);
    const selected = selectedId ? document.getElementById(selectedId) : null;

    if (!area || !input) return;
    if (area.dataset.artisticBound === '1') return;
    area.dataset.artisticBound = '1';

    area.addEventListener('click', () => input.click());

    area.addEventListener('dragover', (e) => {
        e.preventDefault();
        area.classList.add('dragover');
    });

    area.addEventListener('dragleave', () => {
        area.classList.remove('dragover');
    });

    area.addEventListener('drop', (e) => {
        e.preventDefault();
        area.classList.remove('dragover');

        const file = e.dataTransfer.files[0];
        if (file) {
            // Ensure the underlying <input type="file"> actually contains the dropped file,
            // since prediction functions read from `input.files`.
            try {
                const dataTransfer = new DataTransfer();
                dataTransfer.items.add(file);
                input.files = dataTransfer.files;
                input.dispatchEvent(new Event('change', { bubbles: true }));
            } catch (err) {
                // Fallback: still update UI even if we can't set `files` programmatically.
                handleFileSelect(file, input, selected, allowedExtensions);
            }
        }
    });

    input.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleFileSelect(file, input, selected, allowedExtensions);
        }
    });
}

// Audio recording state
let mediaRecorder = null;
let recordedChunks = [];
let recordingStream = null;
let recordingTimerInterval = null;
let recordingStartTime = null;

function setupAudioRecording() {
    const section = document.getElementById('audioRecordSection');
    if (!section) {
        return;
    }

    const recordButton = document.getElementById('audioRecordButton');
    const stopButton = document.getElementById('audioStopButton');
    const statusText = document.getElementById('audioRecordStatusText');
    const indicator = document.getElementById('audioRecordIndicator');
    const timerEl = document.getElementById('audioRecordTimer');
    const errorEl = document.getElementById('audioRecordError');

    // If browser does not support recording, hide the section
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || typeof MediaRecorder === 'undefined') {
        section.classList.add('hidden');
        return;
    }

    const updateTimer = () => {
        if (!recordingStartTime || !timerEl) return;
        const elapsedMs = Date.now() - recordingStartTime;
        const totalSeconds = Math.floor(elapsedMs / 1000);
        const minutes = String(Math.floor(totalSeconds / 60)).padStart(2, '0');
        const seconds = String(totalSeconds % 60).padStart(2, '0');
        timerEl.textContent = `${minutes}:${seconds}`;
    };

    const resetRecordingState = () => {
        if (recordingTimerInterval) {
            clearInterval(recordingTimerInterval);
            recordingTimerInterval = null;
        }
        recordingStartTime = null;
        if (timerEl) {
            timerEl.textContent = '00:00';
            timerEl.classList.add('hidden');
        }
        if (indicator) {
            indicator.className = 'w-2.5 h-2.5 rounded-full bg-gray-300';
        }
        if (statusText) {
            statusText.textContent = 'Microphone idle';
        }
        if (recordButton) {
            recordButton.disabled = false;
        }
        if (stopButton) {
            stopButton.disabled = true;
        }
        if (recordingStream) {
            recordingStream.getTracks().forEach(t => t.stop());
            recordingStream = null;
        }
        mediaRecorder = null;
        recordedChunks = [];
    };

    recordButton.addEventListener('click', async () => {
        if (errorEl) {
            errorEl.textContent = '';
            errorEl.classList.add('hidden');
        }

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            recordingStream = stream;
            recordedChunks = [];

            let options = {};
            let chosenMimeType = null;
            // Prefer formats that the backend already supports: OGG or M4A/MP4 audio
            if (MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported('audio/ogg;codecs=opus')) {
                chosenMimeType = 'audio/ogg;codecs=opus';
                options.mimeType = chosenMimeType;
            } else if (MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported('audio/mp4')) {
                chosenMimeType = 'audio/mp4';
                options.mimeType = chosenMimeType;
            }

            try {
                mediaRecorder = new MediaRecorder(stream, options);
            } catch (e) {
                // Fallback without explicit mimeType if options are rejected
                mediaRecorder = new MediaRecorder(stream);
            }

            mediaRecorder.ondataavailable = (event) => {
                if (event.data && event.data.size > 0) {
                    recordedChunks.push(event.data);
                }
            };

            mediaRecorder.onerror = (event) => {
                console.error('MediaRecorder error:', event.error);
                if (errorEl) {
                    errorEl.textContent = 'Recording error. Please try again or use file upload.';
                    errorEl.classList.remove('hidden');
                }
                resetRecordingState();
            };

            mediaRecorder.onstop = async () => {
                try {
                    if (!recordedChunks.length) {
                        if (errorEl) {
                            errorEl.textContent = 'No audio captured. Please try recording again.';
                            errorEl.classList.remove('hidden');
                        }
                        resetRecordingState();
                        return;
                    }

                    const effectiveMime = mediaRecorder.mimeType || chosenMimeType || 'audio/ogg';
                    let extension = '.ogg';
                    if (effectiveMime.includes('mp4') || effectiveMime.includes('m4a')) {
                        extension = '.m4a';
                    }

                    const blob = new Blob(recordedChunks, { type: effectiveMime });
                    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
                    const fileName = `recording-${timestamp}${extension}`;
                    const file = new File([blob], fileName, { type: effectiveMime });

                    // Populate the existing file input so we fully reuse the current flow
                    const fileInput = document.getElementById('audioFileInput');
                    if (fileInput) {
                        const dataTransfer = new DataTransfer();
                        dataTransfer.items.add(file);
                        fileInput.files = dataTransfer.files;
                        // Trigger change handler to update UI and waveform
                        fileInput.dispatchEvent(new Event('change', { bubbles: true }));
                    }

                    // Automatically trigger analysis using existing flow
                    await predictFromAudio();
                } catch (err) {
                    console.error('Error handling recorded audio:', err);
                    if (errorEl) {
                        errorEl.textContent = 'Failed to process recorded audio. Please try again or upload a file.';
                        errorEl.classList.remove('hidden');
                    }
                } finally {
                    resetRecordingState();
                }
            };

            mediaRecorder.start();

            if (indicator) {
                indicator.className = 'w-2.5 h-2.5 rounded-full bg-red-500 status-training';
            }
            if (statusText) {
                statusText.textContent = 'Recording in progress...';
            }
            if (timerEl) {
                timerEl.classList.remove('hidden');
            }
            recordingStartTime = Date.now();
            updateTimer();
            recordingTimerInterval = setInterval(updateTimer, 1000);

            recordButton.disabled = true;
            stopButton.disabled = false;
        } catch (err) {
            console.error('Microphone access error:', err);
            if (errorEl) {
                errorEl.textContent = 'Could not access microphone. Please allow microphone permissions or use file upload.';
                errorEl.classList.remove('hidden');
            }
            resetRecordingState();
        }
    });

    stopButton.addEventListener('click', () => {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            mediaRecorder.stop();
        } else {
            resetRecordingState();
        }
    });
}

function handleFileSelect(file, input, selected, allowedExtensions) {
    console.log('File selected:', file.name);
    console.log('Input ID:', input.id);
    console.log('Allowed extensions:', allowedExtensions);

    const ext = '.' + file.name.split('.').pop().toLowerCase();
    console.log('Detected extension:', ext);

    if (!allowedExtensions.includes(ext)) {
        let errorMsg = `Invalid file type "${ext}". Allowed: ${allowedExtensions.join(', ')}\n\n`;
        if (ext === '.cha') {
            errorMsg += 'Tip: Use the "CHAT File" tab for .cha files';
        } else if (['.wav', '.mp3', '.flac'].includes(ext)) {
            errorMsg += 'Tip: Use the "Audio Upload" tab for audio files';
        }
        errorMsg += `\n\nFile: ${file.name}`;
        alert(errorMsg);
        return;
    }

    if (selected) {
        selected.textContent = 'Selected: ' + file.name;
    }

    // Enable the corresponding predict button
    if (input.id === 'audioFileInput') {
        document.getElementById('predictAudioBtn').disabled = false;
        // Display waveform for audio files
        currentAudioFile = file;
        displayWaveform(file);
    } else if (input.id === 'chaFileInput') {
        document.getElementById('predictChaBtn').disabled = false;
    }
}

// API calls
let _connectionPollTimer = null;

async function testConnection() {
    const statusDot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');

    try {
        const response = await fetch(`${getApiUrl()}/health`);
        if (response.ok) {
            const data = await response.json();
            statusDot.className = 'w-2.5 h-2.5 rounded-full bg-green-400 status-connected';
            statusText.textContent = `Connected (${data.models_available} models, ${data.features_supported} features)`;

            if (_connectionPollTimer !== null) {
                clearTimeout(_connectionPollTimer);
                _connectionPollTimer = null;
            }

            loadModelsForPrediction();
        } else {
            throw new Error('Not healthy');
        }
    } catch (error) {
        statusDot.className = 'w-2.5 h-2.5 rounded-full bg-red-400';
        statusText.textContent = 'Disconnected';

        if (_connectionPollTimer === null) {
            _connectionPollTimer = setTimeout(function poll() {
                _connectionPollTimer = null;
                testConnection();
            }, 1000);
        }
    }
}

// ---------------------------------------------------------------------------
// Audio prediction pipeline steps (used to render the progress indicator)
// ---------------------------------------------------------------------------
const AUDIO_PIPELINE_STEPS = [
    {
        id: 'prepare',
        label: 'Preparing',
        sub: 'Upload & format conversion',
        minPct: 0, maxPct: 13,
        icon: `<path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"/>`
    },
    {
        id: 'transcribe',
        label: 'Transcribing Speech',
        sub: 'Whisper speech-to-text',
        minPct: 13, maxPct: 32,
        icon: `<path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"/>`
    },
    {
        id: 'pragmatic',
        label: 'Pragmatic & Conversational',
        sub: 'Turn-taking, topic coherence',
        minPct: 32, maxPct: 52,
        icon: `<path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"/>`
    },
    {
        id: 'acoustic',
        label: 'Acoustic & Prosodic',
        sub: 'Pitch, rhythm, voice features',
        minPct: 52, maxPct: 72,
        icon: `<path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3"/>`
    },
    {
        id: 'syntactic',
        label: 'Syntactic & Semantic',
        sub: 'Language structure & meaning',
        minPct: 72, maxPct: 77,
        icon: `<path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"/>`
    },
    {
        id: 'finalise',
        label: 'Finalising Results',
        sub: 'Fusion, annotation & report',
        minPct: 77, maxPct: 100,
        icon: `<path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>`
    },
];

function _stepStatus(step, pct) {
    if (pct >= step.maxPct) return 'done';
    if (pct >= step.minPct) return 'active';
    return 'pending';
}

function showAudioProgressUI(elementId) {
    const stepsHtml = AUDIO_PIPELINE_STEPS.map(s => `
        <div id="apstep-${s.id}" class="flex items-center gap-4 py-3.5 border-b border-primary-50 last:border-0">
            <div class="ap-step-icon flex-shrink-0 w-9 h-9 bg-white border border-primary-200 rounded-xl flex items-center justify-center transition-all duration-400">
                <svg class="ap-icon-svg w-4 h-4 text-primary-300 transition-colors duration-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">${s.icon}</svg>
                <svg class="ap-icon-check hidden w-4 h-4 text-lime-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>
                <div class="ap-icon-spin hidden w-3.5 h-3.5 rounded-full border-2 border-white border-t-transparent animate-spin"></div>
            </div>
            <div class="flex-1 min-w-0">
                <div class="ap-step-label text-sm text-primary-400 font-normal transition-all duration-300">${s.label}</div>
                <div class="ap-step-sub text-xs text-primary-300 mt-0.5 transition-all duration-300">${s.sub}</div>
            </div>
            <div class="ap-step-badge flex-shrink-0"></div>
        </div>
    `).join('');

    document.getElementById(elementId).innerHTML = `
        <div id="audioProgressContainer">
            <div class="mb-1">
                <p class="text-xs font-medium text-primary-500 uppercase tracking-widest mb-5">Analysing your recording</p>
                <div class="flex items-start justify-between gap-4 mb-4">
                    <div class="flex-1 min-w-0">
                        <div id="apStageName" class="text-base font-medium text-primary-900 mb-1">Starting…</div>
                        <div id="apDetailMsg" class="text-sm text-primary-500 leading-relaxed">Preparing to analyse audio</div>
                    </div>
                    <div class="text-right flex-shrink-0">
                        <div id="apPct" class="text-3xl font-normal text-primary-900 tabular-nums" style="letter-spacing:-0.03em">0%</div>
                        <div id="apElapsed" class="text-xs text-primary-400 mt-0.5">0s elapsed</div>
                    </div>
                </div>
                <div class="w-full h-1.5 bg-primary-100 rounded-full overflow-hidden mb-1">
                    <div id="apBar" class="h-full rounded-full bg-primary-900 transition-all duration-700 ease-out" style="width:0%"></div>
                </div>
            </div>

            <div class="mt-6 border border-primary-100 rounded-2xl px-5 bg-primary-50/40">
                ${stepsHtml}
            </div>

            <div class="mt-5 flex items-center gap-2 text-xs text-primary-400">
                <svg class="w-3.5 h-3.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
                Analysis typically takes 3–7 minutes depending on file length — you can leave this tab open
            </div>
        </div>
    `;
}

let _apStartTime = null;

function updateAudioProgressUI(pct, stage, detail) {
    const bar = document.getElementById('apBar');
    const pctEl = document.getElementById('apPct');
    const stageEl = document.getElementById('apStageName');
    const detailEl = document.getElementById('apDetailMsg');
    const elapsedEl = document.getElementById('apElapsed');
    if (!bar) return;

    bar.style.width = `${pct}%`;
    pctEl.textContent = `${pct}%`;
    stageEl.textContent = stage || 'Processing…';
    if (detail) detailEl.textContent = detail;

    if (_apStartTime) {
        const sec = Math.round((Date.now() - _apStartTime) / 1000);
        const min = Math.floor(sec / 60);
        elapsedEl.textContent = min > 0 ? `${min}m ${sec % 60}s elapsed` : `${sec}s elapsed`;
    }

    AUDIO_PIPELINE_STEPS.forEach(s => {
        const el = document.getElementById(`apstep-${s.id}`);
        if (!el) return;
        const iconBox = el.querySelector('.ap-step-icon');
        const iconSvg = el.querySelector('.ap-icon-svg');
        const iconCheck = el.querySelector('.ap-icon-check');
        const iconSpin = el.querySelector('.ap-icon-spin');
        const labelEl = el.querySelector('.ap-step-label');
        const subEl = el.querySelector('.ap-step-sub');
        const badgeEl = el.querySelector('.ap-step-badge');
        const st = _stepStatus(s, pct);

        // Reset all state classes
        iconBox.classList.remove(
            'bg-white', 'bg-primary-900', 'bg-lime-50',
            'border-primary-200', 'border-primary-900', 'border-lime-200'
        );
        iconSvg.classList.remove('text-primary-300', 'text-white');
        iconSvg.classList.add('hidden');
        iconCheck.classList.add('hidden');
        iconSpin.classList.add('hidden');
        badgeEl.innerHTML = '';

        if (st === 'done') {
            iconBox.classList.add('bg-lime-50', 'border-lime-200');
            iconCheck.classList.remove('hidden');
            labelEl.className = 'ap-step-label text-sm text-primary-700 font-medium transition-all duration-300';
            subEl.className = 'ap-step-sub text-xs text-primary-400 mt-0.5 transition-all duration-300';
            badgeEl.innerHTML = `<span class="text-xs text-lime-700 font-medium">Done</span>`;
        } else if (st === 'active') {
            iconBox.classList.add('bg-primary-900', 'border-primary-900');
            iconSpin.classList.remove('hidden'); // spinner replaces the svg icon
            labelEl.className = 'ap-step-label text-sm text-primary-900 font-medium transition-all duration-300';
            subEl.className = 'ap-step-sub text-xs text-primary-500 mt-0.5 transition-all duration-300';
            badgeEl.innerHTML = `<span class="inline-flex items-center gap-1 text-xs text-primary-500"><span class="w-1.5 h-1.5 rounded-full bg-primary-900 animate-pulse"></span>Running</span>`;
        } else {
            iconBox.classList.add('bg-white', 'border-primary-200');
            iconSvg.classList.remove('hidden');
            iconSvg.classList.add('text-primary-300');
            labelEl.className = 'ap-step-label text-sm text-primary-400 font-normal transition-all duration-300';
            subEl.className = 'ap-step-sub text-xs text-primary-300 mt-0.5 transition-all duration-300';
        }
    });
}

async function predictFromAudio() {
    const fileInput = document.getElementById('audioFileInput');
    const useFusion = true;

    if (!fileInput.files[0]) {
        alert('Please select an audio file');
        return;
    }

    currentAudioFile = fileInput.files[0];
    await displayWaveform(currentAudioFile);

    showAudioProgressUI('resultsArea');
    _apStartTime = Date.now();
    updateAudioProgressUI(0, 'Starting…', 'Sending audio file to server');

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('use_fusion', useFusion);

    let jobId = null;
    try {
        const startResp = await fetch(`${getApiUrl()}/predict/audio/start`, {
            method: 'POST',
            body: formData,
        });
        if (!startResp.ok) {
            const err = await startResp.json().catch(() => ({}));
            const msg = (typeof err.detail === 'string' ? err.detail : null)
                || 'Failed to start prediction job.';
            displayError(msg);
            return;
        }
        const startData = await startResp.json();
        jobId = startData.job_id;
    } catch (err) {
        displayError('Connection error: ' + err.message);
        return;
    }

    // Open SSE stream to receive progress updates
    const sse = new EventSource(`${getApiUrl()}/predict/audio/progress/${jobId}`);

    sse.onmessage = async (event) => {
        let data;
        try { data = JSON.parse(event.data); } catch { return; }

        updateAudioProgressUI(data.progress || 0, data.stage, data.detail);

        if (data.status === 'completed') {
            sse.close();
            updateAudioProgressUI(100, 'Complete', 'Analysis finished successfully');
            // Brief pause so the user sees 100% before results replace the bar
            await new Promise(r => setTimeout(r, 600));
            displayResults(data.result);
            if (currentAudioFile) {
                await displayWaveform(currentAudioFile, data.result);
            }
        } else if (data.status === 'error') {
            sse.close();
            const msg = data.error || 'An error occurred during analysis.';
            displayError(msg);
        }
    };

    sse.onerror = () => {
        sse.close();
        displayError('Lost connection to server. Please try again.');
    };
}

async function predictFromText() {
    const text = document.getElementById('textInput').value;
    const useFusion = true;

    if (!text.trim()) {
        alert('Please enter some text');
        return;
    }

    showLoading('resultsArea');

    try {
        const response = await fetch(`${getApiUrl()}/predict/text`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: text,
                participant_id: 'CHI',
                use_fusion: useFusion
            })
        });

        const data = await response.json();

        if (response.ok) {
            displayResults(data);
        } else {
            // Try to surface the most informative backend error message
            let message = 'Prediction failed';
            if (data) {
                if (typeof data.detail === 'string') {
                    message = data.detail;
                } else if (Array.isArray(data.detail) && data.detail[0]?.msg) {
                    message = data.detail[0].msg;
                } else if (typeof data.message === 'string') {
                    message = data.message;
                }
            }
            displayError(message);
        }
    } catch (error) {
        displayError('Connection error: ' + error.message);
    }
}

async function predictFromChatFile() {
    const fileInput = document.getElementById('chaFileInput');
    const useFusion = true;

    if (!fileInput.files[0]) {
        alert('Please select a CHAT file');
        return;
    }

    console.log('Uploading CHAT file:', fileInput.files[0].name, 'Fusion:', useFusion);
    showLoading('resultsArea');

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('use_fusion', useFusion);

    try {
        const response = await fetch(`${getApiUrl()}/predict/transcript`, {
            method: 'POST',
            body: formData
        });

        console.log('Response status:', response.status);

        const data = await response.json();
        console.log('Response data:', data);

        if (response.ok) {
            displayResults(data);
        } else {
            // Try to extract error message from response
            let errorMsg = 'Unknown error';
            try {
                if (data.detail) {
                    errorMsg = data.detail;
                } else if (data.error) {
                    errorMsg = data.error;
                } else if (typeof data === 'string') {
                    errorMsg = data;
                } else if (data.message) {
                    errorMsg = data.message;
                } else {
                    errorMsg = JSON.stringify(data);
                }
            } catch (e) {
                errorMsg = `Error: ${response.status} ${response.statusText}`;
            }
            console.error('Prediction error:', errorMsg, data);
            displayError(errorMsg);
        }
    } catch (error) {
        console.error('Request error:', error);
        displayError('Connection error: ' + error.message);
    }
}

function showLoading(elementId) {
    document.getElementById(elementId).innerHTML = `
        <div class="text-center py-16">
            <div class="spinner mx-auto mb-4"></div>
            <div class="text-sm text-primary-500">Analyzing...</div>
        </div>
    `;
}

function displayResults(data) {
    const isAsd = data.prediction === 'ASD';
    const confidence = (data.confidence * 100).toFixed(1);

    // Component breakdown if fusion was used
    let componentBreakdown = '';
    if (data.component_breakdown && data.component_breakdown.length > 1) {
        const componentNames = {
            'pragmatic_conversational': 'Pragmatic & Conversational',
            'acoustic_prosodic': 'Acoustic & Prosodic',
            'syntactic_semantic': 'Syntactic & Semantic'
        };
        const componentColors = {
            'pragmatic_conversational': 'green',
            'acoustic_prosodic': 'blue',
            'syntactic_semantic': 'purple'
        };

        componentBreakdown = '<div class="mt-5 pt-5 border-t border-primary-200"><div class="text-xs font-medium text-primary-500 uppercase tracking-wide mb-3">Component Breakdown</div><div class="space-y-2">';

        for (const comp of data.component_breakdown) {
            const compName = componentNames[comp.component] || comp.component;
            const color = componentColors[comp.component] || 'gray';
            const compIsAsd = comp.prediction === 'ASD';
            const compConf = (comp.confidence * 100).toFixed(1);
            const asdProb = ((comp.probabilities.ASD || 0) * 100).toFixed(1);
            const tdProb = ((comp.probabilities.TD || 0) * 100).toFixed(1);

            componentBreakdown += `
                <div class="p-4 bg-primary-50 border border-primary-200 rounded-xl">
                    <div class="flex items-center justify-between mb-2">
                        <div class="flex items-center gap-2">
                            <span class="text-sm font-medium text-primary-900">${compName}</span>
                            <span class="px-2 py-0.5 text-xs rounded-full ${compIsAsd ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'}">${comp.prediction}</span>
                        </div>
                        <span class="text-xs text-primary-500">${compConf}% confidence</span>
                    </div>
                    <div class="flex gap-3 text-xs text-primary-500">
                        <span>ASD: ${asdProb}%</span>
                        <span>TD: ${tdProb}%</span>
                    </div>
                </div>
            `;
        }

        componentBreakdown += '</div></div>';
    }

    document.getElementById('resultsArea').innerHTML = `
        <div class="flex items-center justify-between mb-6">
            <span class="inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium ${isAsd ? 'bg-red-100 text-red-700 border border-red-200' : 'bg-green-100 text-green-700 border border-green-200'}">
                <span class="w-2 h-2 rounded-full ${isAsd ? 'bg-red-500' : 'bg-green-500'}"></span>
                ${data.prediction}
            </span>
            <span class="text-xs text-primary-500">
                ${data.features_extracted} features analyzed
            </span>
        </div>
        
        <div class="mb-6">
            <div class="flex justify-between text-sm mb-2">
                <span class="text-primary-600">Confidence</span>
                <span class="font-medium text-primary-900">${confidence}%</span>
            </div>
            <div class="w-full h-1.5 bg-primary-200 rounded-full overflow-hidden">
                <div class="h-full transition-all duration-700 ${isAsd ? 'bg-red-500' : 'bg-green-500'}" style="width: ${confidence}%"></div>
            </div>
        </div>
        
        <div class="grid grid-cols-2 gap-4 mb-6">
            <div class="p-5 bg-red-50 border border-red-100 rounded-xl text-center">
                <div class="text-3xl font-semibold text-red-600">
                    ${(data.probabilities.ASD * 100).toFixed(1)}%
                </div>
                <div class="text-xs text-primary-500 mt-2">ASD Probability</div>
            </div>
            <div class="p-5 bg-green-50 border border-green-100 rounded-xl text-center">
                <div class="text-3xl font-semibold text-green-600">
                    ${(data.probabilities.TD * 100).toFixed(1)}%
                </div>
                <div class="text-xs text-primary-500 mt-2">TD Probability</div>
            </div>
        </div>
        
        ${componentBreakdown}
        
        <div class="text-xs text-primary-500 pt-4 border-t border-primary-200">
            <div class="mb-1.5">
                <span class="font-medium text-primary-600">Model(s) Used:</span>
                ${data.models_used ?
            `<span class="text-primary-500">${data.models_used.join(', ')}</span>` :
            `<span class="text-primary-500">${data.model_used || 'Unknown'}</span>`
        }
            </div>
            <div class="text-primary-400">
                Input: ${data.input_type}${data.component ? ` · Component: ${data.component}` : ''}
            ${data.duration ? ' · Duration: ' + data.duration.toFixed(1) + 's' : ''}
            </div>
        </div>
    `;

    // ==============================
    // Local SHAP Waterfall
    // ==============================
    const localShapSection = document.getElementById('localShapSection');
    const localShapContainer = document.getElementById('localShapContainer');

    localShapContainer.innerHTML = "";

    // SINGLE MODEL
    if (data.local_shap && data.local_shap.waterfall) {

        const img = document.createElement("img");
        img.src = getApiUrl() + data.local_shap.waterfall + '?t=' + Date.now();
        img.className = "shap-image";

        localShapContainer.appendChild(img);
        localShapSection.classList.remove('hidden');

    }

    // FUSION MODEL
    else if (data.fusion_shap && data.fusion_shap.length) {

        data.fusion_shap.forEach(comp => {

            const wrapper = document.createElement("div");
            wrapper.className = "fusion-shap-block";

            const title = document.createElement("h4");
            title.className = "text-sm font-medium text-primary-800 mb-2";
            title.textContent = comp.component
                .replace(/_/g, " ")
                .replace(/\b\w/g, c => c.toUpperCase());

            const img = document.createElement("img");
            img.src = getApiUrl() + comp.waterfall + '?t=' + Date.now();
            img.className = "w-full rounded-xl border border-primary-100";

            wrapper.appendChild(title);
            wrapper.appendChild(img);

            localShapContainer.appendChild(wrapper);
        });

        localShapSection.classList.remove('hidden');
    } else {
        localShapSection.classList.add('hidden');
    }

    // Show annotated transcript with interactive features
    if (data.annotated_transcript_html) {
        document.getElementById('annotationCard').classList.remove('hidden');
        // Store transcript text for semantic coherence analysis
        const transcriptText = data.transcript || extractTranscriptFromHTML(data.annotated_transcript_html);
        renderAnnotatedTranscript(data.annotated_transcript_html, data.annotation_summary || {}, transcriptText);
    }

    //Counterfactuals
    // SINGLE MODEL
    if (data.counterfactual) {

        renderCounterfactual(data.counterfactual);

        document
            .getElementById("cfChatSection")
            .classList.remove("hidden");
    }


    // FUSION MODEL
    else if (data.fusion_counterfactual && data.fusion_counterfactual.length) {

        const tbody = document.getElementById("cfTableBody");
        tbody.innerHTML = "";

        let first = true;

        data.fusion_counterfactual.forEach(cf => {

            renderCounterfactual(
                cf.counterfactual,
                cf.component,
                !first
            );

            first = false;
        });

        document
            .getElementById("cfChatSection")
            .classList.remove("hidden");
    }


}

window.displayResults = displayResults;

function displayError(message) {
    let additionalHelp = '';

    if (message.includes('No models in registry') || message.includes('No models')) {
        additionalHelp = `
            <div class="mt-5 p-5 bg-primary-50 border border-primary-200 rounded-xl text-left">
                <div class="text-xs font-medium text-primary-700 uppercase tracking-wide mb-3">How to fix</div>
                <div class="text-sm text-primary-600 space-y-1.5">
                    <div>1. Switch to <strong class="text-primary-800">Training Mode</strong></div>
                    <div>2. Click <strong class="text-primary-800">Refresh</strong> to load datasets</div>
                    <div>3. Select one or more datasets</div>
                    <div>4. Click <strong class="text-primary-800">Extract Features</strong></div>
                    <div>5. Train a model using the training scripts</div>
                    <div class="mt-3 pt-3 border-t border-primary-200 text-xs text-primary-500">Or restart the API server to reload existing models</div>
                </div>
            </div>
        `;
    }

    document.getElementById('resultsArea').innerHTML = `
        <div class="py-12 text-center">
            <div class="w-12 h-12 bg-primary-100 rounded-xl flex items-center justify-center mx-auto mb-4">
                <svg class="w-6 h-6 text-primary-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
            </div>
            <div class="text-sm font-medium text-primary-900 mb-1">Something went wrong</div>
            <div class="text-sm text-primary-500 mb-4 max-w-sm mx-auto">${message}</div>
            ${additionalHelp}
        </div>
    `;
}

// Training mode functions
async function loadDatasets() {
    // This is for feature extraction - shows dataset paths from file system
    const listEl = document.getElementById('extractionDatasetList');
    if (!listEl) {
        // Fallback to old location if new element doesn't exist
        const oldListEl = document.getElementById('datasetList');
        if (oldListEl) {
            listEl = oldListEl;
        } else {
            console.error('Could not find extractionDatasetList element');
            return;
        }
    }

    listEl.innerHTML = '<div class="text-center py-12"><div class="spinner mx-auto"></div></div>';

    try {
        const response = await fetch(`${getApiUrl()}/training/datasets`);
        const data = await response.json();

        if (data.datasets && data.datasets.length > 0) {
            listEl.innerHTML = data.datasets.map(ds => `
                <div class="flex items-center p-4 bg-white border border-primary-200 rounded-xl mb-2.5 hover:border-primary-300 transition-colors">
                    <input type="checkbox" class="extraction-dataset-checkbox w-4 h-4 text-primary-600 rounded" value="${ds.path}" data-name="${ds.name}">
                    <div class="flex-1 ml-4">
                        <div class="text-sm font-medium text-primary-900">${ds.name}</div>
                        <div class="text-xs text-primary-500 mt-0.5">${ds.chat_files} CHAT files · ${ds.audio_files} audio files</div>
                    </div>
                </div>
            `).join('');
        } else {
            listEl.innerHTML = '<div class="text-center py-12 text-sm text-primary-400">No datasets found</div>';
        }
    } catch (error) {
        listEl.innerHTML = `<div class="text-sm text-red-500 p-4">Error loading datasets: ${error.message}</div>`;
    }
}

async function loadAvailableDatasetsForTraining() {
    // This is for training - shows datasets from CSV
    const listEl = document.getElementById('datasetList');
    const component = document.getElementById('trainingComponent').value;

    listEl.innerHTML = '<div class="text-center py-12"><div class="spinner mx-auto"></div></div>';

    try {
        const response = await fetch(`${getApiUrl()}/training/available-datasets/${component}`);
        const data = await response.json();

        if (data.csv_exists && data.datasets && data.datasets.length > 0) {
            listEl.innerHTML = `
                <div class="mb-4 p-3 bg-green-50 border border-green-200 rounded-xl">
                    <div class="text-xs text-green-700 flex items-center gap-1.5">
                        <svg class="w-3.5 h-3.5 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>
                        <strong>Features CSV found:</strong> ${data.total_samples} total samples from ${data.total_datasets} dataset(s)
                    </div>
                </div>
                ${data.datasets.map(ds => `
                    <div class="flex items-center p-4 bg-white border border-primary-200 rounded-xl mb-2.5 hover:border-primary-300 transition-colors">
                        <input type="checkbox" class="dataset-checkbox w-4 h-4 text-primary-600 rounded" value="${ds.name}" data-name="${ds.name}">
                        <div class="flex-1 ml-4">
                            <div class="text-sm font-medium text-primary-900">${ds.name}</div>
                            <div class="text-xs text-primary-500 mt-0.5">${ds.samples} samples available</div>
                        </div>
                    </div>
                `).join('')}
            `;
        } else {
            listEl.innerHTML = `
                <div class="mb-4 p-3 bg-primary-50 border border-primary-200 rounded-xl">
                    <div class="text-xs text-primary-700 font-medium mb-1">No features CSV found for ${component}</div>
                    <div class="text-xs text-primary-500">Extract features first using the Feature Extraction tab.</div>
                </div>
                <div class="text-center py-10 text-sm text-primary-400">
                    ${data.message || 'No datasets available'}
                </div>
            `;
        }
    } catch (error) {
        listEl.innerHTML = `<div class="text-sm text-red-500 p-4">Error: ${error.message}</div>`;
    }
}

async function extractFeatures() {
    // Get datasets from extraction checkboxes (file system datasets)
    const selectedDatasets = Array.from(document.querySelectorAll('.extraction-dataset-checkbox:checked')).map(cb => cb.value);

    if (selectedDatasets.length === 0) {
        alert('Please select at least one dataset for feature extraction');
        return;
    }

    const component = document.getElementById('extractionComponent').value;
    const maxSamples = document.getElementById('maxSamplesExtraction').value;

    const statusEl = document.getElementById('extractionStatus');
    const statusContent = document.getElementById('extractionStatusContent');
    statusEl.classList.remove('hidden');
    statusContent.innerHTML = '<div class="spinner mx-auto"></div><div class="text-center mt-3 text-sm text-primary-500">Extracting features...</div>';

    try {
        const requestBody = {
            dataset_paths: selectedDatasets,
            component: component,
            output_filename: `${component}_features.csv`
        };

        if (maxSamples && maxSamples.trim() !== '') {
            requestBody.max_samples_per_dataset = parseInt(maxSamples);
        }

        const response = await fetch(`${getApiUrl()}/training/extract-features`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        });

        const data = await response.json();

        if (response.ok) {
            statusContent.innerHTML = `
                <div class="flex items-center gap-2 text-green-700 text-sm font-medium mb-3">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>
                    Feature extraction complete
                </div>
                <div class="space-y-1 text-xs text-primary-600">
                    <div><span class="text-primary-400">Total samples:</span> <strong class="text-primary-700">${data.total_samples || data.new_samples}${data.new_samples ? ` (${data.new_samples} new)` : ''}</strong></div>
                    <div><span class="text-primary-400">Features:</span> <strong class="text-primary-700">${data.features_count}</strong></div>
                    <div><span class="text-primary-400">Output:</span> <span class="text-primary-600 font-mono text-xs">${data.output_file}</span></div>
                    ${data.datasets_updated ? `<div class="text-primary-500">Updated: ${data.datasets_updated.join(', ')}</div>` : ''}
                </div>
            `;

            // Reload available datasets for training after extraction
            setTimeout(() => {
                loadAvailableDatasetsForTraining();
            }, 1000);
        } else {
            statusContent.innerHTML = `<div class="text-sm text-red-600">${data.detail || 'Feature extraction failed'}</div>`;
        }
    } catch (error) {
        statusContent.innerHTML = `<div class="text-sm text-red-600">Error: ${error.message}</div>`;
    }
}

async function startTraining() {
    // Get dataset names (not paths) from checkboxes
    const selectedDatasets = Array.from(document.querySelectorAll('.dataset-checkbox:checked'))
        .map(cb => cb.getAttribute('data-name') || cb.value.split('/').pop() || cb.value);

    if (selectedDatasets.length === 0) {
        alert('Please select at least one dataset');
        return;
    }

    const component = document.getElementById('trainingComponent').value;

    // Get selected model types (only from the model types section)
    const modelTypeCheckboxes = document.querySelectorAll('#modelTypesContainer input[type="checkbox"][value]:checked');
    const selectedModels = Array.from(modelTypeCheckboxes).map(cb => cb.value);

    if (selectedModels.length === 0) {
        alert('Please select at least one model type');
        return;
    }

    // Validate models are allowed for this component
    if (componentModelTypes && componentModelTypes[component]) {
        const allowedModels = componentModelTypes[component];
        const invalidModels = selectedModels.filter(m => !allowedModels.includes(m));
        if (invalidModels.length > 0) {
            alert(`Invalid models for ${component}: ${invalidModels.join(', ')}. Allowed: ${allowedModels.join(', ')}`);
            return;
        }
    }
    const featureSelectionEnabled = document.getElementById('featureSelectionEnabled').checked;
    const nFeatures = parseInt(document.getElementById('nFeatures').value) || 30;
    const testSize = parseFloat(document.getElementById('testSize').value) / 100 || 0.2;
    const randomState = parseInt(document.getElementById('randomState').value) || 42;
    const enableAutoencoder = document.getElementById('enableAutoencoder').checked;
    const customHyperparams = getCustomHyperparameters();
    const classWeightRaw = document.getElementById('classWeightSelect')?.value || 'balanced';
    const classWeight = classWeightRaw === 'none' ? null : classWeightRaw;


    const statusEl = document.getElementById('trainingStatus');
    const statusContent = document.getElementById('trainingStatusContent');
    statusEl.classList.remove('hidden');
    statusContent.innerHTML = '<div class="spinner mx-auto"></div><div class="text-center mt-3 text-sm text-primary-500">Initializing training...</div>';

    try {
        const response = await fetch(`${getApiUrl()}/training/train`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                dataset_names: selectedDatasets,  // Changed from dataset_paths to dataset_names
                model_types: selectedModels,
                component: component,
                feature_selection: featureSelectionEnabled,
                n_features: featureSelectionEnabled ? nFeatures : null,
                test_size: testSize,
                random_state: randomState,
                enable_autoencoder: enableAutoencoder,
                class_weight: classWeight,
                custom_hyperparameters: customHyperparams
            })
        });

        const data = await response.json();

        if (response.ok) {
            // Start polling for progress
            pollTrainingProgress();
        } else {
            statusContent.innerHTML = `<div class="text-sm text-red-600">${data.detail || 'Failed to start training'}</div>`;
        }
    } catch (error) {
        statusContent.innerHTML = `<div class="text-sm text-red-600">Error: ${error.message}</div>`;
    }
}

// Toggle feature count input based on checkbox
// Component-specific model types (fetched from API)
let componentModelTypes = null;

async function fetchComponentModelTypes() {
    try {
        const response = await fetch(`${getApiUrl()}/training/component-models`);
        if (response.ok) {
            const data = await response.json();
            componentModelTypes = data.components;
        }
    } catch (error) {
        console.error('Failed to fetch component model types:', error);
    }
}

function updateModelCheckboxes() {
    const component = document.getElementById('trainingComponent').value;

    if (!componentModelTypes || !component) {
        return;
    }

    const allowedModels = componentModelTypes[component] || [];

    // Map of model values to their display info
    const modelInfo = {
        'random_forest': { label: 'Random Forest' },
        'xgboost': { label: 'XGBoost' },
        'gradient_boosting': { label: 'Gradient Boosting' },
        'adaboost': { label: 'AdaBoost' },
        'lightgbm': { label: 'LightGBM' },
        'svm': { label: 'SVM' }
    };

    // Find the model types container
    const modelTypesContainer = document.getElementById('modelTypesContainer');

    if (!modelTypesContainer) {
        console.error('Model types container not found');
        return;
    }

    // Clear existing checkboxes
    modelTypesContainer.innerHTML = '';

    // Add only allowed models for this component
    allowedModels.forEach((modelValue, index) => {
        const info = modelInfo[modelValue] || { label: modelValue };
        const isChecked = index === 0 || index === 1; // Check first two by default

        const label = document.createElement('label');
        label.className = 'flex items-center cursor-pointer p-4 bg-white border border-primary-200 rounded-xl hover:border-primary-300 transition-colors';
        label.innerHTML = `
            <input type="checkbox" value="${modelValue}" ${isChecked ? 'checked' : ''} class="w-4 h-4 text-primary-600 rounded">
            <span class="ml-3 text-sm text-primary-900">${info.label}</span>
        `;

        modelTypesContainer.appendChild(label);
    });
}

(function _initTrainingControls() {
    const featureSelectionCheckbox = document.getElementById('featureSelectionEnabled');
    const featureCountSection = document.getElementById('featureCountSection');

    if (featureSelectionCheckbox && featureCountSection) {
        featureSelectionCheckbox.addEventListener('change', () => {
            if (featureSelectionCheckbox.checked) {
                featureCountSection.style.opacity = '1';
                featureCountSection.style.pointerEvents = 'auto';
            } else {
                featureCountSection.style.opacity = '0.5';
                featureCountSection.style.pointerEvents = 'none';
            }
        });
    }

    // Reload available datasets when component changes
    const trainingComponent = document.getElementById('trainingComponent');
    if (trainingComponent) {
        trainingComponent.addEventListener('change', () => {
            loadAvailableDatasetsForTraining();
            updateModelCheckboxes(); // Update model checkboxes when component changes
        });

        // Load component model types on page load
        fetchComponentModelTypes().then(() => {
            updateModelCheckboxes(); // Initialize with default component
        });
    }
})();

let trainingPollInterval = null;

function pollTrainingProgress() {
    // Clear any existing interval
    if (trainingPollInterval) {
        clearInterval(trainingPollInterval);
    }

    const statusContent = document.getElementById('trainingStatusContent');

    // Poll every 2 seconds
    trainingPollInterval = setInterval(async () => {
        try {
            const response = await fetch(`${getApiUrl()}/training/status`);
            const status = await response.json();

            updateTrainingUI(status);

            // Stop polling if training is complete or errored
            if (status.status === 'completed' || status.status === 'error' || status.status === 'idle') {
                clearInterval(trainingPollInterval);
                trainingPollInterval = null;

                // Reload models list
                if (status.status === 'completed') {
                    setTimeout(() => {
                        loadAvailableModels();
                    }, 1000);
                }
            }
        } catch (error) {
            console.error('Error polling training status:', error);
        }
    }, 2000);

    // Initial update
    updateTrainingUI({ status: 'training', progress: 0, message: 'Starting...' });
}

function updateTrainingUI(status) {
    const statusContent = document.getElementById('trainingStatusContent');

    if (status.status === 'training') {
        const progressPercent = status.progress || 0;
        const currentModel = status.current_model ? ` · ${status.current_model}` : '';

        statusContent.innerHTML = `
            <div class="mb-3">
                <div class="flex justify-between text-xs text-primary-500 mb-1.5">
                    <span>${status.message}${currentModel}</span>
                    <span class="font-medium text-primary-700">${progressPercent}%</span>
                </div>
                <div class="w-full h-1.5 bg-primary-200 rounded-full overflow-hidden">
                    <div class="h-full bg-primary-700 transition-all duration-500" style="width: ${progressPercent}%"></div>
                </div>
            </div>
            <div class="text-xs text-primary-400">
                Training ${status.total_models || 0} model(s) for ${status.component || 'component'}...
            </div>
        `;
    } else if (status.status === 'completed') {
        let resultsHtml = '';
        if (status.results && Object.keys(status.results).length > 0) {
            resultsHtml = '<div class="mt-3 space-y-2">';
            for (const [model, metrics] of Object.entries(status.results)) {
                resultsHtml += `
                    <div class="flex items-center justify-between p-3 bg-primary-50 border border-primary-200 rounded-xl">
                        <span class="text-sm font-medium text-primary-900">${model}</span>
                        <div class="flex gap-4 text-xs text-primary-500">
                            <span>Acc: <strong class="text-primary-700">${(metrics.accuracy * 100).toFixed(1)}%</strong></span>
                            <span>F1: <strong class="text-primary-700">${(metrics.f1_score * 100).toFixed(1)}%</strong></span>
                        </div>
                    </div>
                `;
            }
            resultsHtml += '</div>';
        }

        statusContent.innerHTML = `
            <div class="flex items-center gap-2 text-green-700 text-sm font-medium mb-2">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>
                ${status.message}
            </div>
            ${resultsHtml}
        `;
    } else if (status.status === 'error') {
        let errorDetails = '';
        const errorMsg = status.error || status.message;

        // Parse common errors and provide helpful solutions
        if (errorMsg.includes('missing diagnosis') || errorMsg.includes('Insufficient samples')) {
            errorDetails = `
                <div class="mt-3 p-3 bg-primary-50 border border-primary-200 rounded-xl text-xs">
                    <div class="font-medium text-primary-700 mb-2">Possible solutions</div>
                    <ul class="space-y-1 text-primary-600">
                        <li>· Some CHAT files may be missing diagnosis labels</li>
                        <li>· Try selecting different datasets</li>
                        <li>· Ensure datasets have proper CHAT format with diagnosis codes</li>
                        <li>· Check that files contain participant diagnosis information</li>
                    </ul>
                </div>
            `;
        } else if (errorMsg.includes('No features extracted')) {
            errorDetails = `
                <div class="mt-3 p-3 bg-primary-50 border border-primary-200 rounded-xl text-xs">
                    <div class="font-medium text-primary-700 mb-2">Possible solutions</div>
                    <ul class="space-y-1 text-primary-600">
                        <li>· Check that selected datasets contain .cha files</li>
                        <li>· Verify CHAT files are properly formatted</li>
                        <li>· Try extracting features first to diagnose issues</li>
                    </ul>
                </div>
            `;
        }

        statusContent.innerHTML = `
            <div class="flex items-center gap-2 text-red-600 text-sm font-medium mb-2">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
                Training failed
            </div>
            <div class="text-xs mt-1 p-3 bg-red-50 border border-red-100 rounded-xl text-red-700">
                ${errorMsg}
            </div>
            ${errorDetails}
        `;
    } else {
        statusContent.innerHTML = `<div class="text-sm text-primary-500">${status.message}</div>`;
    }
}


async function loadAvailableModels() {
    const container = document.getElementById('availableModelsContainer');
    if (!container) return;

    container.innerHTML = '<div class="text-center py-12"><div class="spinner mx-auto"></div></div>';

    try {
        const response = await fetch(`${getApiUrl()}/models`);
        const data = await response.json();

        if (data.models && data.models.length > 0) {
            // Group models by component
            const modelsByComponent = {};
            for (const model of data.models) {
                const component = model.name.split('_')[0] + '_' + model.name.split('_')[1] || 'pragmatic_conversational';
                if (!modelsByComponent[component]) {
                    modelsByComponent[component] = [];
                }
                modelsByComponent[component].push(model);
            }

            let modelsHtml = '';

            // Display models grouped by component in table format
            for (const [component, models] of Object.entries(modelsByComponent)) {
                const componentNames = {
                    'pragmatic_conversational': 'Pragmatic & Conversational',
                    'acoustic_prosodic': 'Acoustic & Prosodic',
                    'syntactic_semantic': 'Syntactic & Semantic'
                };
                const componentColors = {
                    'pragmatic_conversational': 'green',
                    'acoustic_prosodic': 'blue',
                    'syntactic_semantic': 'purple'
                };

                const componentName = componentNames[component] || component;
                const color = componentColors[component] || 'gray';

                modelsHtml += `
                    <div class="mb-8">
                        <div class="flex items-center gap-3 mb-4">
                            <h3 class="text-base font-medium text-primary-900">${componentName}</h3>
                            <span class="px-2 py-0.5 bg-primary-100 text-primary-600 text-xs rounded-full">${models.length} model${models.length > 1 ? 's' : ''}</span>
                        </div>
                        <div class="bg-white rounded-xl overflow-hidden border border-primary-200">
                            <div class="overflow-x-auto">
                                <table class="w-full">
                                    <thead class="bg-primary-50 border-b border-primary-200">
                                        <tr>
                                            <th class="px-5 py-3 text-left text-xs font-medium text-primary-500 uppercase tracking-wide">Model Type</th>
                                            <th class="px-5 py-3 text-center text-xs font-medium text-primary-500 uppercase tracking-wide">Accuracy</th>
                                            <th class="px-5 py-3 text-center text-xs font-medium text-primary-500 uppercase tracking-wide">F1 Score</th>
                                            <th class="px-5 py-3 text-center text-xs font-medium text-primary-500 uppercase tracking-wide">Precision</th>
                                            <th class="px-5 py-3 text-center text-xs font-medium text-primary-500 uppercase tracking-wide">Recall</th>
                                            <th class="px-5 py-3 text-center text-xs font-medium text-primary-500 uppercase tracking-wide">ROC-AUC</th>
                                            <th class="px-5 py-3 text-center text-xs font-medium text-primary-500 uppercase tracking-wide">Features</th>
                                            <th class="px-5 py-3 text-center text-xs font-medium text-primary-500 uppercase tracking-wide">Samples</th>
                                            <th class="px-5 py-3 text-center text-xs font-medium text-primary-500 uppercase tracking-wide">Created</th>
                                            <th class="px-5 py-3 text-center text-xs font-medium text-primary-500 uppercase tracking-wide">Actions</th>
                                        </tr>
                                    </thead>
                                    <tbody class="divide-y divide-primary-100">
                `;

                for (const model of models) {
                    const isBest = model.name === data.best_model;
                    const accuracy = (model.accuracy * 100).toFixed(1);
                    const f1 = (model.f1_score * 100).toFixed(1);
                    const precision = (model.precision * 100).toFixed(1);
                    const recall = (model.recall * 100).toFixed(1);
                    const rocAuc = model.roc_auc ? (model.roc_auc * 100).toFixed(1) : 'N/A';
                    const date = new Date(model.created_at).toLocaleDateString();
                    const time = new Date(model.created_at).toLocaleTimeString();

                    modelsHtml += `
                        <tr class="hover:bg-primary-50 transition-colors ${isBest ? 'bg-primary-50' : ''}">
                            <td class="px-5 py-3.5">
                                <div class="flex items-center gap-2">
                                    <span class="text-sm font-medium text-primary-900">${model.type}</span>
                                    ${isBest ? '<span class="px-2 py-0.5 bg-primary-900 text-white text-xs rounded-full">Best</span>' : ''}
                                </div>
                            </td>
                            <td class="px-5 py-3.5 text-center text-sm text-primary-700">${accuracy}%</td>
                            <td class="px-5 py-3.5 text-center text-sm text-primary-700">${f1}%</td>
                            <td class="px-5 py-3.5 text-center text-sm text-primary-700">${precision}%</td>
                            <td class="px-5 py-3.5 text-center text-sm text-primary-700">${recall}%</td>
                            <td class="px-5 py-3.5 text-center text-sm text-primary-700">${rocAuc}${rocAuc !== 'N/A' ? '%' : ''}</td>
                            <td class="px-5 py-3.5 text-center text-sm text-primary-600">${model.n_features}</td>
                            <td class="px-5 py-3.5 text-center text-sm text-primary-600">${model.training_samples}</td>
                            <td class="px-5 py-3.5 text-center text-xs text-primary-500">${date}<br>${time}</td>
                            <td class="px-5 py-3.5 text-center">
                                <div class="flex items-center justify-center gap-2">
                                    <button class="px-3 py-1.5 bg-primary-900 text-white rounded-lg hover:bg-primary-800 transition-colors text-xs font-medium" onclick='showModelDetails(${JSON.stringify(model)})'>
                                        View
                                    </button>
                                    <button class="px-3 py-1.5 text-red-500 border border-red-200 hover:bg-red-50 rounded-lg transition-colors text-xs" onclick="deleteModel('${model.name}')">
                                        Delete
                                    </button>
                                </div>
                            </td>
                        </tr>
                    `;
                }

                modelsHtml += `
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                `;
            }

            container.innerHTML = modelsHtml;
        } else {
            container.innerHTML = `
                <div class="text-center py-16">
                    <div class="w-12 h-12 bg-primary-100 rounded-xl flex items-center justify-center mx-auto mb-4">
                        <svg class="w-6 h-6 text-primary-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
                        </svg>
                    </div>
                    <div class="text-sm font-medium text-primary-700 mb-1">No models trained yet</div>
                    <div class="text-xs text-primary-400">Train your first model to get started</div>
                </div>
            `;
        }
    } catch (error) {
        container.innerHTML = `<div class="text-sm text-red-500 p-4">Error loading models: ${error.message}</div>`;
    }
}

async function deleteModel(modelName) {
    if (!confirm(`Are you sure you want to delete the model "${modelName}"? This action cannot be undone.`)) {
        return;
    }

    try {
        const response = await fetch(`${getApiUrl()}/models/${modelName}`, {
            method: 'DELETE'
        });

        const data = await response.json();

        if (response.ok) {
            alert(`Model "${modelName}" deleted successfully`);
            loadAvailableModels();
        } else {
            alert(`Error deleting model: ${data.detail || 'Unknown error'}`);
        }
    } catch (error) {
        alert(`Error deleting model: ${error.message}`);
    }
}


// Model Details Modal Functions
function showModelDetails(model) {
    const modal = document.getElementById('modelDetailsModal');
    const content = document.getElementById('modalContent');

    // Render confusion matrix
    const confusionMatrixHtml = renderConfusionMatrix(model.confusion_matrix);

    // Render metrics
    const accuracy = (model.accuracy * 100).toFixed(2);
    const f1 = (model.f1_score * 100).toFixed(2);
    const precision = (model.precision * 100).toFixed(2);
    const recall = (model.recall * 100).toFixed(2);
    const rocAuc = model.roc_auc ? (model.roc_auc * 100).toFixed(2) : 'N/A';
    const matthews = model.matthews_corr ? model.matthews_corr.toFixed(4) : 'N/A';

    content.innerHTML = `
        <div class="space-y-7">
            <!-- Model Info -->
            <div class="bg-primary-50 border border-primary-200 rounded-xl p-5">
                <div class="flex items-center gap-3 mb-4">
                    <h3 class="text-base font-medium text-primary-900">${model.type} Model</h3>
                </div>
                <div class="grid grid-cols-2 gap-3 text-xs">
                    <div><span class="text-primary-400">Component:</span> <span class="font-medium text-primary-700">${model.component || 'pragmatic_conversational'}</span></div>
                    <div><span class="text-primary-400">Features:</span> <span class="font-medium text-primary-700">${model.n_features}</span></div>
                    <div><span class="text-primary-400">Training Samples:</span> <span class="font-medium text-primary-700">${model.training_samples}</span></div>
                    <div><span class="text-primary-400">Created:</span> <span class="font-medium text-primary-700">${new Date(model.created_at).toLocaleString()}</span></div>
                </div>
            </div>
            
            <!-- Performance Metrics -->
            <div>
                <p class="text-xs font-medium text-primary-400 uppercase tracking-wide mb-3">Performance Metrics</p>
                <div class="grid grid-cols-3 md:grid-cols-6 gap-2.5">
                    <div class="bg-primary-50 border border-primary-200 rounded-xl p-3 text-center">
                        <div class="text-lg font-semibold text-primary-900">${accuracy}%</div>
                        <div class="text-xs text-primary-400 mt-1">Accuracy</div>
                    </div>
                    <div class="bg-primary-50 border border-primary-200 rounded-xl p-3 text-center">
                        <div class="text-lg font-semibold text-primary-900">${f1}%</div>
                        <div class="text-xs text-primary-400 mt-1">F1 Score</div>
                    </div>
                    <div class="bg-primary-50 border border-primary-200 rounded-xl p-3 text-center">
                        <div class="text-lg font-semibold text-primary-900">${precision}%</div>
                        <div class="text-xs text-primary-400 mt-1">Precision</div>
                    </div>
                    <div class="bg-primary-50 border border-primary-200 rounded-xl p-3 text-center">
                        <div class="text-lg font-semibold text-primary-900">${recall}%</div>
                        <div class="text-xs text-primary-400 mt-1">Recall</div>
                    </div>
                    <div class="bg-primary-50 border border-primary-200 rounded-xl p-3 text-center">
                        <div class="text-lg font-semibold text-primary-900">${rocAuc}${rocAuc !== 'N/A' ? '%' : ''}</div>
                        <div class="text-xs text-primary-400 mt-1">ROC-AUC</div>
                    </div>
                    <div class="bg-primary-50 border border-primary-200 rounded-xl p-3 text-center">
                        <div class="text-lg font-semibold text-primary-900">${matthews}</div>
                        <div class="text-xs text-primary-400 mt-1">Matthews</div>
                    </div>
                </div>
            </div>
            
            <!-- Confusion Matrix -->
            <div>
                <p class="text-xs font-medium text-primary-400 uppercase tracking-wide mb-3">Confusion Matrix</p>
                ${confusionMatrixHtml}
            </div>
            ${model.shap ? `
            <!-- SHAP Explanations -->
                    <div>
                        <p class="text-xs font-medium text-primary-400 uppercase tracking-wide mb-3">Global SHAP Explanations</p>
                        <p class="text-sm text-primary-500 mb-5">Feature importance across the full training dataset</p>
                        <div class="grid md:grid-cols-2 gap-5">
                            <div class="bg-white border border-primary-200 rounded-xl p-5">
                                <h4 class="text-sm font-medium text-primary-700 mb-3">Beeswarm</h4>
                                <img
                                    src="${getApiUrl()}${model.shap.beeswarm}?t=${Date.now()}"
                                    class="w-full rounded-lg border border-primary-100"
                                />
                            </div>
                            <div class="bg-white border border-primary-200 rounded-xl p-5">
                                <h4 class="text-sm font-medium text-primary-700 mb-3">Mean |SHAP| Importance</h4>
                                <img
                                    src="${getApiUrl()}${model.shap.bar}?t=${Date.now()}"
                                    class="w-full rounded-lg border border-primary-100"
                                />
                            </div>
                        </div>
                    </div>
                ` : `
                <!-- No SHAP -->
                <div class="bg-primary-50 border border-primary-200 rounded-xl p-5">
                    <h3 class="text-sm font-medium text-primary-700 mb-1.5">SHAP Explanations</h3>
                    <p class="text-xs text-primary-500">
                        Not available for this model. This may be due to model type limitations (e.g., SVM) or skipped training.
                    </p>
                </div>
                `}
        </div>
    `;

    modal.classList.remove('hidden');
    modal.classList.add('flex');
}

function closeModelDetails(event) {
    const modal = document.getElementById('modelDetailsModal');
    if (!event || event.target === modal) {
        modal.classList.add('hidden');
        modal.classList.remove('flex');
    }
}

// Hyperparameter Management
const DEFAULT_HYPERPARAMS = {
    'random_forest': {
        'n_estimators': {
            value: 100, type: 'number', min: 10, max: 500,
            description: 'Number of decision trees in the forest',
            range: 'Typical: 50-300',
            effect: 'Higher = better performance but slower training. Too high can overfit.'
        },
        'max_depth': {
            value: 10, type: 'number', min: 2, max: 50,
            description: 'Maximum depth of each decision tree',
            range: 'Typical: 5-20',
            effect: 'Higher = more complex patterns, but can overfit. Lower = simpler, faster.'
        },
        'min_samples_split': {
            value: 5, type: 'number', min: 2, max: 20,
            description: 'Minimum samples required to split a node',
            range: 'Typical: 2-10',
            effect: 'Higher = prevents overfitting, simpler trees. Lower = more detailed splits.'
        },
        'min_samples_leaf': {
            value: 2, type: 'number', min: 1, max: 10,
            description: 'Minimum samples required in a leaf node',
            range: 'Typical: 1-5',
            effect: 'Higher = smoother predictions, less overfitting. Lower = more granular.'
        }
    },
    'xgboost': {
        'n_estimators': {
            value: 100, type: 'number', min: 10, max: 500,
            description: 'Number of gradient boosting rounds',
            range: 'Typical: 50-300',
            effect: 'Higher = better performance but slower. Use with lower learning_rate.'
        },
        'max_depth': {
            value: 6, type: 'number', min: 2, max: 15,
            description: 'Maximum depth of each tree',
            range: 'Typical: 3-10',
            effect: 'Higher = captures complex patterns, risk of overfitting. Lower = faster, simpler.'
        },
        'learning_rate': {
            value: 0.1, type: 'number', min: 0.001, max: 1, step: 0.001,
            description: 'Step size shrinkage for each boosting step',
            range: 'Typical: 0.01-0.3',
            effect: 'Lower = more conservative, needs more trees. Higher = faster but may overfit.'
        },
        'subsample': {
            value: 0.8, type: 'number', min: 0.1, max: 1, step: 0.1,
            description: 'Fraction of samples used for each tree',
            range: 'Typical: 0.6-1.0',
            effect: 'Lower = reduces overfitting, adds randomness. Higher = uses more data per tree.'
        }
    },
    'lightgbm': {
        'n_estimators': {
            value: 100, type: 'number', min: 10, max: 500,
            description: 'Number of boosting iterations',
            range: 'Typical: 50-300',
            effect: 'Higher = better performance but slower. LightGBM is faster than XGBoost.'
        },
        'max_depth': {
            value: 6, type: 'number', min: 2, max: 15,
            description: 'Maximum tree depth',
            range: 'Typical: 3-10',
            effect: 'Higher = more complex patterns. Lower = faster training, less overfitting.'
        },
        'learning_rate': {
            value: 0.1, type: 'number', min: 0.001, max: 1, step: 0.001,
            description: 'Boosting learning rate',
            range: 'Typical: 0.01-0.3',
            effect: 'Lower = more stable, needs more trees. Higher = faster convergence.'
        },
        'subsample': {
            value: 0.8, type: 'number', min: 0.1, max: 1, step: 0.1,
            description: 'Fraction of data to use for training',
            range: 'Typical: 0.6-1.0',
            effect: 'Lower = prevents overfitting. Higher = uses more training data.'
        }
    },
    'gradient_boosting': {
        'n_estimators': {
            value: 100, type: 'number', min: 10, max: 500,
            description: 'Number of boosting stages',
            range: 'Typical: 50-300',
            effect: 'Higher = better fit but slower. Balance with learning_rate.'
        },
        'learning_rate': {
            value: 0.1, type: 'number', min: 0.001, max: 1, step: 0.001,
            description: 'Learning rate for each boosting stage',
            range: 'Typical: 0.01-0.3',
            effect: 'Lower = more conservative, requires more trees. Higher = faster but may overfit.'
        },
        'max_depth': {
            value: 5, type: 'number', min: 2, max: 15,
            description: 'Maximum depth of individual trees',
            range: 'Typical: 3-8',
            effect: 'Higher = captures complex interactions. Lower = simpler, faster, less overfitting.'
        },
        'min_samples_split': {
            value: 5, type: 'number', min: 2, max: 20,
            description: 'Minimum samples to split a node',
            range: 'Typical: 2-10',
            effect: 'Higher = prevents overfitting. Lower = more detailed splits.'
        }
    },
    'adaboost': {
        'n_estimators': {
            value: 100, type: 'number', min: 10, max: 500,
            description: 'Number of weak learners (estimators)',
            range: 'Typical: 50-200',
            effect: 'Higher = better performance but slower. Too high can overfit.'
        },
        'learning_rate': {
            value: 1.0, type: 'number', min: 0.01, max: 2, step: 0.01,
            description: 'Weight applied to each classifier',
            range: 'Typical: 0.5-2.0',
            effect: 'Lower = more conservative updates. Higher = faster adaptation, risk of overfitting.'
        }
    },
    // Note: logistic regression is no longer offered as a selectable model type
    'svm': {
        'C': {
            value: 1.0, type: 'number', min: 0.001, max: 100, step: 0.001,
            description: 'Regularization parameter (penalty for misclassification)',
            range: 'Typical: 0.1-10',
            effect: 'Higher = harder margin, less tolerance for errors. Lower = softer margin, more tolerance.'
        },
        'kernel': {
            value: 'rbf', type: 'select', options: ['rbf', 'linear', 'poly', 'sigmoid'],
            description: 'Kernel function type for non-linear classification',
            range: 'Options: rbf, linear, poly, sigmoid',
            effect: 'rbf=non-linear (default), linear=fast but limited, poly=polynomial, sigmoid=neural network-like.'
        },
        'gamma': {
            value: 'scale', type: 'select', options: ['scale', 'auto'],
            description: 'Kernel coefficient for rbf, poly, sigmoid',
            range: 'Options: scale (default), auto',
            effect: 'scale=1/(n_features*X.var()), auto=1/n_features. Lower = smoother decision boundary.'
        }
    }
};

function toggleHyperparameters() {
    const section = document.getElementById('hyperparamSection');
    const chevron = document.getElementById('hyperparamChevron');
    const isHidden = section.classList.contains('hidden');

    if (isHidden) {
        section.classList.remove('hidden');
        chevron.style.transform = 'rotate(180deg)';
        updateHyperparamControls();
    } else {
        section.classList.add('hidden');
        chevron.style.transform = 'rotate(0deg)';
    }
}

function updateHyperparamControls() {
    const selectedModels = Array.from(document.querySelectorAll('input[type="checkbox"][value]:checked'))
        .filter(cb => ['random_forest', 'xgboost', 'lightgbm', 'svm', 'gradient_boosting', 'adaboost'].includes(cb.value))
        .map(cb => cb.value);

    const container = document.getElementById('hyperparamControls');

    if (selectedModels.length === 0) {
        container.innerHTML = '<p class="text-primary-500 text-center py-4">Select at least one model type above</p>';
        return;
    }

    let html = '';
    selectedModels.forEach(modelType => {
        const params = DEFAULT_HYPERPARAMS[modelType];
        const modelNames = {
            'random_forest': 'Random Forest',
            'xgboost': 'XGBoost',
            'lightgbm': 'LightGBM',
            'gradient_boosting': 'Gradient Boosting',
            'adaboost': 'AdaBoost',
            'svm': 'SVM'
        };

        html += `
            <div class="border border-primary-200 rounded-xl p-5 bg-white">
                <h4 class="text-lg font-semibold text-primary-900 mb-4">${modelNames[modelType]}</h4>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        `;

        for (const [paramName, paramConfig] of Object.entries(params)) {
            const inputId = `hyperparam_${modelType}_${paramName}`;
            const tooltipId = `tooltip_${modelType}_${paramName}`;
            html += `
                <div class="space-y-1">
                    <label class="flex items-center gap-1.5 text-sm font-medium text-primary-900">
                        ${paramName}
                        <div class="group relative">
                            <svg class="w-4 h-4 text-primary-400 hover:text-primary-600 cursor-help transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                            </svg>
                            <div id="${tooltipId}" class="hidden group-hover:block absolute z-50 w-80 p-3 mt-2 bg-primary-900 text-white text-xs rounded-lg shadow-xl left-0 top-full mb-1 pointer-events-none">
                                <div class="font-semibold mb-2 text-primary-50">${paramConfig.description}</div>
                                <div class="text-primary-200 mb-1.5"><span class="font-medium">Range:</span> ${paramConfig.range}</div>
                                <div class="text-primary-200"><span class="font-medium">Effect:</span> ${paramConfig.effect}</div>
                                <div class="absolute -top-1 left-4 w-2 h-2 bg-primary-900 rotate-45"></div>
                            </div>
                        </div>
                    </label>
            `;

            if (paramConfig.type === 'select') {
                html += `<select id="${inputId}" class="w-full px-3 py-2 bg-primary-50 border border-primary-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent">`;
                paramConfig.options.forEach(opt => {
                    html += `<option value="${opt}" ${opt === paramConfig.value ? 'selected' : ''}>${opt}</option>`;
                });
                html += `</select>`;
            } else {
                const step = paramConfig.step || 1;
                html += `<input type="number" id="${inputId}" value="${paramConfig.value}" 
                    min="${paramConfig.min}" max="${paramConfig.max}" step="${step}"
                    class="w-full px-3 py-2 bg-primary-50 border border-primary-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent">`;
            }

            html += `
                    <div class="text-xs text-primary-600 space-y-0.5">
                        <div class="font-medium">${paramConfig.description}</div>
                        <div class="text-primary-500">${paramConfig.range} • ${paramConfig.effect}</div>
                    </div>
                </div>
            `;
        }

        html += `
                </div>
            </div>
        `;
    });

    container.innerHTML = html;
}

function getCustomHyperparameters() {
    const selectedModels = Array.from(document.querySelectorAll('input[type="checkbox"][value]:checked'))
        .filter(cb => ['random_forest', 'xgboost', 'lightgbm', 'svm', 'gradient_boosting', 'adaboost'].includes(cb.value))
        .map(cb => cb.value);

    const customParams = {};

    selectedModels.forEach(modelType => {
        const params = DEFAULT_HYPERPARAMS[modelType];
        customParams[modelType] = {};

        for (const paramName of Object.keys(params)) {
            const inputId = `hyperparam_${modelType}_${paramName}`;
            const input = document.getElementById(inputId);
            if (input) {
                const value = input.type === 'number' ? parseFloat(input.value) : input.value;
                customParams[modelType][paramName] = value;
            }
        }
    });

    return customParams;
}

// Update model checkboxes to refresh hyperparam controls
(function _initModelCheckboxListeners() {
    const modelCheckboxes = document.querySelectorAll('input[type="checkbox"][value]');
    modelCheckboxes.forEach(cb => {
        if (['random_forest', 'xgboost', 'lightgbm', 'svm', 'gradient_boosting', 'adaboost'].includes(cb.value)) {
            cb.addEventListener('change', () => {
                const section = document.getElementById('hyperparamSection');
                if (!section.classList.contains('hidden')) {
                    updateHyperparamControls();
                }
            });
        }
    });
})();

function renderConfusionMatrix(matrix) {
    if (!matrix || matrix.length === 0) {
        return '<div class="text-primary-500 text-center py-8">Confusion matrix not available</div>';
    }

    const labels = ['TD (Negative)', 'ASD (Positive)'];
    const total = matrix.flat().reduce((a, b) => a + b, 0);

    // Calculate percentages and colors
    const getColor = (value, maxValue) => {
        const intensity = Math.round((value / maxValue) * 255);
        return `rgb(${255 - intensity}, ${255 - intensity * 0.5}, 255)`;
    };

    const maxValue = Math.max(...matrix.flat());

    let html = `
        <div class="bg-white rounded-xl p-6 shadow-lg">
            <div class="overflow-x-auto">
                <table class="w-full border-collapse">
                    <thead>
                        <tr>
                            <th class="p-3"></th>
                            <th class="p-3"></th>
                            <th class="p-3 text-center font-medium text-primary-900" colspan="2">Predicted</th>
                        </tr>
                        <tr>
                            <th class="p-3"></th>
                            <th class="p-3"></th>
                            ${labels.map(label => `<th class="p-3 text-center text-sm font-medium text-primary-700">${label}</th>`).join('')}
                        </tr>
                    </thead>
                    <tbody>
    `;

    matrix.forEach((row, i) => {
        html += `<tr>`;
        // Add "Actual" label only for first row
        if (i === 0) {
            html += `<th rowspan="${matrix.length}" class="p-3 text-center font-medium text-primary-900 align-middle border-r border-primary-300" style="vertical-align: middle;">
                <div style="writing-mode: vertical-rl; transform: rotate(180deg); white-space: nowrap;">Actual</div>
            </th>`;
        }
        html += `<th class="p-3 text-left text-sm font-medium text-primary-700 align-middle">${labels[i]}</th>`;

        row.forEach((value, j) => {
            const percentage = ((value / total) * 100).toFixed(1);
            const bgColor = getColor(value, maxValue);
            const borderClass = j < row.length - 1 ? 'border-r border-primary-200' : '';
            html += `
                <td class="p-6 text-center border border-primary-200 align-middle ${borderClass}" style="background-color: ${bgColor}">
                    <div class="text-2xl font-bold text-primary-900">${value}</div>
                    <div class="text-xs text-primary-600 mt-1">${percentage}%</div>
                </td>
            `;
        });

        html += `</tr>`;
    });

    html += `
                    </tbody>
                </table>
            </div>
            <div class="mt-6 grid grid-cols-2 gap-4 text-sm">
                <div class="bg-green-50 p-4 rounded-lg">
                    <div class="font-medium text-green-900">True Negatives (TN)</div>
                    <div class="text-green-700">Correctly predicted TD: ${matrix[0][0]}</div>
                </div>
                <div class="bg-red-50 p-4 rounded-lg">
                    <div class="font-medium text-red-900">False Positives (FP)</div>
                    <div class="text-red-700">Wrongly predicted ASD: ${matrix[0][1]}</div>
                </div>
                <div class="bg-orange-50 p-4 rounded-lg">
                    <div class="font-medium text-orange-900">False Negatives (FN)</div>
                    <div class="text-orange-700">Missed ASD cases: ${matrix[1][0]}</div>
                </div>
                <div class="bg-blue-50 p-4 rounded-lg">
                    <div class="font-medium text-blue-900">True Positives (TP)</div>
                    <div class="text-blue-700">Correctly predicted ASD: ${matrix[1][1]}</div>
                </div>
            </div>
        </div>
    `;

    return html;
}

function generateWhatIfText(counterfactual) {
    if (!counterfactual.top_changes || counterfactual.top_changes.length === 0) {
        return "No meaningful counterfactual changes could be generated.";
    }

    const top = counterfactual.top_changes[0];

    return `
        If the <strong>${top.feature.replaceAll("_", " ")}</strong>
        were adjusted from <strong>${top.from.toFixed(2)}</strong>
        to <strong>${top.to.toFixed(2)}</strong>,
        the model’s prediction would change from
        <strong>ASD</strong> to <strong>TD</strong>.
`;
}

function renderCounterfactual(counterfactual, componentName = null, append = false) {
    if (!counterfactual) return;

    const section = document.getElementById("counterfactualSection");
    section.classList.remove("hidden");

    if (!append) {
        document.getElementById("whatIfBox").innerHTML =
            generateWhatIfText(counterfactual);

        document.getElementById("cfFlipped").textContent =
            counterfactual.prediction_flipped ? "Yes" : "No";

        document.getElementById("cfL2").textContent =
            counterfactual.l2_change.toFixed(3);

        document.getElementById("cfTotal").textContent =
            counterfactual.total_features_changed;
    }

    const tbody = document.getElementById("cfTableBody");

    if (!append) {
        tbody.innerHTML = "";
    }

    const changes = counterfactual.top_changes;
    const rowspan = changes.length;

    changes.forEach((change, index) => {

        const row = document.createElement("tr");
        row.className = "border-b last:border-b-0";

        let componentCell = "";

        // Only create component cell for first row
        // Color mapping for components
        const componentColors = {
            syntactic_semantic: "bg-blue-100 text-blue-800",
            pragmatic_conversational: "bg-green-100 text-green-800",
            acoustic_prosodic: "bg-purple-100 text-purple-800"
        };

        // Get color for current component
        let colorClass = componentColors[componentName] || "bg-gray-100 text-gray-800";

        if (componentName && index === 0) {
            componentCell = `
                <td rowspan="${rowspan}" 
                    class="align-middle text-center px-4">
                    
                    <div class="${colorClass} text-xs font-semibold px-3 py-2 rounded-lg inline-block">
                        ${componentName.replace(/_/g," ").toUpperCase()}
                    </div>
                </td>
            `;
        }

        row.innerHTML = `
            ${componentCell}

            <td class="py-2 font-medium">
                ${change.feature.replaceAll("_", " ")}
            </td>

            <td class="py-2">
                ${change.from.toFixed(3)}
            </td>

            <td class="py-2">
                ${change.to.toFixed(3)}
            </td>

            <td class="py-2 ${change.change > 0 ? "text-green-600" : "text-red-600"}">
                ${change.change > 0 ? "+" : ""}
                ${change.change.toFixed(3)}
            </td>
        `;

        tbody.appendChild(row);
    });
}

// Load models for prediction dropdowns
async function loadModelsForPrediction() {
    const selects = {
        'audioModelSelect': ['pragmatic_conversational', 'acoustic_prosodic'], // Audio can use pragmatic or acoustic
        'textModelSelect': ['pragmatic_conversational', 'syntactic_semantic'], // Text can use pragmatic or semantic
        'chaModelSelect': ['pragmatic_conversational', 'syntactic_semantic']  // CHAT can use pragmatic or semantic
    };

    try {
        const response = await fetch(`${getApiUrl()}/models`);
        const data = await response.json();

        if (data.models && data.models.length > 0) {
            // Group models by component for better organization
            const modelsByComponent = {};
            for (const model of data.models) {
                const component = model.component || (model.name.split('_').slice(0, 2).join('_'));
                if (!modelsByComponent[component]) {
                    modelsByComponent[component] = [];
                }
                modelsByComponent[component].push(model);
            }

            // Update each select dropdown with compatible models only
            for (const [selectId, compatibleComponents] of Object.entries(selects)) {
                const select = document.getElementById(selectId);
                if (!select) continue;

                // Keep the "Best Model" option
                select.innerHTML = '<option value="">Best Model (Auto)</option>';

                // Add models grouped by component (only compatible ones)
                for (const [component, models] of Object.entries(modelsByComponent)) {
                    // Only add if component is compatible with this input type
                    if (!compatibleComponents.includes(component)) {
                        continue;
                    }

                    const componentNames = {
                        'pragmatic_conversational': 'Pragmatic & Conversational',
                        'acoustic_prosodic': 'Acoustic & Prosodic',
                        'syntactic_semantic': 'Syntactic & Semantic'
                    };
                    const componentName = componentNames[component] || component;

                    // Add optgroup
                    const optgroup = document.createElement('optgroup');
                    optgroup.label = componentName;

                    models.forEach(model => {
                        const option = document.createElement('option');
                        option.value = model.name;
                        const isBest = model.name === data.best_model;
                        option.textContent = `${model.type}${isBest ? ' (Best)' : ''} - ${(model.f1_score * 100).toFixed(1)}% F1`;
                        optgroup.appendChild(option);
                    });

                    select.appendChild(optgroup);
                }
            }
        }
    } catch (error) {
        console.error('Error loading models:', error);
    }
}

// ==============================
// Annotated Transcript Rendering
// ==============================

// Feature type categories (aligned with ANNOTATION_CATEGORIES)
const FEATURE_CATEGORIES = {
    'Turn-Taking': {
        types: ['turn_start', 'turn_end', 'overlap', 'interruption', 'long_pause', 'response_latency'],
        color: '#3B5B8B'
    },
    'Pragmatic Markers': {
        types: ['echolalia', 'pronoun_reversal', 'stereotyped_phrase', 'social_greeting', 'question'],
        color: '#8B2B35'
    },
    'Conversational': {
        types: ['topic_shift', 'topic_maintenance', 'repair_initiation', 'repair_completion', 'clarification_request'],
        color: '#2A6040'
    },
    'Linguistic': {
        types: ['complex_sentence', 'simple_sentence', 'filled_pause', 'discourse_marker'],
        color: '#4A3080'
    },
    'Syntactic & Semantic': {
        types: ['complex_syntax', 'grammatical_error', 'low_semantic_density', 'semantic_mismatch'],
        color: '#00796B'
    },
    'General': {
        types: ['feature_region'],
        color: '#5A6470'
    }
};

// Muted, professional color mapping for annotation types
// Organised by category: Turn-taking (slate), Pragmatic (crimson), Conversational (forest), Linguistic (indigo), General (grey)
const ANNOTATION_COLORS = {
    // Turn-taking — slate blue
    'turn_start': '#3B5B8B',
    'turn_end': '#4A6A9A',
    'overlap': '#5A7AAA',
    'interruption': '#6A8AAA',
    'long_pause': '#3A5878',
    'response_latency': '#4A6888',
    // Pragmatic Markers — muted crimson
    'echolalia': '#8B2B35',
    'pronoun_reversal': '#9B3545',
    'stereotyped_phrase': '#AA4555',
    'social_greeting': '#B85C5C',
    'question': '#9B6050',
    // Conversational — forest green
    'topic_shift': '#2A6040',
    'topic_maintenance': '#3A7050',
    'repair_initiation': '#2E6B55',
    'repair_completion': '#3A7860',
    'clarification_request': '#347A6A',
    // Linguistic — deep indigo
    'complex_sentence': '#4A3080',
    'simple_sentence': '#7868A8',
    'filled_pause': '#5A3888',
    'discourse_marker': '#6848A0',
    // Syntactic & Semantic — teal family
    'complex_syntax': '#00796B',
    'grammatical_error': '#D84315',
    'low_semantic_density': '#546E7A',
    'semantic_mismatch': '#5D4037',
    // General
    'feature_region': '#5A6470'
};

// Category metadata for grouping chips
const ANNOTATION_CATEGORIES = {
    'Turn-taking': { color: '#3B5B8B', types: ['turn_start', 'turn_end', 'overlap', 'interruption', 'long_pause', 'response_latency'] },
    'Pragmatic': { color: '#8B2B35', types: ['echolalia', 'pronoun_reversal', 'stereotyped_phrase', 'social_greeting', 'question'] },
    'Conversational': { color: '#2A6040', types: ['topic_shift', 'topic_maintenance', 'repair_initiation', 'repair_completion', 'clarification_request'] },
    'Linguistic': { color: '#4A3080', types: ['complex_sentence', 'simple_sentence', 'filled_pause', 'discourse_marker'] },
    'Syntactic & Semantic': { color: '#00796B', types: ['complex_syntax', 'grammatical_error', 'low_semantic_density', 'semantic_mismatch'] },
    'General': { color: '#5A6470', types: ['feature_region'] }
};

let currentTranscriptData = null;
let currentTranscriptText = null;
let isCompactView = true; // Always start in compact mode
let semanticCoherenceData = null;
let isSemanticCoherenceActive = false;

function renderAnnotatedTranscript(htmlContent, annotationSummary, transcriptText = null) {
    const container = document.getElementById('annotatedTranscript');
    const summaryPanel = document.getElementById('featureSummaryContent');
    const filterSelect = document.getElementById('featureFilter');
    const annotationCount = document.getElementById('annotationCount');

    if (!container || !summaryPanel || !filterSelect || !annotationCount) {
        console.error('Required elements not found for transcript rendering');
        return;
    }

    // Store current data
    currentTranscriptData = { html: htmlContent, summary: annotationSummary || {} };
    currentTranscriptText = transcriptText;

    // Parse the HTML to extract annotation data
    const parser = new DOMParser();
    const doc = parser.parseFromString(htmlContent, 'text/html');

    // Try to find transcript content, fallback to body if structure is different
    let transcriptDiv = doc.querySelector('.transcript-content');
    if (!transcriptDiv) {
        transcriptDiv = doc.querySelector('.annotated-transcript');
    }
    if (!transcriptDiv) {
        transcriptDiv = doc.body;
    }

    // Count total annotations
    const totalAnnotations = annotationSummary ?
        Object.values(annotationSummary).reduce((sum, count) => sum + count, 0) : 0;
    annotationCount.textContent = `${totalAnnotations} Feature${totalAnnotations !== 1 ? 's' : ''} Marked`;

    // Render feature summary chips — grouped by category
    summaryPanel.innerHTML = '';

    if (annotationSummary && Object.keys(annotationSummary).length > 0) {
        const featureEntries = Object.entries(annotationSummary).sort((a, b) => b[1] - a[1]);

        // Group by category
        const grouped = {};
        featureEntries.forEach(([featureType, count]) => {
            const catName = getFeatureCategoryName(featureType);
            if (!grouped[catName]) grouped[catName] = [];
            grouped[catName].push({ featureType, count });
        });

        Object.entries(grouped).forEach(([catName, features]) => {
            const catColor = (ANNOTATION_CATEGORIES[catName] || {}).color || '#5A6470';

            // Category label
            const catLabel = document.createElement('div');
            catLabel.className = 'w-full flex items-center gap-2 mt-3 mb-1.5 first:mt-0';
            catLabel.innerHTML = `
                <span class="w-2 h-2 rounded-full flex-shrink-0" style="background-color:${catColor}"></span>
                <span class="text-xs font-medium uppercase tracking-wider" style="color:${catColor}">${catName}</span>
                <span class="flex-1 h-px bg-primary-100"></span>
            `;
            summaryPanel.appendChild(catLabel);

            // Chips for this category
            features.forEach(({ featureType, count }) => {
                const chip = document.createElement('button');
                chip.className = 'feature-chip inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs transition-all cursor-pointer';
                chip.style.background = '#f8f9fa';
                chip.style.border = '1px solid #e9ecef';
                chip.style.color = '#343a40';
                chip.dataset.featureType = featureType;
                chip.innerHTML = `
                    <span class="font-medium">${formatFeatureName(featureType)}</span>
                    <span class="inline-flex items-center justify-center w-4 h-4 rounded-full text-white text-xs font-semibold" style="background-color:${catColor};font-size:10px">${count}</span>
                `;
                chip.addEventListener('click', () => {
                    // Active state
                    summaryPanel.querySelectorAll('.feature-chip').forEach(c => {
                        c.style.background = '#f8f9fa';
                        c.style.border = '1px solid #e9ecef';
                        c.style.color = '#343a40';
                    });
                    chip.style.background = catColor + '14';
                    chip.style.border = `1px solid ${catColor}50`;
                    chip.style.color = '#212529';
                    filterByFeatureType(featureType);
                    filterSelect.value = featureType;
                });
                summaryPanel.appendChild(chip);
            });
        });

        // Populate filter dropdown
        filterSelect.innerHTML = '<option value="all">All Features</option>';
        featureEntries.forEach(([featureType, count]) => {
            const option = document.createElement('option');
            option.value = featureType;
            option.textContent = `${formatFeatureName(featureType)} (${count})`;
            filterSelect.appendChild(option);
        });
    } else {
        summaryPanel.innerHTML = '<p class="text-xs text-primary-400">No features detected in this transcript.</p>';
        filterSelect.innerHTML = '<option value="all">All Features</option>';
    }

    // Render transcript with enhanced styling
    container.innerHTML = transcriptDiv.innerHTML || htmlContent;

    // Always apply compact view by default
    container.classList.add('compact-view');
    const toggleText = document.getElementById('viewToggleText');
    if (toggleText) {
        toggleText.textContent = 'Expanded View';
    }

    // Enhance annotations with interactive features
    enhanceAnnotations(container);

    // Setup event listeners
    setupTranscriptInteractivity();

    // Show statistics
    if (annotationSummary) {
        renderTranscriptStats(annotationSummary);
    }
}

function enhanceAnnotations(container) {
    // --- Annotation interactivity ---
    const annotations = container.querySelectorAll('.annotation, [class*="annotation"]');

    annotations.forEach(ann => {
        // Style inline annotations using the muted palette
        const type = ann.getAttribute('data-type') || ann.dataset.type || '';
        const color = ANNOTATION_COLORS[type];
        if (color) {
            ann.style.backgroundColor = color + '22';
            ann.style.color = color;
            ann.style.borderBottom = `1px solid ${color}60`;
        }

        ann.addEventListener('click', function () {
            container.querySelectorAll('.annotation-highlighted').forEach(el => {
                el.classList.remove('annotation-highlighted');
            });
            this.classList.add('annotation-highlighted');
            this.scrollIntoView({ behavior: 'smooth', block: 'center' });
        });
    });

    // --- Timestamp extraction (safe — no errors if absent) ---
    try {
        const utterances = container.querySelectorAll('.utterance');
        utterances.forEach(utt => {
            // 1. Check for data attributes from the backend
            let startMs = utt.getAttribute('data-start') || utt.getAttribute('data-timestamp');
            let endMs = utt.getAttribute('data-end');

            // 2. If not in attributes, look for CHAT bullet timing pattern •start_end• in text nodes
            if (!startMs) {
                const textEl = utt.querySelector('.text');
                if (textEl) {
                    // CHAT timing: bullet char (U+0015) or escaped as •digits_digits•
                    const raw = textEl.textContent || '';
                    const m = raw.match(/[•\u0015](\d+)_(\d+)[•\u0015]/);
                    if (m) {
                        startMs = m[1];
                        endMs = m[2];
                        // Strip the timing from the visible text
                        textEl.innerHTML = textEl.innerHTML.replace(/[•\u0015]\d+_\d+[•\u0015]/g, '').trim();
                    }
                }
            }

            if (startMs) {
                const fmt = ms => {
                    const s = Math.floor(parseInt(ms, 10) / 1000);
                    const m2 = Math.floor(s / 60);
                    const s2 = s % 60;
                    return `${m2}:${String(s2).padStart(2, '0')}`;
                };
                const label = document.createElement('span');
                label.className = 'ts-label';
                label.textContent = endMs ? `${fmt(startMs)}–${fmt(endMs)}` : fmt(startMs);
                utt.appendChild(label);
            }
        });
    } catch (_) {
        // Timestamp extraction is best-effort; never throw
    }
}

function setupTranscriptInteractivity() {
    // Search functionality
    const searchInput = document.getElementById('transcriptSearch');
    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            const query = e.target.value.toLowerCase();
            searchTranscript(query);
        });
    }

    // Filter functionality
    const filterSelect = document.getElementById('featureFilter');
    if (filterSelect) {
        filterSelect.addEventListener('change', (e) => {
            if (e.target.value === 'all') {
                clearFilters();
            } else {
                filterByFeatureType(e.target.value);
            }
        });
    }

    // Clear filters
    const clearBtn = document.getElementById('clearFilters');
    if (clearBtn) {
        clearBtn.addEventListener('click', clearFilters);
    }



    // Semantic coherence toggle
    const coherenceToggle = document.getElementById('semanticCoherenceToggle');
    if (coherenceToggle) {
        coherenceToggle.addEventListener('change', async (e) => {
            if (e.target.checked) {
                await analyzeSemanticCoherence();
            } else {
                clearSemanticCoherence();
            }
        });
    }
}

function searchTranscript(query) {
    const container = document.getElementById('annotatedTranscript');
    if (!container) return;

    const utterances = container.querySelectorAll('.utterance');

    if (!query.trim()) {
        utterances.forEach(utt => {
            utt.style.display = '';
            utt.classList.remove('search-highlight');
        });
        // Remove search marks
        container.querySelectorAll('mark.search-match').forEach(mark => {
            const parent = mark.parentNode;
            parent.replaceChild(document.createTextNode(mark.textContent), mark);
            parent.normalize();
        });
        return;
    }

    const escapedQuery = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const regex = new RegExp(`(${escapedQuery})`, 'gi');

    utterances.forEach(utt => {
        const text = utt.textContent.toLowerCase();
        if (text.includes(query.toLowerCase())) {
            utt.style.display = '';
            utt.classList.add('search-highlight');

            // Highlight matching text in text span
            const textSpan = utt.querySelector('.text');
            if (textSpan) {
                // Remove previous marks
                textSpan.querySelectorAll('mark.search-match').forEach(mark => {
                    const parent = mark.parentNode;
                    parent.replaceChild(document.createTextNode(mark.textContent), mark);
                    parent.normalize();
                });

                // Add new marks
                const originalHTML = textSpan.innerHTML;
                textSpan.innerHTML = originalHTML.replace(regex, '<mark class="search-match">$1</mark>');
            }
        } else {
            utt.style.display = 'none';
        }
    });
}

function filterByFeatureType(featureType) {
    const container = document.getElementById('annotatedTranscript');
    if (!container) return;

    const annotations = container.querySelectorAll('.annotation, [class*="annotation"]');
    let firstMatch = null;

    annotations.forEach(ann => {
        const annType = ann.getAttribute('data-type');
        if (annType === featureType) {
            ann.classList.add('annotation-filtered');
            if (!firstMatch) {
                firstMatch = ann;
            }
        } else {
            ann.classList.remove('annotation-filtered');
        }
    });

    // Scroll to first match
    if (firstMatch) {
        firstMatch.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    // Highlight utterances with this feature
    const utterances = container.querySelectorAll('.utterance');
    let hasAnyMatch = false;

    utterances.forEach(utt => {
        const hasFeature = utt.querySelector(`[data-type="${featureType}"]`);
        if (hasFeature) {
            utt.classList.add('utterance-highlighted');
            hasAnyMatch = true;
        } else {
            utt.classList.remove('utterance-highlighted');
        }
    });

    // If no matches found, show a message
    if (!hasAnyMatch && firstMatch === null) {
        console.log(`No annotations found for feature type: ${featureType}`);
    }
}

function clearFilters() {
    const container = document.getElementById('annotatedTranscript');
    const searchInput = document.getElementById('transcriptSearch');
    const filterSelect = document.getElementById('featureFilter');

    // Clear search
    if (searchInput) {
        searchInput.value = '';
    }

    // Clear filter
    if (filterSelect) {
        filterSelect.value = 'all';
    }

    // Reset all highlights
    container.querySelectorAll('.annotation-filtered, .annotation-highlighted, .utterance-highlighted, .search-highlight').forEach(el => {
        el.classList.remove('annotation-filtered', 'annotation-highlighted', 'utterance-highlighted', 'search-highlight');
    });

    container.querySelectorAll('.utterance').forEach(utt => {
        utt.style.display = '';
    });

    // Remove search marks
    container.querySelectorAll('mark.search-match').forEach(mark => {
        mark.outerHTML = mark.textContent;
    });
}

function toggleTranscriptView() {
    const container = document.getElementById('annotatedTranscript');
    const toggleText = document.getElementById('viewToggleText');
    isCompactView = !isCompactView;

    if (isCompactView) {
        container.classList.add('compact-view');
        toggleText.textContent = 'Expanded View';
    } else {
        container.classList.remove('compact-view');
        toggleText.textContent = 'Compact View';
    }
}

function renderTranscriptStats() {
    // Statistics panel removed from UI
}

function getFeatureCategory(featureType) {
    for (const [categoryName, category] of Object.entries(FEATURE_CATEGORIES)) {
        if (category.types.includes(featureType)) {
            return category;
        }
    }
    return FEATURE_CATEGORIES['General'];
}

function getFeatureCategoryName(featureType) {
    for (const [categoryName, category] of Object.entries(ANNOTATION_CATEGORIES)) {
        if (category.types.includes(featureType)) {
            return categoryName;
        }
    }
    return 'General';
}

function formatFeatureName(featureType) {
    return featureType
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

// ==============================
// Semantic Coherence Analysis
// ==============================

function extractTranscriptFromHTML(htmlContent) {
    // Extract transcript text from HTML
    const parser = new DOMParser();
    const doc = parser.parseFromString(htmlContent, 'text/html');
    const utterances = doc.querySelectorAll('.utterance');

    const transcriptLines = [];
    utterances.forEach(utt => {
        const speaker = utt.querySelector('.speaker')?.textContent.replace('*', '').replace(':', '').trim() || 'CHI';
        const text = utt.querySelector('.text')?.textContent.trim() || '';
        if (text) {
            transcriptLines.push(`*${speaker}: ${text}`);
        }
    });

    return transcriptLines.join('\n');
}

async function analyzeSemanticCoherence() {
    const container = document.getElementById('annotatedTranscript');
    if (!container || !currentTranscriptData) {
        console.error('Transcript container or data not available');
        return;
    }

    // Use stored transcript text if available, otherwise extract from HTML
    let transcriptText = currentTranscriptText;
    if (!transcriptText) {
        // Extract text from transcript
        const utterances = container.querySelectorAll('.utterance');
        if (utterances.length === 0) {
            console.error('No utterances found in transcript');
            return;
        }

        // Build transcript text from utterances
        const transcriptLines = [];
        utterances.forEach(utt => {
            const speaker = utt.querySelector('.speaker')?.textContent.replace('*', '').replace(':', '').trim() || 'CHI';
            const text = utt.querySelector('.text')?.textContent.trim() || '';
            if (text) {
                transcriptLines.push(`*${speaker}: ${text}`);
            }
        });

        transcriptText = transcriptLines.join('\n');
    }

    try {
        // Show loading state
        const toggle = document.getElementById('semanticCoherenceToggle');
        if (toggle) {
            toggle.disabled = true;
        }

        // Call API
        const formData = new FormData();
        formData.append('text', transcriptText);

        const response = await fetch(`${getApiUrl()}/analyze/semantic-coherence`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        const data = await response.json();
        semanticCoherenceData = data;
        isSemanticCoherenceActive = true;

        // Apply semantic coherence highlighting
        applySemanticCoherenceHighlighting(data);

        // Re-enable toggle
        if (toggle) {
            toggle.disabled = false;
        }

    } catch (error) {
        console.error('Semantic coherence analysis failed:', error);
        alert('Failed to analyze semantic coherence. Please try again.');

        // Re-enable toggle and uncheck
        const toggle = document.getElementById('semanticCoherenceToggle');
        if (toggle) {
            toggle.disabled = false;
            toggle.checked = false;
        }
    }
}

function applySemanticCoherenceHighlighting(data) {
    const container = document.getElementById('annotatedTranscript');
    if (!container) return;

    const utterances = container.querySelectorAll('.utterance');

    utterances.forEach((utt, idx) => {
        // Remove previous coherence classes
        utt.classList.remove('coherent-utterance', 'incoherent-utterance', 'coherence-unknown');

        const coherenceInfo = data.coherence_scores[idx];
        if (!coherenceInfo) return;

        if (coherenceInfo.is_coherent === true) {
            utt.classList.add('coherent-utterance');
            // Add tooltip with similarity score
            const similarity = (coherenceInfo.similarity * 100).toFixed(1);
            utt.title = `Semantically coherent (similarity: ${similarity}%)`;
        } else if (coherenceInfo.is_coherent === false) {
            utt.classList.add('incoherent-utterance');
            // Add tooltip with similarity score
            const similarity = (coherenceInfo.similarity * 100).toFixed(1);
            utt.title = `Semantically incoherent (similarity: ${similarity}%)`;
        } else {
            utt.classList.add('coherence-unknown');
            utt.title = 'Coherence analysis not available for this utterance';
        }
    });

    // Show overall coherence score
    showCoherenceSummary(data);
}

function showCoherenceSummary(data) {
    // Create or update summary element
    let summaryEl = document.getElementById('coherenceSummary');
    if (!summaryEl) {
        summaryEl = document.createElement('div');
        summaryEl.id = 'coherenceSummary';
        summaryEl.className = 'mt-4 p-4 bg-primary-50 border border-primary-200 rounded-xl';

        const container = document.getElementById('annotatedTranscript').parentElement;
        container.appendChild(summaryEl);
    }

    const overallScore = (data.overall_coherence * 100).toFixed(1);
    const coherentCount = data.coherence_scores.filter(s => s.is_coherent === true).length;
    const incoherentCount = data.coherence_scores.filter(s => s.is_coherent === false).length;

    summaryEl.innerHTML = `
        <div class="flex items-center justify-between">
            <div>
                <p class="text-xs font-medium text-primary-500 uppercase tracking-wide mb-2">Semantic Coherence</p>
                <div class="flex flex-wrap gap-4 text-xs">
                    <span class="flex items-center gap-1.5">
                        <span class="w-2 h-2 rounded-full bg-green-500"></span>
                        <strong class="text-primary-800">${coherentCount}</strong>
                        <span class="text-primary-500">coherent transitions</span>
                    </span>
                    <span class="flex items-center gap-1.5">
                        <span class="w-2 h-2 rounded-full bg-red-400"></span>
                        <strong class="text-primary-800">${incoherentCount}</strong>
                        <span class="text-primary-500">incoherent transitions</span>
                    </span>
                    <span class="flex items-center gap-1.5 text-primary-600">
                        Overall: <strong>${overallScore}%</strong>
                    </span>
                </div>
            </div>
            <button onclick="clearSemanticCoherence()" class="px-3 py-1.5 text-xs border border-primary-200 bg-white text-primary-600 rounded-lg hover:border-primary-400 hover:text-primary-900 transition-colors">
                Clear
            </button>
        </div>
    `;
}

function clearSemanticCoherence() {
    const container = document.getElementById('annotatedTranscript');
    if (!container) return;

    const utterances = container.querySelectorAll('.utterance');
    utterances.forEach(utt => {
        utt.classList.remove('coherent-utterance', 'incoherent-utterance', 'coherence-unknown');
        utt.title = '';
    });

    // Remove summary
    const summaryEl = document.getElementById('coherenceSummary');
    if (summaryEl) {
        summaryEl.remove();
    }

    semanticCoherenceData = null;
    isSemanticCoherenceActive = false;

    // Uncheck toggle
    const toggle = document.getElementById('semanticCoherenceToggle');
    if (toggle) {
        toggle.checked = false;
    }
}

// ==============================
// Waveform Visualization
// ==============================

let currentAudioFile = null;

/**
 * Extract waveform data from an audio file with feature analysis
 */
async function extractWaveform(audioFile) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = async (e) => {
            try {
                const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                const arrayBuffer = e.target.result;
                const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

                // Get channel data (use first channel)
                const channelData = audioBuffer.getChannelData(0);
                const sampleRate = audioBuffer.sampleRate;
                const duration = audioBuffer.duration;

                // Downsample for visualization (take every Nth sample)
                const samplesToShow = 2000; // Number of points to display
                const step = Math.max(1, Math.floor(channelData.length / samplesToShow));
                const waveform = [];

                // Calculate energy for each segment (for energy feature annotations)
                const energyThreshold = 0.1; // Threshold for high energy regions
                const energyRegions = [];

                for (let i = 0; i < channelData.length; i += step) {
                    // Get max and min in this chunk for better visualization
                    let max = 0;
                    let min = 0;
                    let energy = 0;
                    for (let j = i; j < Math.min(i + step, channelData.length); j++) {
                        const absValue = Math.abs(channelData[j]);
                        max = Math.max(max, absValue);
                        min = Math.min(min, -absValue);
                        energy += absValue * absValue;
                    }
                    energy = Math.sqrt(energy / step); // RMS energy
                    waveform.push({ max, min, energy });

                    // Track high energy regions (for energy feature annotations)
                    const timePos = (i / channelData.length) * duration;
                    if (energy > energyThreshold) {
                        energyRegions.push({ time: timePos, energy: energy });
                    }
                }

                // Calculate statistics for energy envelope and silence detection
                const energies = waveform.map(w => w.energy);
                const avgEnergy = energies.reduce((a, b) => a + b, 0) / energies.length;
                const maxEnergy = Math.max(...energies);
                const minEnergy = Math.min(...energies);

                // Compute smoothed energy envelope using simple moving average
                // This provides a cleaner visualization of relative loudness over time
                const smoothingWindow = 5; // Number of samples to average
                const smoothedEnergy = [];
                for (let i = 0; i < energies.length; i++) {
                    let sum = 0;
                    let count = 0;
                    for (let j = Math.max(0, i - smoothingWindow); j <= Math.min(energies.length - 1, i + smoothingWindow); j++) {
                        sum += energies[j];
                        count++;
                    }
                    smoothedEnergy.push(sum / count);
                }

                // Identify silence regions (low energy) for subtle shading
                // Threshold: regions below 30% of average energy are considered silence
                const silenceThreshold = avgEnergy * 0.3;
                const silenceRegions = waveform
                    .map((w, idx) => ({
                        idx,
                        energy: w.energy,
                        smoothedEnergy: smoothedEnergy[idx],
                        time: (idx / waveform.length) * duration,
                        isSilence: w.energy < silenceThreshold
                    }))
                    .filter(w => w.isSilence);

                resolve({
                    waveform,
                    duration,
                    sampleRate,
                    sampleCount: channelData.length,
                    energyEnvelope: smoothedEnergy, // Smoothed RMS energy for visualization
                    energyStats: {
                        avg: avgEnergy,
                        max: maxEnergy,
                        min: minEnergy
                    },
                    silenceRegions: silenceRegions // For subtle shading of pause regions
                });
            } catch (error) {
                reject(error);
            }
        };
        reader.onerror = reject;
        reader.readAsArrayBuffer(audioFile);
    });
}

/**
 * Render waveform on canvas with energy envelope overlay and speech activity bar
 * 
 * Research-grade enhancements:
 * - Blue waveform: Raw audio signal (amplitude vs time) - the primary visualization
 * - Energy envelope: Light, semi-transparent overlay showing relative loudness (RMS energy)
 * - Activity bar: Thin bar below waveform showing speech vs silence regions
 * - Hover tooltips: Interactive feedback for speech activity regions
 * 
 * Design rationale:
 * - Activity bar provides intuitive visual summary of speech dynamics
 * - Color-coded regions (active speech vs pauses) aid pattern recognition
 * - Minimal, calm design suitable for ASD research context
 * - No numeric values or overwhelming visual complexity
 * 
 * Scientific accuracy:
 * - This visualization shows signal-level properties only (amplitude, energy, silence)
 * - Acoustic features (pitch, MFCCs, spectral features) are NOT visualized here
 * - Features are computed as global statistics, not from specific time segments
 * - This is for user understanding/explainability, not model inference
 */
function renderWaveform(canvas, waveformData, color = '#3B82F6', featureInfo = null) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width = canvas.offsetWidth;
    const activityBarHeight = 8; // Height of activity bar below waveform
    const waveformHeight = 142; // Waveform area (leaving space for activity bar)
    const height = canvas.height = 150; // Total height: waveform + activity bar

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    if (!waveformData || !waveformData.waveform || waveformData.waveform.length === 0) {
        ctx.fillStyle = '#9CA3AF';
        ctx.font = '14px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('No waveform data available', width / 2, height / 2);
        return;
    }

    const waveform = waveformData.waveform;
    const centerY = waveformHeight / 2; // Center of waveform area
    const stepX = width / waveform.length;

    // Draw background
    ctx.fillStyle = '#F3F4F6';
    ctx.fillRect(0, 0, width, waveformHeight);

    // Draw subtle silence shading (low energy regions = pauses)
    // This provides visual context for speech vs silence without implying feature extraction
    if (waveformData.silenceRegions && waveformData.silenceRegions.length > 0) {
        ctx.fillStyle = 'rgba(156, 163, 175, 0.15)'; // Very subtle gray shading
        waveformData.silenceRegions.forEach(region => {
            const x = (region.idx / waveform.length) * width;
            ctx.fillRect(x, 0, stepX, waveformHeight);
        });
    }

    // Draw energy envelope overlay (smoothed RMS energy)
    // Enhanced visibility: slightly increased opacity for better perceptual clarity
    // This shows relative loudness over time as a semi-transparent overlay
    if (waveformData.energyEnvelope && waveformData.energyEnvelope.length > 0) {
        const maxEnergy = waveformData.energyStats?.max || 1;
        const envelope = waveformData.energyEnvelope;

        // Draw upper envelope (positive side)
        // Slightly increased opacity (0.45 vs 0.4) for better visibility while maintaining subtlety
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(251, 146, 60, 0.45)'; // Light orange, slightly more visible
        ctx.lineWidth = 1.5;
        ctx.moveTo(0, centerY);

        for (let i = 0; i < envelope.length; i++) {
            const x = i * stepX;
            // Normalize energy to waveform height (0 to centerY)
            const energyHeight = (envelope[i] / maxEnergy) * centerY * 0.8;
            const y = centerY - energyHeight;
            ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Draw lower envelope (negative side, symmetric)
        ctx.beginPath();
        ctx.moveTo(width, centerY);
        for (let i = envelope.length - 1; i >= 0; i--) {
            const x = i * stepX;
            const energyHeight = (envelope[i] / maxEnergy) * centerY * 0.8;
            const y = centerY + energyHeight;
            ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Fill envelope area with slightly increased visibility
        ctx.beginPath();
        ctx.moveTo(0, centerY);
        for (let i = 0; i < envelope.length; i++) {
            const x = i * stepX;
            const energyHeight = (envelope[i] / maxEnergy) * centerY * 0.8;
            ctx.lineTo(x, centerY - energyHeight);
        }
        for (let i = envelope.length - 1; i >= 0; i--) {
            const x = i * stepX;
            const energyHeight = (envelope[i] / maxEnergy) * centerY * 0.8;
            ctx.lineTo(x, centerY + energyHeight);
        }
        ctx.closePath();
        ctx.fillStyle = 'rgba(251, 146, 60, 0.1)'; // Slightly more visible fill (0.1 vs 0.08)
        ctx.fill();
    }

    // Draw main waveform (raw audio signal - amplitude vs time)
    // This is the primary visualization showing the actual speech signal
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();

    for (let i = 0; i < waveform.length; i++) {
        const x = i * stepX;
        const maxY = centerY - (waveform[i].max * centerY * 0.9);
        const minY = centerY - (waveform[i].min * centerY * 0.9);

        if (i === 0) {
            ctx.moveTo(x, maxY);
        } else {
            ctx.lineTo(x, maxY);
        }
    }

    // Draw bottom half
    for (let i = waveform.length - 1; i >= 0; i--) {
        const x = i * stepX;
        const minY = centerY - (waveform[i].min * centerY * 0.9);
        ctx.lineTo(x, minY);
    }

    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
    ctx.stroke();

    // Draw center line (zero amplitude reference)
    ctx.strokeStyle = '#9CA3AF';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, centerY);
    ctx.lineTo(width, centerY);
    ctx.stroke();

    // Draw speech activity bar below waveform
    // This provides an intuitive visual summary of speech vs silence regions
    // Design: Thin horizontal bar with color-coded segments
    const activityBarY = waveformHeight;
    const silenceThreshold = waveformData.energyStats?.avg * 0.3 || 0.1;

    // Draw activity bar background
    ctx.fillStyle = '#E5E7EB';
    ctx.fillRect(0, activityBarY, width, activityBarHeight);

    // Draw speech activity segments
    if (waveformData.energyEnvelope && waveformData.energyEnvelope.length > 0) {
        const envelope = waveformData.energyEnvelope;
        const segmentWidth = stepX;

        for (let i = 0; i < envelope.length; i++) {
            const x = i * segmentWidth;
            const energy = envelope[i];
            const isSpeech = energy > silenceThreshold;

            // Color coding: active speech (dark green) vs pause/silence (light gray)
            if (isSpeech) {
                // Gradient from light to darker green based on energy level
                const maxEnergy = waveformData.energyStats?.max || 1;
                const energyRatio = Math.min(energy / maxEnergy, 1);
                // Light green for low energy speech, darker for high energy
                const r = Math.floor(34 - (energyRatio * 15)); // 34-19 (dark green range)
                const g = Math.floor(197 - (energyRatio * 50)); // 197-147 (green range)
                const b = Math.floor(94 - (energyRatio * 30)); // 94-64 (green range)
                ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
            } else {
                // Light gray for silence/pause regions
                ctx.fillStyle = '#D1D5DB';
            }

            ctx.fillRect(x, activityBarY, Math.max(segmentWidth, 1), activityBarHeight);
        }
    }

    // Draw time markers at bottom (for temporal reference only)
    if (waveformData.duration) {
        ctx.fillStyle = '#6B7280';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'center';
        const timeMarkers = 5;
        for (let i = 0; i <= timeMarkers; i++) {
            const x = (i / timeMarkers) * width;
            const time = (i / timeMarkers) * waveformData.duration;
            ctx.fillText(time.toFixed(1) + 's', x, height - 2);
        }
    }

    // Store waveform data for hover tooltips
    canvas._waveformData = waveformData;
    canvas._stepX = stepX;
    canvas._silenceThreshold = silenceThreshold;
}

/**
 * Generate a plain-language textual summary of observed speech characteristics
 * Based on energy patterns, pause distribution, and overall speech activity
 * Uses descriptive language without numerical values or diagnostic claims
 * Improved analysis with more nuanced thresholds for better differentiation
 */
function generateSpeechSummary(waveformData) {
    if (!waveformData || !waveformData.energyEnvelope || !waveformData.energyStats) {
        return '<p>Analyzing speech characteristics...</p>';
    }

    const energies = waveformData.energyEnvelope;
    const avgEnergy = waveformData.energyStats.avg;
    const maxEnergy = waveformData.energyStats.max;
    const minEnergy = waveformData.energyStats.min;
    const silenceThreshold = avgEnergy * 0.3;

    // Calculate actual pause ratio by checking energy values directly
    let silenceCount = 0;
    let speechCount = 0;
    for (let i = 0; i < energies.length; i++) {
        if (energies[i] < silenceThreshold) {
            silenceCount++;
        } else {
            speechCount++;
        }
    }
    const pauseRatio = silenceCount / energies.length;
    const activeRatio = speechCount / energies.length;

    // Calculate energy variability (coefficient of variation)
    const energyMean = energies.reduce((a, b) => a + b, 0) / energies.length;
    const energyVariance = energies.reduce((sum, e) => sum + Math.pow(e - energyMean, 2), 0) / energies.length;
    const energyStd = Math.sqrt(energyVariance);
    const energyCV = energyMean > 0 ? energyStd / energyMean : 0;

    // Calculate energy range (how much variation between min and max)
    const energyRange = maxEnergy - minEnergy;
    const energyRangeRatio = maxEnergy > 0 ? energyRange / maxEnergy : 0;

    // Determine energy level description (more nuanced)
    let energyLevel = '';
    const energyPercentile = avgEnergy / maxEnergy;
    if (energyPercentile > 0.7) {
        energyLevel = 'generally higher energy';
    } else if (energyPercentile > 0.5) {
        energyLevel = 'moderate to higher energy';
    } else if (energyPercentile > 0.3) {
        energyLevel = 'moderate energy levels';
    } else if (energyPercentile > 0.15) {
        energyLevel = 'moderate to lower energy';
    } else {
        energyLevel = 'generally lower energy';
    }

    // Determine energy variability (more nuanced with range consideration)
    let variability = '';
    if (energyCV > 0.6 || energyRangeRatio > 0.8) {
        variability = 'shows considerable variation in loudness';
    } else if (energyCV > 0.35 || energyRangeRatio > 0.5) {
        variability = 'shows moderate variation in loudness';
    } else if (energyCV > 0.15 || energyRangeRatio > 0.25) {
        variability = 'shows some variation in loudness';
    } else {
        variability = 'shows relatively consistent loudness';
    }

    // Determine pause pattern (more nuanced thresholds)
    let pausePattern = '';
    if (pauseRatio > 0.5) {
        pausePattern = 'includes frequent pauses and breaks throughout';
    } else if (pauseRatio > 0.35) {
        pausePattern = 'includes frequent pauses and breaks';
    } else if (pauseRatio > 0.2) {
        pausePattern = 'includes occasional pauses';
    } else if (pauseRatio > 0.1) {
        pausePattern = 'includes some pauses';
    } else {
        pausePattern = 'shows relatively continuous speech with few pauses';
    }

    // Determine overall activity (more nuanced)
    let activity = '';
    if (activeRatio > 0.85) {
        activity = 'predominantly active speech';
    } else if (activeRatio > 0.7) {
        activity = 'mostly active speech';
    } else if (activeRatio > 0.5) {
        activity = 'mixed speech and silence periods';
    } else if (activeRatio > 0.3) {
        activity = 'more silence than active speech';
    } else {
        activity = 'predominantly silence with limited speech';
    }

    // Generate refined, non-diagnostic characteristics description
    // Focus on signal-level qualitative observations
    const characteristics = [];

    // Add loudness variation description
    if (energyCV > 0.5 || energyRangeRatio > 0.7) {
        characteristics.push('Variation in speech loudness across the recording');
    } else if (energyCV > 0.25 || energyRangeRatio > 0.4) {
        characteristics.push('Some variation in speech loudness across the recording');
    } else {
        characteristics.push('Relatively consistent speech loudness across the recording');
    }

    // Add pause pattern description
    if (pauseRatio > 0.3) {
        characteristics.push('Presence of short and longer pauses between speech segments');
    } else if (pauseRatio > 0.15) {
        characteristics.push('Presence of some pauses between speech segments');
    } else {
        characteristics.push('Relatively continuous speech with minimal pauses');
    }

    // Add activity pattern description
    if (activeRatio > 0.7 && pauseRatio > 0.2) {
        characteristics.push('Periods of continuous speech interspersed with low-activity intervals');
    } else if (activeRatio > 0.5) {
        characteristics.push('Mixed periods of active speech and silence');
    }

    // Add signal quality assessment
    if (maxEnergy > 0.1 && energyRangeRatio > 0.2) {
        characteristics.push('Overall signal quality suitable for acoustic analysis');
    } else if (maxEnergy > 0.05) {
        characteristics.push('Signal quality appears adequate for analysis');
    }

    // Combine into formatted list with HTML
    const summary = characteristics.length > 0
        ? characteristics.map(char => `<p>• ${char}</p>`).join('')
        : '<p>Signal characteristics are being analyzed.</p>';

    return summary;
}

/**
 * Display waveform for uploaded audio file with feature annotations
 * Shows waveform only before the annotated transcript section
 */
async function displayWaveform(audioFile, featureInfo = null) {
    const waveformSectionResults = document.getElementById('waveformSectionResults');
    const waveformCanvasResults = document.getElementById('waveformCanvasResults');
    const waveformInfoResults = document.getElementById('waveformInfoResults');
    const waveformAudioResults = document.getElementById('waveformAudioResults');
    const waveformSummaryResults = document.getElementById('waveformSummaryResults');

    if (!audioFile) {
        if (waveformSectionResults) waveformSectionResults.classList.add('hidden');
        return;
    }

    try {
        if (waveformInfoResults) waveformInfoResults.textContent = 'Processing waveform...';
        if (waveformSectionResults) waveformSectionResults.classList.remove('hidden');

        // Create audio element for playback
        const audioUrl = URL.createObjectURL(audioFile);
        if (waveformAudioResults) {
            waveformAudioResults.src = audioUrl;
            waveformAudioResults.style.display = 'block';
        }

        // Extract and render waveform
        const waveformData = await extractWaveform(audioFile);
        if (waveformCanvasResults) {
            renderWaveform(waveformCanvasResults, waveformData, '#3B82F6', featureInfo);
            setupWaveformTooltips(waveformCanvasResults, waveformData);
        }

        // Generate and display speech characteristics summary
        if (waveformSummaryResults) {
            const summary = generateSpeechSummary(waveformData);
            waveformSummaryResults.innerHTML = summary;
        }

        // Update info with feature extraction details
        const duration = waveformData.duration.toFixed(2);
        let finalInfoText = `Duration: ${duration}s | Sample Rate: ${waveformData.sampleRate}Hz | Samples: ${waveformData.sampleCount.toLocaleString()}`;

        if (featureInfo && featureInfo.features_extracted) {
            finalInfoText += ` | Features Extracted: ${featureInfo.features_extracted}`;
        }

        if (waveformInfoResults) waveformInfoResults.textContent = finalInfoText;

        // Handle window resize to redraw waveform
        let resizeTimeout;
        const resizeHandler = () => {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(() => {
                if (waveformCanvasResults) {
                    renderWaveform(waveformCanvasResults, waveformData, '#3B82F6', featureInfo);
                    setupWaveformTooltips(waveformCanvasResults, waveformData);
                }
            }, 250);
        };

        // Remove old resize listeners and add new one
        window.removeEventListener('resize', resizeHandler);
        window.addEventListener('resize', resizeHandler);

    } catch (error) {
        console.error('Error displaying waveform:', error);
        if (waveformInfoResults) waveformInfoResults.textContent = 'Error loading waveform';
        if (waveformSectionResults) waveformSectionResults.classList.remove('hidden');
    }
}


function simulateCounterfactualChat() {
    const responseBox = document.getElementById("cfChatResponse");

    responseBox.innerHTML = `
        <strong>Simulated Response:</strong><br><br>
        Increasing the frequency of <em>continuation markers</em> (e.g., “uhm”)
        would increase the model’s estimated likelihood of ASD.
        <br><br>
        In the current model, this feature is associated with
        disrupted conversational flow and increased hesitation.
        A change of this magnitude alone may not flip the prediction,
        but it would contribute positively toward ASD risk.
        <br><br>
       
    `;

    responseBox.classList.remove("hidden");
}

/**
 * Setup hover tooltips for waveform and activity bar
 * Provides interactive feedback with descriptive, non-numeric energy descriptions
 * Enhanced with relative energy level descriptions for better user understanding
 */
function setupWaveformTooltips(canvas, waveformData) {
    // Remove existing tooltip if present
    const existingTooltip = document.getElementById('waveformTooltip');
    if (existingTooltip) {
        existingTooltip.remove();
    }

    // Create tooltip element with refined styling
    const tooltip = document.createElement('div');
    tooltip.id = 'waveformTooltip';
    tooltip.style.cssText = `
        position: fixed;
        background: rgba(17, 24, 39, 0.92);
        color: white;
        padding: 6px 10px;
        border-radius: 4px;
        font-size: 11px;
        pointer-events: none;
        z-index: 1000;
        display: none;
        white-space: nowrap;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
        transition: opacity 0.15s ease;
    `;
    document.body.appendChild(tooltip);

    const activityBarY = 142; // Top of activity bar
    const activityBarHeight = 8;
    const waveformHeight = 142;
    const silenceThreshold = waveformData.energyStats?.avg * 0.3 || 0.1;
    const avgEnergy = waveformData.energyStats?.avg || 0.1;
    const maxEnergy = waveformData.energyStats?.max || 1;

    /**
     * Get descriptive, non-numeric energy level description
     * Uses relative terms: high, moderate, low, pause
     */
    function getEnergyDescription(energy, isSpeech) {
        if (!isSpeech) {
            return 'Pause region';
        }

        // Relative energy levels: high (>70% of max), moderate (30-70%), low (threshold to 30%)
        const energyRatio = energy / maxEnergy;
        if (energyRatio > 0.7) {
            return 'High energy speech';
        } else if (energyRatio > 0.3) {
            return 'Moderate energy speech';
        } else {
            return 'Low energy speech';
        }
    }

    canvas.addEventListener('mousemove', (e) => {
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        if (!waveformData.energyEnvelope) {
            tooltip.style.display = 'none';
            return;
        }

        const stepX = canvas.width / waveformData.energyEnvelope.length;
        const segmentIndex = Math.floor(x / stepX);

        if (segmentIndex >= 0 && segmentIndex < waveformData.energyEnvelope.length) {
            const energy = waveformData.energyEnvelope[segmentIndex];
            const isSpeech = energy > silenceThreshold;
            const description = getEnergyDescription(energy, isSpeech);

            // Show tooltip for both waveform area and activity bar
            if (y >= 0 && y <= waveformHeight + activityBarHeight) {
                tooltip.textContent = description;

                // Position tooltip near cursor with offset (10px right, 12px below)
                const offsetX = 10;
                const offsetY = 12;
                let tooltipX = e.clientX + offsetX;
                let tooltipY = e.clientY + offsetY;

                // Show tooltip first to measure dimensions
                tooltip.style.display = 'block';
                tooltip.style.visibility = 'hidden'; // Temporarily hide to measure
                tooltip.style.left = tooltipX + 'px';
                tooltip.style.top = tooltipY + 'px';

                // Get tooltip dimensions after it's in the DOM
                const tooltipRect = tooltip.getBoundingClientRect();
                const tooltipWidth = tooltipRect.width;
                const tooltipHeight = tooltipRect.height;

                // Keep tooltip within viewport bounds
                const viewportWidth = window.innerWidth;
                const viewportHeight = window.innerHeight;

                // Adjust horizontal position if tooltip would go off-screen right
                if (tooltipX + tooltipWidth > viewportWidth) {
                    tooltipX = e.clientX - tooltipWidth - offsetX; // Position to the left of cursor
                }

                // Adjust horizontal position if tooltip would go off-screen left
                if (tooltipX < 0) {
                    tooltipX = offsetX;
                }

                // Adjust vertical position if tooltip would go off-screen bottom
                if (tooltipY + tooltipHeight > viewportHeight) {
                    tooltipY = e.clientY - tooltipHeight - offsetY; // Position above cursor
                }

                // Adjust vertical position if tooltip would go off-screen top
                if (tooltipY < 0) {
                    tooltipY = offsetY;
                }

                // Apply final position and make visible
                tooltip.style.left = tooltipX + 'px';
                tooltip.style.top = tooltipY + 'px';
                tooltip.style.visibility = 'visible';
            } else {
                tooltip.style.display = 'none';
            }
        } else {
            tooltip.style.display = 'none';
        }
    });

    canvas.addEventListener('mouseleave', () => {
        tooltip.style.display = 'none';
    });
}

async function askCounterfactualGPT() {

    const input = document.getElementById("cfUserInput");
    const responseBox = document.getElementById("cfChatResponse");

    const question = input?.value?.trim();
    if (!question) return;

    responseBox?.classList.remove("hidden");
    if (responseBox) responseBox.innerHTML = "Thinking...";

    try {
        const res = await fetch(`${getApiUrl()}/counterfactual/chat`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ question })
        });

        const data = await res.json();

        if (responseBox) {
            responseBox.innerHTML = `
                <strong>Result:</strong><br>
                Prediction: ${data.new_prediction}<br>
                Confidence: ${(data.confidence * 100).toFixed(1)}%<br><br>
                <strong>Clinical Explanation:</strong><br>
                ${data.explanation}
            `;
        }

    } catch (err) {
        console.error(err);
        if (responseBox) {
            responseBox.innerHTML = "Failed to generate explanation.";
        }
    }
}

// Test connection on load
setTimeout(testConnection, 500);
