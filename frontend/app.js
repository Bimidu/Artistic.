// API URL
function getApiUrl() {
    return document.getElementById('apiUrl').value.replace(/\/$/, '');
}

// Mode switching with toggle
const toggleOptions = document.querySelectorAll('.toggle-option');
const toggleSlider = document.getElementById('toggleSlider');

function updateToggleSlider() {
    const activeOption = document.querySelector('.toggle-option.active');
    const toggleSwitch = document.getElementById('modeToggle');
    
    if (activeOption && toggleSwitch) {
        const optionRect = activeOption.getBoundingClientRect();
        const switchRect = toggleSwitch.getBoundingClientRect();
        const relativeLeft = optionRect.left - switchRect.left;
        
        toggleSlider.style.width = `${optionRect.width}px`;
        toggleSlider.style.transform = `translateX(${relativeLeft - 4}px)`;
    }
}

toggleOptions.forEach(option => {
    option.addEventListener('click', () => {
        toggleOptions.forEach(opt => opt.classList.remove('active'));
        option.classList.add('active');
        
        const mode = option.dataset.mode;
        document.querySelectorAll('.mode-content').forEach(c => {
            c.classList.add('hidden');
        });
        document.getElementById(mode + 'Mode').classList.remove('hidden');
        
        updateToggleSlider();
    });
});

// Initialize toggle slider position
setTimeout(updateToggleSlider, 100);
window.addEventListener('resize', updateToggleSlider);

// Tab switching
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(t => {
            t.classList.remove('border-primary-900', 'text-primary-900');
            t.classList.add('border-transparent', 'text-primary-500');
        });
        tab.classList.add('border-primary-900', 'text-primary-900');
        tab.classList.remove('border-transparent', 'text-primary-500');
        
        const inputType = tab.dataset.input;
        document.querySelectorAll('.input-panel').forEach(p => p.classList.add('hidden'));
        document.getElementById(inputType + 'Panel').classList.remove('hidden');
    });
});

// File upload handling
function setupUploadArea(areaId, inputId, selectedId, allowedExtensions) {
    const area = document.getElementById(areaId);
    const input = document.getElementById(inputId);
    const selected = selectedId ? document.getElementById(selectedId) : null;
    
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
            handleFileSelect(file, input, selected, allowedExtensions);
        }
    });
    
    input.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleFileSelect(file, input, selected, allowedExtensions);
        }
    });
}

function handleFileSelect(file, input, selected, allowedExtensions) {
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!allowedExtensions.includes(ext)) {
        alert('Invalid file type. Allowed: ' + allowedExtensions.join(', '));
        return;
    }
    
    if (selected) {
        selected.textContent = 'Selected: ' + file.name;
    }
    
    // Enable the corresponding predict button
    if (input.id === 'audioFileInput') {
        document.getElementById('predictAudioBtn').disabled = false;
    } else if (input.id === 'chaFileInput') {
        document.getElementById('predictChaBtn').disabled = false;
    }
}

setupUploadArea('audioUploadArea', 'audioFileInput', 'selectedAudioFile', ['.wav', '.mp3', '.flac']);
setupUploadArea('chaUploadArea', 'chaFileInput', 'selectedChaFile', ['.cha']);
setupUploadArea('inspectUploadArea', 'inspectFileInput', null, ['.wav', '.cha', '.txt']);

// API calls
async function testConnection() {
    const statusDot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');
    
    try {
        const response = await fetch(`${getApiUrl()}/health`);
        if (response.ok) {
            const data = await response.json();
            statusDot.className = 'w-2.5 h-2.5 rounded-full bg-green-400 status-connected';
            statusText.textContent = `Connected (${data.models_available} models, ${data.features_supported} features)`;
        } else {
            throw new Error('Not healthy');
        }
    } catch (error) {
        statusDot.className = 'w-2.5 h-2.5 rounded-full bg-red-400';
        statusText.textContent = 'Disconnected';
    }
}

async function predictFromAudio() {
    const fileInput = document.getElementById('audioFileInput');
    const participantId = document.getElementById('audioParticipantId').value || 'CHI';
    
    if (!fileInput.files[0]) {
        alert('Please select an audio file');
        return;
    }
    
    showLoading('resultsArea');
    
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('participant_id', participantId);
    
    try {
        const response = await fetch(`${getApiUrl()}/predict/audio`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displayResults(data);
        } else {
            displayError(data.detail || 'Prediction failed');
        }
    } catch (error) {
        displayError('Connection error: ' + error.message);
    }
}

async function predictFromText() {
    const text = document.getElementById('textInput').value;
    
    if (!text.trim()) {
        alert('Please enter some text');
        return;
    }
    
    showLoading('resultsArea');
    
    try {
        const response = await fetch(`${getApiUrl()}/predict/text`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text, participant_id: 'CHI' })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displayResults(data);
        } else {
            displayError(data.detail || 'Prediction failed');
        }
    } catch (error) {
        displayError('Connection error: ' + error.message);
    }
}

async function predictFromChatFile() {
    const fileInput = document.getElementById('chaFileInput');
    
    if (!fileInput.files[0]) {
        alert('Please select a CHAT file');
        return;
    }
    
    showLoading('resultsArea');
    
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    try {
        const response = await fetch(`${getApiUrl()}/predict/transcript`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displayResults(data);
        } else {
            displayError(data.detail || 'Prediction failed');
        }
    } catch (error) {
        displayError('Connection error: ' + error.message);
    }
}

function showLoading(elementId) {
    document.getElementById(elementId).innerHTML = `
        <div class="text-center py-24">
            <div class="spinner mx-auto mb-6"></div>
            <div class="text-lg text-primary-600">Analyzing...</div>
        </div>
    `;
}

function displayResults(data) {
    const isAsd = data.prediction === 'ASD';
    const confidence = (data.confidence * 100).toFixed(1);
    
    document.getElementById('resultsArea').innerHTML = `
        <div class="flex items-center justify-between mb-8">
            <span class="px-10 py-4 rounded-full text-3xl ${isAsd ? 'bg-red-500 text-white' : 'bg-green-500 text-white'}">
                ${data.prediction}
            </span>
            <span class="text-base text-primary-600">
                ${data.features_extracted} features analyzed
            </span>
        </div>
        
        <div class="mb-8">
            <div class="flex justify-between text-xl mb-4">
                <span class="text-primary-900">Confidence</span>
                <span class="text-primary-900">${confidence}%</span>
            </div>
            <div class="w-full h-2 bg-primary-200 rounded-full overflow-hidden">
                <div class="h-full transition-all duration-700 ${isAsd ? 'bg-red-500' : 'bg-green-500'}" style="width: ${confidence}%"></div>
            </div>
        </div>
        
        <div class="grid grid-cols-2 gap-6 mb-8">
            <div class="text-center p-8 bg-red-50 rounded-3xl">
                <div class="text-5xl font-medium text-red-600">
                    ${(data.probabilities.ASD * 100).toFixed(1)}%
                </div>
                <div class="text-base text-primary-600 mt-3">ASD Probability</div>
            </div>
            <div class="text-center p-8 bg-green-50 rounded-3xl">
                <div class="text-5xl font-medium text-green-600">
                    ${(data.probabilities.TD * 100).toFixed(1)}%
                </div>
                <div class="text-base text-primary-600 mt-3">TD Probability</div>
            </div>
        </div>
        
        <div class="text-sm text-primary-500 pt-6">
            Model: ${data.model_used} | Input: ${data.input_type}
            ${data.duration ? ' | Duration: ' + data.duration.toFixed(1) + 's' : ''}
        </div>
    `;
    
    // Show annotated transcript
    if (data.annotated_transcript_html) {
        document.getElementById('annotationCard').classList.remove('hidden');
        document.getElementById('annotatedTranscript').innerHTML = data.annotated_transcript_html;
    }
}

function displayError(message) {
    document.getElementById('resultsArea').innerHTML = `
        <div class="text-center py-24">
            <div class="text-6xl mb-6">⚠️</div>
            <div class="text-xl text-primary-900">${message}</div>
        </div>
    `;
}

// Training mode functions
async function loadDatasets() {
    const listEl = document.getElementById('datasetList');
    listEl.innerHTML = '<div class="text-center py-16"><div class="spinner mx-auto"></div></div>';
    
    try {
        const response = await fetch(`${getApiUrl()}/training/datasets`);
        const data = await response.json();
        
        if (data.datasets && data.datasets.length > 0) {
            listEl.innerHTML = data.datasets.map(ds => `
                <div class="flex items-center p-6 bg-white rounded-2xl mb-4 hover:bg-primary-100 transition-colors">
                    <input type="checkbox" class="dataset-checkbox w-5 h-5 text-primary-600 rounded" value="${ds.path}">
                    <div class="flex-1 ml-5">
                        <div class="text-lg text-primary-900">${ds.name}</div>
                        <div class="text-base text-primary-500 mt-1">${ds.chat_files} CHAT files, ${ds.audio_files} audio files</div>
                    </div>
                </div>
            `).join('');
        } else {
            listEl.innerHTML = '<div class="text-center py-16 text-primary-400 text-xl">No datasets found</div>';
        }
    } catch (error) {
        listEl.innerHTML = `<div class="text-red-500 text-base p-6">Error loading datasets: ${error.message}</div>`;
    }
}

async function extractFeatures() {
    const selectedDatasets = Array.from(document.querySelectorAll('.dataset-checkbox:checked')).map(cb => cb.value);
    
    if (selectedDatasets.length === 0) {
        alert('Please select at least one dataset');
        return;
    }
    
    const statusEl = document.getElementById('trainingStatus');
    const statusContent = document.getElementById('trainingStatusContent');
    statusEl.classList.remove('hidden');
    statusContent.innerHTML = '<div class="spinner mx-auto"></div>';
    
    try {
        const response = await fetch(`${getApiUrl()}/training/extract-features`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ dataset_paths: selectedDatasets })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            statusContent.innerHTML = `
                <div class="text-green-600 text-lg mb-3">✓ Feature extraction complete</div>
                <div class="text-lg text-primary-900 mb-2">
                    ${data.total_samples} samples, ${data.features_count} features
                </div>
                <div class="text-sm text-primary-500">
                    Output: ${data.output_file}
                </div>
            `;
        } else {
            statusContent.innerHTML = `<div class="text-red-500 text-base">${data.detail}</div>`;
        }
    } catch (error) {
        statusContent.innerHTML = `<div class="text-red-500 text-base">Error: ${error.message}</div>`;
    }
}

async function startTraining() {
    alert('Training functionality will be available in a future update. For now, use the example training scripts.');
}

async function loadFeatures() {
    const gridEl = document.getElementById('featureGrid');
    gridEl.innerHTML = '<div class="col-span-full text-center"><div class="spinner mx-auto"></div></div>';
    
    try {
        const response = await fetch(`${getApiUrl()}/features`);
        const data = await response.json();
        
        if (data.features && data.features.length > 0) {
            gridEl.innerHTML = data.features.map(f => `<div class="px-5 py-4 bg-white rounded-2xl text-sm font-mono text-primary-700 hover:bg-primary-100 transition-colors">${f}</div>`).join('');
        } else {
            gridEl.innerHTML = '<div class="col-span-full text-center text-primary-400 text-xl">No features found</div>';
        }
    } catch (error) {
        gridEl.innerHTML = `<div class="col-span-full text-red-500 text-base">Error: ${error.message}</div>`;
    }
}

// Inspect file handling
document.getElementById('inspectFileInput').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    const resultsEl = document.getElementById('inspectionResults');
    resultsEl.innerHTML = '<div class="spinner mx-auto mt-8"></div>';
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch(`${getApiUrl()}/training/inspect-features`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            resultsEl.innerHTML = `
                <div class="bg-white rounded-2xl p-8">
                    <div class="text-lg space-y-3">
                        <div><span class="text-primary-600">Participant:</span> <span class="text-primary-900 font-medium">${data.participant_id}</span></div>
                        <div><span class="text-primary-600">Input Type:</span> <span class="text-primary-900 font-medium">${data.input_type}</span></div>
                        <div><span class="text-primary-600">Features Extracted:</span> <span class="text-primary-900 font-medium">${data.total_features}</span></div>
                        <div><span class="text-primary-600">Utterances:</span> <span class="text-primary-900 font-medium">${data.utterance_count}</span></div>
                    </div>
                </div>
                <div class="mt-8">
                    <h4 class="text-2xl font-medium text-primary-900 mb-4">Annotated Transcript</h4>
                    <div class="bg-white rounded-2xl p-8 max-h-96 overflow-y-auto font-mono text-sm leading-loose">${data.annotated_transcript_html}</div>
                </div>
            `;
        } else {
            resultsEl.innerHTML = `<div class="text-red-500 text-base">${data.detail}</div>`;
        }
    } catch (error) {
        resultsEl.innerHTML = `<div class="text-red-500 text-base">Error: ${error.message}</div>`;
    }
});

// Test connection on load
setTimeout(testConnection, 500);
