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
        
        // Toggle API config bar visibility
        const apiConfigBar = document.getElementById('apiConfigBar');
        if (mode === 'training') {
            apiConfigBar.classList.remove('hidden');
            // Auto-load models when entering training mode
            setTimeout(() => {
                loadAvailableModels();
            }, 100);
        } else {
            apiConfigBar.classList.add('hidden');
        }
        
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
        
        // Load models when switching to a prediction tab
        loadModelsForPrediction();
    });
});

// Toggle model selection based on fusion checkbox
function setupFusionToggle() {
    const fusionCheckboxes = [
        { checkbox: 'audioUseFusion', select: 'audioModelSelect', container: 'audioModelSelectContainer', note: 'audioModelSelectNote' },
        { checkbox: 'textUseFusion', select: 'textModelSelect', container: 'textModelSelectContainer', note: 'textModelSelectNote' },
        { checkbox: 'chaUseFusion', select: 'chaModelSelect', container: 'chaModelSelectContainer', note: 'chaModelSelectNote' }
    ];
    
    fusionCheckboxes.forEach(({ checkbox, select, container, note }) => {
        const checkboxEl = document.getElementById(checkbox);
        const selectEl = document.getElementById(select);
        const containerEl = document.getElementById(container);
        const noteEl = document.getElementById(note);
        
        if (checkboxEl && selectEl && containerEl && noteEl) {
            checkboxEl.addEventListener('change', () => {
                if (checkboxEl.checked) {
                    // Fusion enabled - disable model selection
                    selectEl.disabled = true;
                    selectEl.value = ''; // Reset to "Best Model (Auto)"
                    containerEl.style.opacity = '0.5';
                    containerEl.style.pointerEvents = 'none';
                    noteEl.classList.remove('hidden');
                } else {
                    // Fusion disabled - enable model selection
                    selectEl.disabled = false;
                    containerEl.style.opacity = '1';
                    containerEl.style.pointerEvents = 'auto';
                    noteEl.classList.add('hidden');
                }
            });
        }
    });
}

// Initialize fusion toggles on page load
document.addEventListener('DOMContentLoaded', () => {
    setupFusionToggle();
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
            
            // Load models for prediction dropdowns
            loadModelsForPrediction();
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
    const modelName = document.getElementById('audioModelSelect').value || null;
    const useFusion = document.getElementById('audioUseFusion').checked;
    
    if (!fileInput.files[0]) {
        alert('Please select an audio file');
        return;
    }
    
    showLoading('resultsArea');
    
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('participant_id', participantId);
    if (modelName) {
        formData.append('model_name', modelName);
    }
    formData.append('use_fusion', useFusion);
    
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
    const modelName = document.getElementById('textModelSelect').value || null;
    const useFusion = document.getElementById('textUseFusion').checked;
    
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
                model_name: modelName,
                use_fusion: useFusion
            })
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
    const modelName = document.getElementById('chaModelSelect').value || null;
    const useFusion = document.getElementById('chaUseFusion').checked;
    
    if (!fileInput.files[0]) {
        alert('Please select a CHAT file');
        return;
    }
    
    console.log('Uploading CHAT file:', fileInput.files[0].name, 'Fusion:', useFusion, 'Model:', modelName);
    showLoading('resultsArea');
    
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('use_fusion', useFusion);
    if (modelName) {
        formData.append('model_name', modelName);
    }
    
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
        <div class="text-center py-24">
            <div class="spinner mx-auto mb-6"></div>
            <div class="text-lg text-primary-600">Analyzing...</div>
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
        
        componentBreakdown = '<div class="mt-6 pt-6 border-t border-primary-200"><div class="text-lg font-medium text-primary-900 mb-4">Component Breakdown</div><div class="space-y-3">';
        
        for (const comp of data.component_breakdown) {
            const compName = componentNames[comp.component] || comp.component;
            const color = componentColors[comp.component] || 'gray';
            const compIsAsd = comp.prediction === 'ASD';
            const compConf = (comp.confidence * 100).toFixed(1);
            const asdProb = ((comp.probabilities.ASD || 0) * 100).toFixed(1);
            const tdProb = ((comp.probabilities.TD || 0) * 100).toFixed(1);
            
            componentBreakdown += `
                <div class="p-4 bg-${color}-50 rounded-xl">
                    <div class="flex items-center justify-between mb-2">
                        <div class="flex items-center gap-2">
                            <span class="text-base font-medium text-primary-900">${compName}</span>
                            <span class="px-2 py-1 text-xs rounded-full ${compIsAsd ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'}">${comp.prediction}</span>
                        </div>
                        <span class="text-sm text-primary-600">${compConf}% confidence</span>
                    </div>
                    <div class="flex gap-2 text-xs">
                        <span class="text-primary-600">ASD: ${asdProb}%</span>
                        <span class="text-primary-600">TD: ${tdProb}%</span>
                    </div>
                </div>
            `;
        }
        
        componentBreakdown += '</div></div>';
    }
    
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
        
        ${componentBreakdown}
        
        <div class="text-sm text-primary-500 pt-6 border-t border-primary-200">
            <div class="mb-2">
                <span class="font-medium text-primary-700">Model(s) Used:</span>
                ${data.models_used ? 
                    `<span class="text-primary-600">${data.models_used.join(', ')}</span>` : 
                    `<span class="text-primary-600">${data.model_used || 'Unknown'}</span>`
                }
            </div>
            <div class="text-xs text-primary-500">
                Input: ${data.input_type}${data.component ? ` | Component: ${data.component}` : ''}
                ${data.duration ? ' | Duration: ' + data.duration.toFixed(1) + 's' : ''}
            </div>
        </div>
    `;

    // ==============================
    // Local SHAP Waterfall
    // ==============================
    const localShapSection = document.getElementById('localShapSection');
    const localShapImg = document.getElementById('localShapWaterfall');

    if (data.local_shap && data.local_shap.waterfall) {
        localShapImg.src =
            getApiUrl() + data.local_shap.waterfall + '?t=' + Date.now();
        localShapSection.classList.remove('hidden');
    } else {
        localShapSection.classList.add('hidden');
    }

    // Show annotated transcript
    if (data.annotated_transcript_html) {
        document.getElementById('annotationCard').classList.remove('hidden');
        document.getElementById('annotatedTranscript').innerHTML = data.annotated_transcript_html;
    }


}

function displayError(message) {
    let additionalHelp = '';
    
    if (message.includes('No models in registry') || message.includes('No models')) {
        additionalHelp = `
            <div class="mt-6 p-6 bg-yellow-50 rounded-2xl text-left">
                <div class="text-base text-primary-900 font-medium mb-3">💡 How to fix:</div>
                <div class="text-sm text-primary-700 space-y-2">
                    <div>1. Switch to <strong>Training Mode</strong></div>
                    <div>2. Click <strong>Refresh</strong> to load datasets</div>
                    <div>3. Select one or more datasets</div>
                    <div>4. Click <strong>Extract Features</strong></div>
                    <div>5. Train a model using the training scripts</div>
                    <div class="mt-3 pt-3 border-t border-primary-200">Or restart the API server to reload existing models</div>
                </div>
            </div>
        `;
    }
    
    document.getElementById('resultsArea').innerHTML = `
        <div class="text-center py-24">
            <div class="text-6xl mb-6">⚠️</div>
            <div class="text-xl text-primary-900 mb-4">${message}</div>
            ${additionalHelp}
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
    const selectedDatasets = Array.from(document.querySelectorAll('.dataset-checkbox:checked')).map(cb => cb.value);
    
    if (selectedDatasets.length === 0) {
        alert('Please select at least one dataset');
        return;
    }
    
    // Get selected model types
    const selectedModels = Array.from(document.querySelectorAll('input[type="checkbox"][value]:checked'))
        .filter(cb => ['random_forest', 'xgboost', 'lightgbm', 'svm', 'logistic', 'gradient_boosting', 'adaboost'].includes(cb.value))
        .map(cb => cb.value);
    
    if (selectedModels.length === 0) {
        alert('Please select at least one model type');
        return;
    }
    
    const component = document.getElementById('trainingComponent').value;
    const featureSelectionEnabled = document.getElementById('featureSelectionEnabled').checked;
    const nFeatures = parseInt(document.getElementById('nFeatures').value) || 30;
    const testSize = parseFloat(document.getElementById('testSize').value) / 100 || 0.2;
    const randomState = parseInt(document.getElementById('randomState').value) || 42;
    const customHyperparams = getCustomHyperparameters();
    
    const statusEl = document.getElementById('trainingStatus');
    const statusContent = document.getElementById('trainingStatusContent');
    statusEl.classList.remove('hidden');
    statusContent.innerHTML = '<div class="spinner mx-auto"></div><div class="text-center mt-4 text-base text-primary-600">Initializing training...</div>';
    
    try {
        const response = await fetch(`${getApiUrl()}/training/train`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                dataset_paths: selectedDatasets,
                model_types: selectedModels,
                component: component,
                feature_selection: featureSelectionEnabled,
                n_features: featureSelectionEnabled ? nFeatures : null,
                test_size: testSize,
                random_state: randomState,
                custom_hyperparameters: customHyperparams
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            // Start polling for progress
            pollTrainingProgress();
        } else {
            statusContent.innerHTML = `<div class="text-red-500 text-base">${data.detail || 'Failed to start training'}</div>`;
        }
    } catch (error) {
        statusContent.innerHTML = `<div class="text-red-500 text-base">Error: ${error.message}</div>`;
    }
}

// Toggle feature count input based on checkbox
document.addEventListener('DOMContentLoaded', () => {
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
});

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
        const currentModel = status.current_model ? ` - ${status.current_model}` : '';
        
        statusContent.innerHTML = `
            <div class="mb-4">
                <div class="flex justify-between text-sm text-primary-600 mb-2">
                    <span>${status.message}${currentModel}</span>
                    <span>${progressPercent}%</span>
                </div>
                <div class="w-full h-3 bg-primary-200 rounded-full overflow-hidden">
                    <div class="h-full bg-primary-600 transition-all duration-500" style="width: ${progressPercent}%"></div>
                </div>
            </div>
            <div class="text-sm text-primary-500">
                Training ${status.total_models || 0} models for ${status.component || 'component'}...
            </div>
        `;
    } else if (status.status === 'completed') {
        let resultsHtml = '';
        if (status.results && Object.keys(status.results).length > 0) {
            resultsHtml = '<div class="mt-4 space-y-2">';
            for (const [model, metrics] of Object.entries(status.results)) {
                resultsHtml += `
                    <div class="flex items-center justify-between p-3 bg-green-50 rounded-lg">
                        <span class="font-medium text-primary-900">${model}</span>
                        <div class="flex gap-4 text-sm">
                            <span class="text-primary-600">Acc: ${(metrics.accuracy * 100).toFixed(1)}%</span>
                            <span class="text-primary-600">F1: ${(metrics.f1_score * 100).toFixed(1)}%</span>
                        </div>
                    </div>
                `;
            }
            resultsHtml += '</div>';
        }
        
        statusContent.innerHTML = `
            <div class="text-green-600 text-lg mb-3 flex items-center gap-2">
                <span class="text-2xl">✓</span>
                <span>${status.message}</span>
            </div>
            ${resultsHtml}
        `;
    } else if (status.status === 'error') {
        let errorDetails = '';
        const errorMsg = status.error || status.message;
        
        // Parse common errors and provide helpful solutions
        if (errorMsg.includes('missing diagnosis') || errorMsg.includes('Insufficient samples')) {
            errorDetails = `
                <div class="mt-4 p-4 bg-yellow-50 rounded-lg text-sm">
                    <div class="font-medium text-primary-900 mb-2">💡 Possible Solutions:</div>
                    <ul class="list-disc list-inside space-y-1 text-primary-700">
                        <li>Some CHAT files may be missing diagnosis labels</li>
                        <li>Try selecting different datasets</li>
                        <li>Ensure datasets have proper CHAT format with diagnosis codes</li>
                        <li>Check that files contain participant diagnosis information</li>
                    </ul>
                </div>
            `;
        } else if (errorMsg.includes('No features extracted')) {
            errorDetails = `
                <div class="mt-4 p-4 bg-yellow-50 rounded-lg text-sm">
                    <div class="font-medium text-primary-900 mb-2">💡 Possible Solutions:</div>
                    <ul class="list-disc list-inside space-y-1 text-primary-700">
                        <li>Check that selected datasets contain .cha files</li>
                        <li>Verify CHAT files are properly formatted</li>
                        <li>Try extracting features first to diagnose issues</li>
                    </ul>
                </div>
            `;
        }
        
        statusContent.innerHTML = `
            <div class="text-red-500 text-base">
                <div class="text-lg mb-2 flex items-center gap-2">
                    <span class="text-2xl">✗</span>
                    <span>Training Failed</span>
                </div>
                <div class="text-sm mt-2 p-3 bg-red-50 rounded-lg text-red-700">
                    ${errorMsg}
                </div>
                ${errorDetails}
            </div>
        `;
    } else {
        statusContent.innerHTML = `<div class="text-primary-500 text-base">${status.message}</div>`;
    }
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

async function loadAvailableModels() {
    const container = document.getElementById('availableModelsContainer');
    if (!container) return;
    
    container.innerHTML = '<div class="text-center py-8"><div class="spinner mx-auto"></div></div>';
    
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
            
            // Display models grouped by component
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
                        <h3 class="text-2xl font-medium text-primary-900 mb-4 flex items-center gap-3">
                            ${componentName}
                            <span class="px-3 py-1 bg-${color}-100 text-${color}-700 text-sm rounded-full">${models.length} model${models.length > 1 ? 's' : ''}</span>
                        </h3>
                        <div class="space-y-4">
                `;
                
                for (const model of models) {
                    const isBest = model.name === data.best_model;
                    const accuracy = (model.accuracy * 100).toFixed(1);
                    const f1 = (model.f1_score * 100).toFixed(1);
                    const precision = (model.precision * 100).toFixed(1);
                    const recall = (model.recall * 100).toFixed(1);
                    const rocAuc = model.roc_auc ? (model.roc_auc * 100).toFixed(1) : null;
                    const matthews = model.matthews_corr ? model.matthews_corr.toFixed(3) : 'N/A';
                    const date = new Date(model.created_at).toLocaleDateString();
                    const time = new Date(model.created_at).toLocaleTimeString();
                    
                    modelsHtml += `
                        <div class="p-5 bg-white rounded-2xl hover:bg-primary-50 transition-colors ${isBest ? 'ring-2 ring-primary-600' : ''}">
                            <div class="flex items-start justify-between mb-3">
                                <div class="flex-1">
                                    <div class="flex items-center gap-2 mb-1">
                                        <h4 class="text-lg font-medium text-primary-900">${model.type}</h4>
                                        ${isBest ? '<span class="px-2 py-0.5 bg-primary-600 text-white text-xs rounded-full">Best</span>' : ''}
                                    </div>
                                    <div class="text-xs text-primary-500">${date} at ${time}</div>
                                </div>
                                <button class="px-3 py-1.5 text-red-600 hover:bg-red-50 rounded-lg transition-colors text-xs" onclick="deleteModel('${model.name}')">
                                    Delete
                                </button>
                            </div>
                            
                            <div class="grid grid-cols-3 md:grid-cols-6 gap-2 mb-3">
                                <div class="text-center p-2 bg-primary-50 rounded-lg">
                                    <div class="text-lg font-medium text-primary-900">${accuracy}%</div>
                                    <div class="text-xs text-primary-600">Accuracy</div>
                                </div>
                                <div class="text-center p-2 bg-primary-50 rounded-lg">
                                    <div class="text-lg font-medium text-primary-900">${f1}%</div>
                                    <div class="text-xs text-primary-600">F1</div>
                                </div>
                                <div class="text-center p-2 bg-primary-50 rounded-lg">
                                    <div class="text-lg font-medium text-primary-900">${precision}%</div>
                                    <div class="text-xs text-primary-600">Precision</div>
                                </div>
                                <div class="text-center p-2 bg-primary-50 rounded-lg">
                                    <div class="text-lg font-medium text-primary-900">${recall}%</div>
                                    <div class="text-xs text-primary-600">Recall</div>
                                </div>
                                <div class="text-center p-2 bg-primary-50 rounded-lg">
                                    <div class="text-lg font-medium text-primary-900">${model.n_features}</div>
                                    <div class="text-xs text-primary-600">Features</div>
                                </div>
                                <div class="text-center p-2 bg-primary-50 rounded-lg">
                                    <div class="text-lg font-medium text-primary-900">${model.training_samples}</div>
                                    <div class="text-xs text-primary-600">Samples</div>
                                </div>
                            </div>
                            ${rocAuc ? `<div class="mb-3 flex gap-2 items-center justify-center"><span class="text-xs text-primary-600">ROC-AUC:</span><span class="font-medium text-sm">${rocAuc}%</span><span class="text-xs text-primary-600 ml-3">Matthews:</span><span class="font-medium text-sm">${matthews}</span></div>` : ''}
                            <button class="w-full px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors text-sm" onclick='showModelDetails(${JSON.stringify(model)})'>
                                View Detailed Metrics & Graphs
                            </button>
                        </div>
                    `;
                }
                
                modelsHtml += `
                        </div>
                    </div>
                `;
            }
            
            container.innerHTML = modelsHtml;
        } else {
            container.innerHTML = `
                <div class="text-center py-16">
                    <div class="text-6xl mb-4">📦</div>
                    <div class="text-xl text-primary-600 mb-2">No models trained yet</div>
                    <div class="text-base text-primary-500">Train your first model to get started</div>
                </div>
            `;
        }
    } catch (error) {
        container.innerHTML = `<div class="text-red-500 text-base p-6">Error loading models: ${error.message}</div>`;
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
        <div class="space-y-8">
            <!-- Model Info -->
            <div class="bg-primary-50 rounded-2xl p-6">
                <h3 class="text-2xl font-medium text-primary-900 mb-4">${model.type} Model</h3>
                <div class="grid grid-cols-2 gap-4 text-sm">
                    <div><span class="text-primary-600">Component:</span> <span class="font-medium">${model.component || 'pragmatic_conversational'}</span></div>
                    <div><span class="text-primary-600">Features:</span> <span class="font-medium">${model.n_features}</span></div>
                    <div><span class="text-primary-600">Training Samples:</span> <span class="font-medium">${model.training_samples}</span></div>
                    <div><span class="text-primary-600">Created:</span> <span class="font-medium">${new Date(model.created_at).toLocaleString()}</span></div>
                </div>
            </div>
            
            <!-- Performance Metrics -->
            <div>
                <h3 class="text-xl font-medium text-primary-900 mb-3">Performance Metrics</h3>
                <div class="grid grid-cols-3 md:grid-cols-6 gap-3">
                    <div class="bg-primary-50 rounded-lg p-3 text-center border border-primary-200">
                        <div class="text-xl font-semibold text-primary-900">${accuracy}%</div>
                        <div class="text-xs text-primary-600 mt-1">Accuracy</div>
                    </div>
                    <div class="bg-primary-50 rounded-lg p-3 text-center border border-primary-200">
                        <div class="text-xl font-semibold text-primary-900">${f1}%</div>
                        <div class="text-xs text-primary-600 mt-1">F1 Score</div>
                    </div>
                    <div class="bg-primary-50 rounded-lg p-3 text-center border border-primary-200">
                        <div class="text-xl font-semibold text-primary-900">${precision}%</div>
                        <div class="text-xs text-primary-600 mt-1">Precision</div>
                    </div>
                    <div class="bg-primary-50 rounded-lg p-3 text-center border border-primary-200">
                        <div class="text-xl font-semibold text-primary-900">${recall}%</div>
                        <div class="text-xs text-primary-600 mt-1">Recall</div>
                    </div>
                    <div class="bg-primary-50 rounded-lg p-3 text-center border border-primary-200">
                        <div class="text-xl font-semibold text-primary-900">${rocAuc}${rocAuc !== 'N/A' ? '%' : ''}</div>
                        <div class="text-xs text-primary-600 mt-1">ROC-AUC</div>
                    </div>
                    <div class="bg-primary-50 rounded-lg p-3 text-center border border-primary-200">
                        <div class="text-xl font-semibold text-primary-900">${matthews}</div>
                        <div class="text-xs text-primary-600 mt-1">Matthews</div>
                    </div>
                </div>
            </div>
            
            <!-- Confusion Matrix -->
            <div>
                <h3 class="text-xl font-medium text-primary-900 mb-3">Confusion Matrix</h3>
                ${confusionMatrixHtml}
            </div>
            <!-- SHAP Explanations -->
                    <div class="mt-10">
                        <h3 class="text-2xl font-medium text-primary-900 mb-4">
                            Global SHAP Explanations
                        </h3>
            
                        <p class="text-sm text-primary-600 mb-6">
                            Feature importance across the full training dataset
                        </p>
            
                        <div class="grid md:grid-cols-2 gap-8">
                            <div class="bg-white rounded-2xl p-6 border border-primary-200">
                                <h4 class="text-lg font-medium mb-3">Beeswarm</h4>
                                <img
                                    src="${getApiUrl()}${model.shap.beeswarm}?t=${Date.now()}"
                                    class="w-full rounded-xl border"
                                />
                            </div>
            
                            <div class="bg-white rounded-2xl p-6 border border-primary-200">
                                <h4 class="text-lg font-medium mb-3">Mean |SHAP| Importance</h4>
                                <img
                                    src="${getApiUrl()}${model.shap.bar}?t=${Date.now()}"
                                    class="w-full rounded-xl border"
                                />
                            </div>
                        </div>
                    </div>
                \`;
            }
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
    'logistic': {
        'C': {
            value: 1.0, type: 'number', min: 0.001, max: 100, step: 0.001,
            description: 'Inverse regularization strength',
            range: 'Typical: 0.01-10',
            effect: 'Higher = less regularization, more complex model. Lower = more regularization, simpler model.'
        },
        'max_iter': {
            value: 1000, type: 'number', min: 100, max: 5000,
            description: 'Maximum iterations for solver convergence',
            range: 'Typical: 100-2000',
            effect: 'Higher = more attempts to converge. Too low may not converge. Too high wastes time.'
        }
    },
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
        .filter(cb => ['random_forest', 'xgboost', 'lightgbm', 'svm', 'logistic', 'gradient_boosting', 'adaboost'].includes(cb.value))
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
            'logistic': 'Logistic Regression',
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
        .filter(cb => ['random_forest', 'xgboost', 'lightgbm', 'svm', 'logistic', 'gradient_boosting', 'adaboost'].includes(cb.value))
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
document.addEventListener('DOMContentLoaded', () => {
    const modelCheckboxes = document.querySelectorAll('input[type="checkbox"][value]');
    modelCheckboxes.forEach(cb => {
        if (['random_forest', 'xgboost', 'lightgbm', 'svm', 'logistic', 'gradient_boosting', 'adaboost'].includes(cb.value)) {
            cb.addEventListener('change', () => {
                const section = document.getElementById('hyperparamSection');
                if (!section.classList.contains('hidden')) {
                    updateHyperparamControls();
                }
            });
        }
    });
});

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

// Test connection on load
setTimeout(testConnection, 500);
