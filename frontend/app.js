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
    const useFusion = document.getElementById('chaUseFusion').checked;
    
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
            const errorMsg = data.detail || data.error || JSON.stringify(data);
            console.error('Prediction error:', errorMsg);
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
        .filter(cb => ['random_forest', 'xgboost', 'lightgbm', 'svm'].includes(cb.value))
        .map(cb => cb.value);
    
    if (selectedModels.length === 0) {
        alert('Please select at least one model type');
        return;
    }
    
    const component = document.getElementById('trainingComponent').value;
    const featureSelectionEnabled = document.getElementById('featureSelectionEnabled').checked;
    const nFeatures = parseInt(document.getElementById('nFeatures').value) || 30;
    
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
                n_features: featureSelectionEnabled ? nFeatures : null
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
                    const date = new Date(model.created_at).toLocaleDateString();
                    
                    modelsHtml += `
                        <div class="p-6 bg-white rounded-2xl hover:bg-primary-50 transition-colors ${isBest ? 'ring-2 ring-primary-600' : ''}">
                            <div class="flex items-start justify-between mb-4">
                                <div class="flex-1">
                                    <div class="flex items-center gap-3 mb-2">
                                        <h4 class="text-xl font-medium text-primary-900">${model.type}</h4>
                                        ${isBest ? '<span class="px-3 py-1 bg-primary-600 text-white text-xs rounded-full">Best Overall</span>' : ''}
                                    </div>
                                    <div class="text-sm text-primary-500">Created: ${date}</div>
                                </div>
                                <button class="px-4 py-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors text-sm" onclick="deleteModel('${model.name}')">
                                    Delete
                                </button>
                            </div>
                            
                            <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                                <div class="text-center p-3 bg-primary-50 rounded-xl">
                                    <div class="text-2xl font-medium text-primary-900">${accuracy}%</div>
                                    <div class="text-xs text-primary-600 mt-1">Accuracy</div>
                                </div>
                                <div class="text-center p-3 bg-primary-50 rounded-xl">
                                    <div class="text-2xl font-medium text-primary-900">${f1}%</div>
                                    <div class="text-xs text-primary-600 mt-1">F1 Score</div>
                                </div>
                                <div class="text-center p-3 bg-primary-50 rounded-xl">
                                    <div class="text-2xl font-medium text-primary-900">${model.n_features}</div>
                                    <div class="text-xs text-primary-600 mt-1">Features</div>
                                </div>
                                <div class="text-center p-3 bg-primary-50 rounded-xl">
                                    <div class="text-2xl font-medium text-primary-900">${model.training_samples}</div>
                                    <div class="text-xs text-primary-600 mt-1">Samples</div>
                                </div>
                            </div>
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

// Test connection on load
setTimeout(testConnection, 500);
