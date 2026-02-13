// API URL
function getApiUrl() {
    return document.getElementById('apiUrl').value.replace(/\/$/, '');
}

// Mode switching with toggle
const toggleOptions = document.querySelectorAll('.toggle-option');
const toggleSlider = document.getElementById('toggleSlider');
const landingSection = document.getElementById('landingSection');

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
            if (landingSection) {
                landingSection.classList.add('hidden');
            }
            // Auto-load models when entering training mode
            setTimeout(() => {
                loadAvailableModels();
            }, 100);
        } else {
            apiConfigBar.classList.add('hidden');
            if (landingSection) {
                landingSection.classList.remove('hidden');
            }
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
        // Display waveform for audio files
        currentAudioFile = file;
        displayWaveform(file);
    } else if (input.id === 'chaFileInput') {
        document.getElementById('predictChaBtn').disabled = false;
    }
}

setupUploadArea('audioUploadArea', 'audioFileInput', 'selectedAudioFile', ['.wav', '.mp3', '.flac']);
setupUploadArea('chaUploadArea', 'chaFileInput', 'selectedChaFile', ['.cha']);

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
    
    // Store audio file for waveform display
    currentAudioFile = fileInput.files[0];
    
    // Display waveform immediately
    await displayWaveform(currentAudioFile);
    
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
            // Ensure waveform is still visible after results with feature info
            if (currentAudioFile) {
                await displayWaveform(currentAudioFile, data);
            }
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

    // Show annotated transcript with interactive features
    if (data.annotated_transcript_html) {
        document.getElementById('annotationCard').classList.remove('hidden');
        // Store transcript text for semantic coherence analysis
        const transcriptText = data.transcript || extractTranscriptFromHTML(data.annotated_transcript_html);
        renderAnnotatedTranscript(data.annotated_transcript_html, data.annotation_summary || {}, transcriptText);
    }

    //Counterfactuals
    if (data.counterfactual) {
    renderCounterfactual(data.counterfactual);
    document
        .getElementById("cfChatSection")
        .classList.remove("hidden");
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
    
    listEl.innerHTML = '<div class="text-center py-16"><div class="spinner mx-auto"></div></div>';
    
    try {
        const response = await fetch(`${getApiUrl()}/training/datasets`);
        const data = await response.json();
        
        if (data.datasets && data.datasets.length > 0) {
            listEl.innerHTML = data.datasets.map(ds => `
                <div class="flex items-center p-4 bg-primary-50 rounded-xl mb-3 hover:bg-primary-100 transition-colors">
                    <input type="checkbox" class="extraction-dataset-checkbox w-5 h-5 text-primary-600 rounded" value="${ds.path}" data-name="${ds.name}">
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

async function loadAvailableDatasetsForTraining() {
    // This is for training - shows datasets from CSV
    const listEl = document.getElementById('datasetList');
    const component = document.getElementById('trainingComponent').value;
    
    listEl.innerHTML = '<div class="text-center py-16"><div class="spinner mx-auto"></div></div>';
    
    try {
        const response = await fetch(`${getApiUrl()}/training/available-datasets/${component}`);
        const data = await response.json();
        
        if (data.csv_exists && data.datasets && data.datasets.length > 0) {
            listEl.innerHTML = `
                <div class="mb-4 p-4 bg-green-50 rounded-xl border border-green-200">
                    <div class="text-sm text-green-700">
                        <strong>✓ Features CSV found:</strong> ${data.total_samples} total samples from ${data.total_datasets} dataset(s)
                    </div>
                </div>
                ${data.datasets.map(ds => `
                    <div class="flex items-center p-6 bg-white rounded-2xl mb-4 hover:bg-primary-100 transition-colors">
                        <input type="checkbox" class="dataset-checkbox w-5 h-5 text-primary-600 rounded" value="${ds.name}" data-name="${ds.name}">
                        <div class="flex-1 ml-5">
                            <div class="text-lg text-primary-900">${ds.name}</div>
                            <div class="text-base text-primary-500 mt-1">${ds.samples} samples available</div>
                        </div>
                    </div>
                `).join('')}
            `;
        } else {
            listEl.innerHTML = `
                <div class="mb-4 p-4 bg-yellow-50 rounded-xl border border-yellow-200">
                    <div class="text-sm text-yellow-700">
                        <strong>⚠ No features CSV found for ${component}</strong>
                    </div>
                    <div class="text-xs text-yellow-600 mt-2">
                        Please extract features first using the "Extract Features" section.
                    </div>
                </div>
                <div class="text-center py-16 text-primary-400 text-xl">
                    ${data.message || 'No datasets available'}
                </div>
            `;
        }
    } catch (error) {
        listEl.innerHTML = `<div class="text-center py-16 text-red-500 text-xl">Error: ${error.message}</div>`;
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
    statusContent.innerHTML = '<div class="spinner mx-auto"></div><div class="text-center mt-4 text-base text-primary-600">Extracting features...</div>';
    
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
                <div class="text-green-600 text-lg mb-3">✓ Feature extraction complete</div>
                <div class="text-base text-primary-900 mb-2">
                    <strong>Total samples:</strong> ${data.total_samples || data.new_samples}
                    ${data.new_samples ? ` (${data.new_samples} new)` : ''}
                </div>
                <div class="text-base text-primary-900 mb-2">
                    <strong>Features:</strong> ${data.features_count}
                </div>
                <div class="text-sm text-primary-500 mb-2">
                    <strong>Output:</strong> ${data.output_file}
                </div>
                ${data.datasets_updated ? `<div class="text-sm text-primary-600 mt-2">Updated datasets: ${data.datasets_updated.join(', ')}</div>` : ''}
            `;
            
            // Reload available datasets for training after extraction
            setTimeout(() => {
                loadAvailableDatasetsForTraining();
            }, 1000);
        } else {
            statusContent.innerHTML = `<div class="text-red-500 text-base">${data.detail || 'Feature extraction failed'}</div>`;
        }
    } catch (error) {
        statusContent.innerHTML = `<div class="text-red-500 text-base">Error: ${error.message}</div>`;
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
    
    const statusEl = document.getElementById('trainingStatus');
    const statusContent = document.getElementById('trainingStatusContent');
    statusEl.classList.remove('hidden');
    statusContent.innerHTML = '<div class="spinner mx-auto"></div><div class="text-center mt-4 text-base text-primary-600">Initializing training...</div>';
    
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
        'logistic': { label: 'Logistic Regression' },
        'gradient_boosting': { label: 'Gradient Boosting' },
        'adaboost': { label: 'AdaBoost' },
        'lightgbm': { label: 'LightGBM' },
        'svm': { label: 'SVM (RBF)' }
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
        label.className = 'flex items-center cursor-pointer p-4 bg-white rounded-2xl hover:bg-primary-100 transition-colors';
        label.innerHTML = `
            <input type="checkbox" value="${modelValue}" ${isChecked ? 'checked' : ''} class="w-5 h-5 text-primary-600 rounded">
            <span class="ml-3 text-base text-primary-900">${info.label}</span>
        `;
        
        modelTypesContainer.appendChild(label);
    });
}

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
                        <h3 class="text-2xl font-medium text-primary-900 mb-4 flex items-center gap-3">
                            ${componentName}
                            <span class="px-3 py-1 bg-${color}-100 text-${color}-700 text-sm rounded-full">${models.length} model${models.length > 1 ? 's' : ''}</span>
                        </h3>
                        <div class="bg-white rounded-2xl overflow-hidden border border-primary-200">
                            <div class="overflow-x-auto">
                                <table class="w-full">
                                    <thead class="bg-primary-50">
                                        <tr>
                                            <th class="px-6 py-4 text-left text-sm font-semibold text-primary-900">Model Type</th>
                                            <th class="px-6 py-4 text-center text-sm font-semibold text-primary-900">Accuracy</th>
                                            <th class="px-6 py-4 text-center text-sm font-semibold text-primary-900">F1 Score</th>
                                            <th class="px-6 py-4 text-center text-sm font-semibold text-primary-900">Precision</th>
                                            <th class="px-6 py-4 text-center text-sm font-semibold text-primary-900">Recall</th>
                                            <th class="px-6 py-4 text-center text-sm font-semibold text-primary-900">ROC-AUC</th>
                                            <th class="px-6 py-4 text-center text-sm font-semibold text-primary-900">Features</th>
                                            <th class="px-6 py-4 text-center text-sm font-semibold text-primary-900">Samples</th>
                                            <th class="px-6 py-4 text-center text-sm font-semibold text-primary-900">Created</th>
                                            <th class="px-6 py-4 text-center text-sm font-semibold text-primary-900">Actions</th>
                                        </tr>
                                    </thead>
                                    <tbody class="divide-y divide-primary-200">
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
                        <tr class="hover:bg-primary-50 transition-colors ${isBest ? 'bg-primary-100' : ''}">
                            <td class="px-6 py-4">
                                <div class="flex items-center gap-2">
                                    <span class="text-base font-medium text-primary-900">${model.type}</span>
                                    ${isBest ? '<span class="px-2 py-0.5 bg-primary-600 text-white text-xs rounded-full">Best</span>' : ''}
                                </div>
                            </td>
                            <td class="px-6 py-4 text-center text-sm text-primary-700">${accuracy}%</td>
                            <td class="px-6 py-4 text-center text-sm text-primary-700">${f1}%</td>
                            <td class="px-6 py-4 text-center text-sm text-primary-700">${precision}%</td>
                            <td class="px-6 py-4 text-center text-sm text-primary-700">${recall}%</td>
                            <td class="px-6 py-4 text-center text-sm text-primary-700">${rocAuc}${rocAuc !== 'N/A' ? '%' : ''}</td>
                            <td class="px-6 py-4 text-center text-sm text-primary-700">${model.n_features}</td>
                            <td class="px-6 py-4 text-center text-sm text-primary-700">${model.training_samples}</td>
                            <td class="px-6 py-4 text-center text-sm text-primary-600">${date}<br><span class="text-xs">${time}</span></td>
                            <td class="px-6 py-4 text-center">
                                <div class="flex items-center justify-center gap-2">
                                    <button class="px-3 py-1.5 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors text-xs" onclick='showModelDetails(${JSON.stringify(model)})'>
                                        View
                                    </button>
                                    <button class="px-3 py-1.5 text-red-600 hover:bg-red-50 rounded-lg transition-colors text-xs" onclick="deleteModel('${model.name}')">
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
            ${model.shap ? `
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
                ` : `
                <!-- No SHAP -->
                <div class="mt-10 bg-primary-50 border border-primary-200 rounded-2xl p-6">
                    <h3 class="text-lg font-medium text-primary-900 mb-2">
                        SHAP Explanations
                    </h3>
                    <p class="text-sm text-primary-600">
                        SHAP explanations are not available for this model.
                        <br />
                        This may be due to model type limitations (e.g., SVM) or skipped training.
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

function renderCounterfactual(counterfactual) {
    if (!counterfactual) return;

    // Show section
    document
        .getElementById("counterfactualSection")
        .classList.remove("hidden");

    // What-if text
    document.getElementById("whatIfBox").innerHTML =
        generateWhatIfText(counterfactual);

    // Summary
    document.getElementById("cfFlipped").textContent =
        counterfactual.prediction_flipped ? "Yes " : "No ";

    document.getElementById("cfL2").textContent =
        counterfactual.l2_change.toFixed(3);

    document.getElementById("cfTotal").textContent =
        counterfactual.total_features_changed;

    // Table
    const tbody = document.getElementById("cfTableBody");
    tbody.innerHTML = "";

    counterfactual.top_changes.forEach(change => {
        const row = document.createElement("tr");
        row.className = "border-b last:border-b-0";

        row.innerHTML = `
            <td class="py-2 font-medium">
                ${change.feature.replaceAll("_", " ")}
            </td>
            <td class="py-2">
                ${change.from.toFixed(3)}
            </td>
            <td class="py-2">
                ${change.to.toFixed(3)}
            </td>
            <td class="py-2 ${
                change.change > 0 ? "text-green-600" : "text-red-600"
            }">
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

// Feature type categories and colors
const FEATURE_CATEGORIES = {
    'Turn-Taking': {
        types: ['turn_start', 'turn_end', 'overlap', 'interruption', 'long_pause', 'response_latency'],
        color: '#2196F3',
        icon: '🔄'
    },
    'Pragmatic Markers': {
        types: ['echolalia', 'pronoun_reversal', 'stereotyped_phrase', 'social_greeting', 'question'],
        color: '#F44336',
        icon: '💬'
    },
    'Conversational': {
        types: ['topic_shift', 'topic_maintenance', 'repair_initiation', 'repair_completion', 'clarification_request'],
        color: '#4CAF50',
        icon: '🗣️'
    },
    'Linguistic': {
        types: ['complex_sentence', 'simple_sentence', 'filled_pause', 'discourse_marker'],
        color: '#9C27B0',
        icon: '📝'
    },
    'General': {
        types: ['feature_region'],
        color: '#607D8B',
        icon: '📍'
    }
};

// Color mapping for annotation types
const ANNOTATION_COLORS = {
    'turn_start': '#2196F3',
    'turn_end': '#1976D2',
    'overlap': '#03A9F4',
    'interruption': '#00BCD4',
    'long_pause': '#0097A7',
    'response_latency': '#00838F',
    'echolalia': '#F44336',
    'pronoun_reversal': '#E91E63',
    'stereotyped_phrase': '#FF5722',
    'social_greeting': '#FF9800',
    'question': '#FFC107',
    'topic_shift': '#4CAF50',
    'topic_maintenance': '#8BC34A',
    'repair_initiation': '#CDDC39',
    'repair_completion': '#009688',
    'clarification_request': '#00BFA5',
    'complex_sentence': '#9C27B0',
    'simple_sentence': '#E1BEE7',
    'filled_pause': '#7B1FA2',
    'discourse_marker': '#AB47BC',
    'feature_region': '#607D8B'
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
    
    // Render feature summary chips
    summaryPanel.innerHTML = '';
    
    if (annotationSummary && Object.keys(annotationSummary).length > 0) {
        const featureEntries = Object.entries(annotationSummary).sort((a, b) => b[1] - a[1]);
        
        featureEntries.forEach(([featureType, count]) => {
            const category = getFeatureCategory(featureType);
            const color = ANNOTATION_COLORS[featureType] || category.color;
            
            const chip = document.createElement('button');
            chip.className = 'feature-chip px-4 py-2 rounded-lg text-sm font-medium transition-all hover:scale-105 cursor-pointer';
            chip.style.backgroundColor = color + '20';
            chip.style.borderLeft = `4px solid ${color}`;
            chip.style.color = '#1a1a1a';
            chip.dataset.featureType = featureType;
            chip.innerHTML = `
                <span class="font-semibold">${formatFeatureName(featureType)}</span>
                <span class="ml-2 px-2 py-0.5 rounded-full text-xs" style="background-color: ${color}; color: white;">
                    ${count}
                </span>
            `;
            
            chip.addEventListener('click', () => {
                filterByFeatureType(featureType);
                filterSelect.value = featureType;
            });
            
            summaryPanel.appendChild(chip);
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
        summaryPanel.innerHTML = '<p class="text-sm text-primary-500">No features detected</p>';
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
    const annotations = container.querySelectorAll('.annotation, [class*="annotation"]');
    
    annotations.forEach(ann => {
        // Add click handler for highlighting
        ann.addEventListener('click', function() {
            // Remove previous highlights
            container.querySelectorAll('.annotation-highlighted').forEach(el => {
                el.classList.remove('annotation-highlighted');
            });
            
            // Highlight this annotation
            this.classList.add('annotation-highlighted');
            
            // Scroll into view
            this.scrollIntoView({ behavior: 'smooth', block: 'center' });
        });
        
        // Add hover effect
        ann.addEventListener('mouseenter', function() {
            this.style.transform = 'scale(1.05)';
            this.style.zIndex = '10';
        });
        
        ann.addEventListener('mouseleave', function() {
            this.style.transform = 'scale(1)';
            this.style.zIndex = '1';
        });
    });
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
    
    // Toggle view
    const toggleView = document.getElementById('toggleTranscriptView');
    if (toggleView) {
        toggleView.addEventListener('click', toggleTranscriptView);
    }
    
    // Toggle feature summary
    const toggleSummary = document.getElementById('toggleFeatureSummary');
    if (toggleSummary) {
        toggleSummary.addEventListener('click', () => {
            const content = document.getElementById('featureSummaryContent');
            const toggleText = document.getElementById('summaryToggleText');
            if (content.style.display === 'none') {
                content.style.display = 'grid';
                toggleText.textContent = 'Hide';
            } else {
                content.style.display = 'none';
                toggleText.textContent = 'Show';
            }
        });
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

function renderTranscriptStats(summary) {
    const statsPanel = document.getElementById('transcriptStats');
    const statsContent = document.getElementById('statsContent');
    
    if (!statsPanel || !statsContent) return;
    
    const totalFeatures = Object.values(summary).reduce((sum, count) => sum + count, 0);
    const uniqueFeatureTypes = Object.keys(summary).length;
    const mostCommon = Object.entries(summary).sort((a, b) => b[1] - a[1])[0];
    
    statsContent.innerHTML = `
        <div class="stat-card p-4 bg-primary-50 rounded-lg">
            <div class="text-2xl font-bold text-primary-900">${totalFeatures}</div>
            <div class="text-sm text-primary-600 mt-1">Total Annotations</div>
        </div>
        <div class="stat-card p-4 bg-primary-50 rounded-lg">
            <div class="text-2xl font-bold text-primary-900">${uniqueFeatureTypes}</div>
            <div class="text-sm text-primary-600 mt-1">Feature Types</div>
        </div>
        <div class="stat-card p-4 bg-primary-50 rounded-lg">
            <div class="text-2xl font-bold text-primary-900">${mostCommon ? mostCommon[1] : 0}</div>
            <div class="text-sm text-primary-600 mt-1">Most Common</div>
            <div class="text-xs text-primary-500 mt-1">${mostCommon ? formatFeatureName(mostCommon[0]) : 'N/A'}</div>
        </div>
        <div class="stat-card p-4 bg-primary-50 rounded-lg">
            <div class="text-2xl font-bold text-primary-900">${(totalFeatures / uniqueFeatureTypes).toFixed(1)}</div>
            <div class="text-sm text-primary-600 mt-1">Avg per Type</div>
        </div>
    `;
    
    statsPanel.classList.remove('hidden');
}

function getFeatureCategory(featureType) {
    for (const [categoryName, category] of Object.entries(FEATURE_CATEGORIES)) {
        if (category.types.includes(featureType)) {
            return category;
        }
    }
    return FEATURE_CATEGORIES['General'];
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
        summaryEl.className = 'mt-4 p-4 bg-blue-50 rounded-lg border border-blue-200';
        
        const container = document.getElementById('annotatedTranscript').parentElement;
        container.appendChild(summaryEl);
    }
    
    const overallScore = (data.overall_coherence * 100).toFixed(1);
    const coherentCount = data.coherence_scores.filter(s => s.is_coherent === true).length;
    const incoherentCount = data.coherence_scores.filter(s => s.is_coherent === false).length;
    
    summaryEl.innerHTML = `
        <div class="flex items-center justify-between">
            <div>
                <h4 class="text-sm font-semibold text-primary-900 mb-2">Semantic Coherence Analysis</h4>
                <div class="flex gap-4 text-sm">
                    <span class="text-green-700">
                        <strong>${coherentCount}</strong> coherent transitions
                    </span>
                    <span class="text-red-700">
                        <strong>${incoherentCount}</strong> incoherent transitions
                    </span>
                    <span class="text-primary-700">
                        Overall: <strong>${overallScore}%</strong>
                    </span>
                </div>
            </div>
            <button onclick="clearSemanticCoherence()" class="px-3 py-1 text-xs bg-white text-primary-700 rounded hover:bg-primary-100 transition-colors">
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
            
            // Color coding: active speech (teal) vs pause/silence (light gray)
            if (isSpeech) {
                // Gradient from light to darker teal based on energy level
                const maxEnergy = waveformData.energyStats?.max || 1;
                const energyRatio = Math.min(energy / maxEnergy, 1);
                // Light teal for low energy speech, darker for high energy
                const r = Math.floor(94 + (energyRatio * 20)); // 94-114
                const g = Math.floor(234 - (energyRatio * 30)); // 234-204
                const b = Math.floor(212 - (energyRatio * 20)); // 212-192
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

// Test connection on load
setTimeout(testConnection, 500);
