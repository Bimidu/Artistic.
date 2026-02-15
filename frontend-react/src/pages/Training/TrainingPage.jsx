import React, { useState, useEffect } from 'react';
import { predictionService } from '@services/predictionService';

export const TrainingPage = () => {
    const [activeTab, setActiveTab] = useState('feature-extraction');
    const [datasets, setDatasets] = useState([]);
    const [extractionDatasets, setExtractionDatasets] = useState([]);
    const [trainingDatasets, setTrainingDatasets] = useState([]);
    const [component, setComponent] = useState('pragmatic_conversational');
    const [trainingComponent, setTrainingComponent] = useState('pragmatic_conversational');
    const [maxSamples, setMaxSamples] = useState('');
    const [featureSelectionEnabled, setFeatureSelectionEnabled] = useState(true);
    const [nFeatures, setNFeatures] = useState(30);
    const [enableAutoencoder, setEnableAutoencoder] = useState(false);
    const [modelTypes, setModelTypes] = useState(['svm', 'logistic']);
    const [testSize, setTestSize] = useState(20);
    const [randomState, setRandomState] = useState(42);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [result, setResult] = useState(null);
    const [trainedModels, setTrainedModels] = useState([]);

    const loadDatasets = async () => {
        try {
            const data = await predictionService.loadDatasets();
            setDatasets(data.datasets || []);
        } catch (err) {
            console.error('Failed to load datasets:', err);
            setError('Failed to load datasets');
        }
    };

    const handleExtractFeatures = async () => {
        if (extractionDatasets.length === 0) {
            setError('Please select at least one dataset');
            return;
        }
        setLoading(true);
        setError('');
        setResult(null);
        try {
            const data = await predictionService.extractFeatures(extractionDatasets, component, `${component}_features.csv`, maxSamples || null);
            setResult(data);
        } catch (err) {
            setError(err.response?.data?.detail || 'Feature extraction failed');
        } finally {
            setLoading(false);
        }
    };

    const handleStartTraining = async () => {
        if (trainingDatasets.length === 0) {
            setError('Please select at least one dataset');
            return;
        }
        setLoading(true);
        setError('');
        setResult(null);
        try {
            const data = await predictionService.startTraining(trainingDatasets, trainingComponent, modelTypes, featureSelectionEnabled ? nFeatures : null);
            setResult(data);
        } catch (err) {
            setError(err.response?.data?.detail || 'Training failed');
        } finally {
            setLoading(false);
        }
    };

    const loadTrainedModels = async () => {
        try {
            const data = await predictionService.getAvailableModels();
            setTrainedModels(data.models || []);
        } catch (err) {
            console.error('Failed to load models:', err);
        }
    };

    const toggleDataset = (dataset, type) => {
        const setter = type === 'extraction' ? setExtractionDatasets : setTrainingDatasets;
        const current = type === 'extraction' ? extractionDatasets : trainingDatasets;
        setter(current.includes(dataset) ? current.filter(d => d !== dataset) : [...current, dataset]);
    };

    const componentOptions = [
        { value: 'pragmatic_conversational', label: 'Pragmatic & Conversational' },
        { value: 'acoustic_prosodic', label: 'Acoustic & Prosodic' },
        { value: 'syntactic_semantic', label: 'Syntactic & Semantic' },
    ];

    const modelTypeOptions = [
        { value: 'svm', label: 'SVM' },
        { value: 'logistic', label: 'Logistic Regression' },
        { value: 'random_forest', label: 'Random Forest' },
        { value: 'xgboost', label: 'XGBoost' },
        { value: 'lightgbm', label: 'LightGBM' },
    ];

    const tabs = [
        { key: 'feature-extraction', label: 'Feature Extraction' },
        { key: 'training', label: 'Training' },
        { key: 'trained-models', label: 'Trained Models' },
    ];

    return (
        <div>
            {/* Training mode tabs row */}
            <div className="bg-primary-100 border-b-2 border-primary-200 -mx-12">
                <div className="max-w-7xl mx-auto px-12">
                    <div className="flex gap-4">
                        {tabs.map(tab => (
                            <button
                                key={tab.key}
                                type="button"
                                onClick={() => { setActiveTab(tab.key); setError(''); setResult(null); }}
                                className={`px-8 py-5 text-lg border-b-2 transition-colors ${activeTab === tab.key
                                        ? 'border-primary-900 text-primary-900'
                                        : 'border-transparent text-primary-500 hover:text-primary-900'
                                    }`}
                            >
                                {tab.label}
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            {/* Error Display */}
            {error && (
                <div className="mt-8 bg-red-50 border-l-4 border-red-500 p-6 rounded">
                    <p className="text-red-700 text-base">{error}</p>
                </div>
            )}

            {/* Feature Extraction Tab */}
            {activeTab === 'feature-extraction' && (
                <div className="py-12">
                    <div className="mb-12 bg-primary-50 border-t-2 border-black rounded-b-3xl overflow-hidden">
                        <div className="px-10 py-8 bg-white flex items-center justify-between">
                            <h2 className="text-5xl text-primary-900">Feature Extraction</h2>
                            <button onClick={loadDatasets} className="px-6 py-3 bg-primary-900 text-white rounded-2xl text-base hover:bg-primary-800 transition-all">
                                Refresh
                            </button>
                        </div>

                        <div className="p-10">
                            <p className="text-lg text-primary-600 mb-6">
                                Select datasets from your file system to extract features. Extracted features will be saved to CSV files.
                            </p>

                            <div className="grid lg:grid-cols-2 gap-8">
                                {/* Dataset Selection */}
                                <div className="bg-white rounded-2xl p-6" style={{ maxHeight: '500px', overflowY: 'auto' }}>
                                    <h3 className="text-2xl text-primary-900 mb-4">Select Datasets to Extract</h3>
                                    {datasets.length === 0 ? (
                                        <div className="text-center py-16 text-primary-400 text-xl">
                                            Click Refresh to load datasets
                                        </div>
                                    ) : (
                                        <div className="space-y-2">
                                            {datasets.map(dataset => (
                                                <label key={dataset} className="flex items-center gap-3 p-3 hover:bg-primary-50 rounded-lg cursor-pointer">
                                                    <input type="checkbox" checked={extractionDatasets.includes(dataset)}
                                                        onChange={() => toggleDataset(dataset, 'extraction')}
                                                        className="w-5 h-5 text-primary-600 rounded" />
                                                    <span className="text-base text-primary-900">{dataset}</span>
                                                </label>
                                            ))}
                                        </div>
                                    )}
                                </div>

                                {/* Extraction Settings */}
                                <div className="bg-white rounded-2xl p-6">
                                    <h3 className="text-2xl text-primary-900 mb-4">Extraction Settings</h3>

                                    <div className="mb-6">
                                        <label className="block text-lg text-primary-900 mb-3">Component</label>
                                        <select value={component} onChange={(e) => setComponent(e.target.value)}
                                            className="w-full px-6 py-4 bg-primary-50 rounded-2xl text-base focus:outline-none focus:bg-primary-100 transition-all">
                                            {componentOptions.map(opt => (
                                                <option key={opt.value} value={opt.value}>{opt.label}</option>
                                            ))}
                                        </select>
                                        <p className="text-sm text-primary-500 mt-2">Select which component's features to extract</p>
                                    </div>

                                    <div className="mb-6">
                                        <label className="block text-lg text-primary-900 mb-3">Max Samples per Dataset</label>
                                        <input type="number" value={maxSamples} onChange={(e) => setMaxSamples(e.target.value)}
                                            min="1" placeholder="Leave empty for all samples"
                                            className="w-full px-6 py-4 bg-primary-50 rounded-2xl text-base focus:outline-none focus:bg-primary-100 transition-all" />
                                        <p className="text-sm text-primary-500 mt-2">Limit samples for large datasets (e.g., TD). Leave empty to extract all.</p>
                                    </div>

                                    <button onClick={handleExtractFeatures} disabled={extractionDatasets.length === 0 || loading}
                                        className="w-full px-8 py-5 bg-primary-900 text-white rounded-2xl text-xl hover:bg-primary-800 transition-all disabled:opacity-40 disabled:cursor-not-allowed">
                                        {loading ? 'Extracting...' : 'Extract Features'}
                                    </button>

                                    {result && (
                                        <div className="mt-6 bg-primary-50 rounded-xl p-4">
                                            <p className="text-primary-900 font-medium">Features extracted successfully!</p>
                                            {result.samples_processed && <p className="text-primary-700 text-sm mt-2">Samples: {result.samples_processed}</p>}
                                            {result.output_file && <p className="text-primary-700 text-sm mt-1">Output: {result.output_file}</p>}
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Training Tab */}
            {activeTab === 'training' && (
                <div className="py-12">
                    <div className="mb-12 bg-primary-50 border-t-2 border-black rounded-b-3xl overflow-hidden">
                        <div className="px-10 py-8 bg-white flex items-center justify-between">
                            <h2 className="text-5xl text-primary-900">Training</h2>
                            <button onClick={loadDatasets} className="px-6 py-3 bg-primary-900 text-white rounded-2xl text-base hover:bg-primary-800 transition-all">
                                Refresh
                            </button>
                        </div>

                        <div className="p-10">
                            <div className="grid lg:grid-cols-2 gap-8 items-stretch">
                                {/* Dataset Selection */}
                                <div className="bg-white rounded-2xl p-6 flex flex-col">
                                    <h3 className="text-2xl text-primary-900 mb-4 flex-shrink-0">Available Datasets (from CSV)</h3>
                                    <div className="flex-1 overflow-y-auto min-h-0">
                                        {datasets.length === 0 ? (
                                            <div className="text-center py-16 text-primary-400 text-xl">
                                                Click Refresh to load datasets
                                            </div>
                                        ) : (
                                            <div className="space-y-2">
                                                {datasets.map(dataset => (
                                                    <label key={dataset} className="flex items-center gap-3 p-3 hover:bg-primary-50 rounded-lg cursor-pointer">
                                                        <input type="checkbox" checked={trainingDatasets.includes(dataset)}
                                                            onChange={() => toggleDataset(dataset, 'training')}
                                                            className="w-5 h-5 text-primary-600 rounded" />
                                                        <span className="text-base text-primary-900">{dataset}</span>
                                                    </label>
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                </div>

                                {/* Training Controls */}
                                <div className="bg-white rounded-2xl p-6">
                                    <h3 className="text-2xl text-primary-900 mb-4">Training Controls</h3>

                                    {/* Component */}
                                    <div className="mb-8">
                                        <label className="block text-xl text-primary-900 mb-4">Component</label>
                                        <select value={trainingComponent} onChange={(e) => setTrainingComponent(e.target.value)}
                                            className="w-full px-6 py-4 bg-white rounded-2xl text-base focus:outline-none focus:bg-primary-100 transition-all">
                                            {componentOptions.map(opt => (
                                                <option key={opt.value} value={opt.value}>{opt.label}</option>
                                            ))}
                                        </select>
                                        <p className="text-sm text-primary-500 mt-2">Note: Acoustic & Syntactic use placeholder features for testing</p>
                                    </div>

                                    {/* Feature Selection */}
                                    <div className="mb-8">
                                        <label className="block text-xl text-primary-900 mb-4">Feature Selection</label>
                                        <div className="space-y-4">
                                            <label className="flex items-center cursor-pointer p-5 bg-white rounded-2xl hover:bg-primary-100 transition-colors">
                                                <input type="checkbox" checked={featureSelectionEnabled}
                                                    onChange={(e) => setFeatureSelectionEnabled(e.target.checked)}
                                                    className="w-5 h-5 text-primary-600 rounded" />
                                                <span className="ml-4 text-lg text-primary-900">Enable feature selection</span>
                                            </label>
                                            {featureSelectionEnabled && (
                                                <div>
                                                    <label className="block text-base text-primary-700 mb-2">Number of features to select</label>
                                                    <input type="number" value={nFeatures} onChange={(e) => setNFeatures(parseInt(e.target.value) || 30)}
                                                        min="1" max="218"
                                                        className="w-full px-6 py-4 bg-white rounded-2xl text-base focus:outline-none focus:bg-primary-100 transition-all" />
                                                    <p className="text-sm text-primary-500 mt-2">Default: 30 (max: 218 for pragmatic, 20 for others)</p>
                                                </div>
                                            )}
                                        </div>
                                    </div>

                                    {/* Counterfactual Explanations */}
                                    <div className="mb-8">
                                        <label className="block text-xl text-primary-900 mb-4">Counterfactual Explanations</label>
                                        <div className="space-y-4">
                                            <label className="flex items-center cursor-pointer p-5 bg-white rounded-2xl hover:bg-primary-100 transition-colors">
                                                <input type="checkbox" checked={enableAutoencoder}
                                                    onChange={(e) => setEnableAutoencoder(e.target.checked)}
                                                    className="w-5 h-5 text-primary-600 rounded" />
                                                <div className="ml-4 flex-1">
                                                    <span className="text-lg text-primary-900">Enable counterfactual autoencoder</span>
                                                    <p className="text-sm text-primary-500 mt-1">Train autoencoder for counterfactual explanations (may crash on macOS - disabled by default)</p>
                                                </div>
                                            </label>
                                        </div>
                                    </div>

                                    {/* Model Types */}
                                    <div className="mb-8">
                                        <label className="block text-xl text-primary-900 mb-4">Model Types</label>
                                        <p className="text-sm text-primary-500 mb-3">Available models change based on selected component</p>
                                        <div className="grid grid-cols-2 gap-3">
                                            {modelTypeOptions.map(opt => (
                                                <label key={opt.value} className="flex items-center cursor-pointer p-4 bg-white rounded-2xl hover:bg-primary-100 transition-colors">
                                                    <input type="checkbox" value={opt.value}
                                                        checked={modelTypes.includes(opt.value)}
                                                        onChange={(e) => {
                                                            if (e.target.checked) setModelTypes([...modelTypes, opt.value]);
                                                            else setModelTypes(modelTypes.filter(m => m !== opt.value));
                                                        }}
                                                        className="w-5 h-5 text-primary-600 rounded" />
                                                    <span className="ml-3 text-base text-primary-900">{opt.label}</span>
                                                </label>
                                            ))}
                                        </div>
                                    </div>

                                    {/* Training Parameters */}
                                    <div className="mb-8">
                                        <label className="block text-xl text-primary-900 mb-4">Training Parameters</label>
                                        <div className="space-y-4">
                                            <div className="bg-white rounded-2xl p-4">
                                                <label className="block text-sm text-primary-700 mb-2">Test Set Size (%)</label>
                                                <input type="number" value={testSize} onChange={(e) => setTestSize(parseInt(e.target.value) || 20)}
                                                    min="10" max="40" step="5"
                                                    className="w-full px-4 py-3 bg-primary-50 rounded-xl text-base focus:outline-none focus:bg-primary-100 transition-all" />
                                                <p className="text-xs text-primary-500 mt-1">Percentage of data reserved for testing</p>
                                            </div>
                                            <div className="bg-white rounded-2xl p-4">
                                                <label className="block text-sm text-primary-700 mb-2">Random Seed</label>
                                                <input type="number" value={randomState} onChange={(e) => setRandomState(parseInt(e.target.value) || 42)}
                                                    min="0" max="999"
                                                    className="w-full px-4 py-3 bg-primary-50 rounded-xl text-base focus:outline-none focus:bg-primary-100 transition-all" />
                                                <p className="text-xs text-primary-500 mt-1">For reproducible results</p>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Start Training Button */}
                                    <button onClick={handleStartTraining} disabled={trainingDatasets.length === 0 || loading}
                                        className="w-full px-8 py-5 bg-primary-900 text-white rounded-2xl text-xl hover:bg-primary-800 transition-all disabled:opacity-40 disabled:cursor-not-allowed">
                                        {loading ? 'Training...' : 'Start Training'}
                                    </button>

                                    {/* Training Result */}
                                    {result && (
                                        <div className="mt-8 bg-primary-50 rounded-xl p-4">
                                            {result.status === 'success' && (
                                                <p className="text-green-700 font-medium mb-2">Training completed successfully!</p>
                                            )}
                                            {result.best_model && (
                                                <p className="text-primary-900">
                                                    Best Model: {typeof result.best_model === 'string' ? result.best_model : (result.best_model.name || result.best_model.id || 'Model')}
                                                </p>
                                            )}
                                            {result.metrics && Object.entries(result.metrics).map(([key, value]) => {
                                                if (typeof value === 'object' && value !== null) return null;
                                                return (
                                                    <p key={key} className="text-primary-700 text-sm mt-1">
                                                        {key.replace('_', ' ')}: {typeof value === 'number' ? value.toFixed(4) : String(value)}
                                                    </p>
                                                );
                                            })}
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Trained Models Tab */}
            {activeTab === 'trained-models' && (
                <div className="py-12">
                    <div className="bg-primary-50 border-t-2 border-black rounded-b-3xl overflow-hidden">
                        <div className="px-10 py-8 bg-white flex items-center justify-between">
                            <h2 className="text-5xl text-primary-900">Trained Models</h2>
                            <button onClick={loadTrainedModels} className="px-6 py-3 bg-primary-900 text-white rounded-2xl text-base hover:bg-primary-800 transition-all">
                                Refresh
                            </button>
                        </div>

                        <div className="p-10">
                            {trainedModels.length === 0 ? (
                                <div className="text-center py-16 text-primary-400 text-xl">
                                    Click Refresh to load trained models
                                </div>
                            ) : (
                                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                                    {trainedModels.map((model, idx) => {
                                        const modelName = typeof model === 'string' ? model : (model.name || model.id || `Model ${idx + 1}`);
                                        const modelComponent = typeof model === 'object' ? model.component : '';
                                        const modelAccuracy = typeof model === 'object' ? model.accuracy : null;
                                        return (
                                            <div key={idx} className="model-card bg-white rounded-2xl p-6 border border-primary-200">
                                                <h4 className="text-xl font-medium text-primary-900 mb-2">{modelName}</h4>
                                                {modelComponent && <p className="text-sm text-primary-600 mb-2">Component: {modelComponent}</p>}
                                                {modelAccuracy != null && (
                                                    <p className="text-sm text-primary-700">
                                                        Accuracy: {typeof modelAccuracy === 'number' ? (modelAccuracy * 100).toFixed(1) + '%' : String(modelAccuracy)}
                                                    </p>
                                                )}
                                            </div>
                                        );
                                    })}
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}

            {/* Loading Overlay */}
            {loading && (
                <div className="fixed inset-0 bg-black bg-opacity-20 flex items-center justify-center z-50">
                    <div className="bg-white rounded-2xl p-8 shadow-xl">
                        <div className="spinner w-12 h-12 mx-auto mb-4"></div>
                        <p className="text-primary-700 text-lg">Processing...</p>
                    </div>
                </div>
            )}
        </div>
    );
};
