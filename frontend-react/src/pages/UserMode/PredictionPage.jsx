import React, { useState, useRef } from 'react';
import { predictionService } from '@services/predictionService';

export const PredictionPage = () => {
    const [activeTab, setActiveTab] = useState('audio');
    const [audioFile, setAudioFile] = useState(null);
    const [textInput, setTextInput] = useState('');
    const [chatFile, setChatFile] = useState(null);
    const [participantId, setParticipantId] = useState('CHI');
    const [useFusion, setUseFusion] = useState(false);
    const [selectedModel, setSelectedModel] = useState('');
    const [models, setModels] = useState([]);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState('');

    const audioInputRef = useRef(null);
    const chatInputRef = useRef(null);

    React.useEffect(() => {
        loadModels();
    }, []);

    const loadModels = async () => {
        try {
            const data = await predictionService.getAvailableModels();
            setModels(data.models || []);
        } catch (err) {
            console.error('Failed to load models:', err);
        }
    };

    const handleAudioFileChange = (e) => {
        const file = e.target.files?.[0];
        if (file) setAudioFile(file);
    };

    const handleChatFileChange = (e) => {
        const file = e.target.files?.[0];
        if (file) setChatFile(file);
    };

    const handlePredict = async (type) => {
        setLoading(true);
        setError('');
        setResult(null);

        try {
            let data;
            if (type === 'audio') {
                if (!audioFile) { setError('Please select an audio file'); setLoading(false); return; }
                data = await predictionService.predictFromAudio(audioFile, participantId, selectedModel, useFusion);
            } else if (type === 'text') {
                if (!textInput.trim()) { setError('Please enter some text'); setLoading(false); return; }
                data = await predictionService.predictFromText(textInput, participantId, selectedModel, useFusion);
            } else {
                if (!chatFile) { setError('Please select a CHAT file'); setLoading(false); return; }
                data = await predictionService.predictFromChatFile(chatFile, participantId, selectedModel, useFusion);
            }
            setResult(data);
        } catch (err) {
            setError(err.response?.data?.detail || 'Prediction failed');
        } finally {
            setLoading(false);
        }
    };

    const ModelSelect = () => (
        <div className="mt-8" id="modelSelectContainer">
            <label className="block text-xl text-primary-900 mb-4">Model Selection</label>
            <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                disabled={useFusion}
                className="w-full px-6 py-4 bg-white rounded-2xl text-base focus:outline-none focus:bg-primary-100 transition-all disabled:opacity-40"
            >
                <option value="">Best Model (Auto)</option>
                {models.map((model) => {
                    const modelId = typeof model === 'string' ? model : (model.name || model.id || JSON.stringify(model));
                    const modelLabel = typeof model === 'string' ? model : (model.name || model.id || 'Unknown Model');
                    return <option key={modelId} value={modelId}>{modelLabel}</option>;
                })}
            </select>
            <p className="text-sm text-primary-500 mt-2">Leave as "Best Model" to automatically select the best performing model</p>
            {useFusion && <p className="text-sm text-primary-400 mt-1">Model selection is disabled when fusion is enabled</p>}
        </div>
    );

    const FusionCheckbox = ({ id }) => (
        <div className="mt-6 flex items-center gap-3">
            <input type="checkbox" id={id} checked={useFusion} onChange={(e) => setUseFusion(e.target.checked)} className="w-5 h-5 text-primary-600 rounded" />
            <label htmlFor={id} className="text-base text-primary-700">Use multi-component fusion (if available)</label>
        </div>
    );

    return (
        <div className="grid lg:grid-cols-2 gap-12">
            {/* Input Section */}
            <div className="bg-primary-50 border-t-2 border-black rounded-b-3xl overflow-hidden">
                <div className="px-10 py-8 bg-white">
                    <h2 className="text-5xl text-primary-900">Analyze Speech</h2>
                </div>

                {/* Tabs */}
                <div className="bg-primary-100">
                    <div className="flex px-10 gap-4">
                        {[
                            { key: 'audio', label: 'Audio Upload' },
                            { key: 'text', label: 'Text Input' },
                            { key: 'chat', label: 'CHAT File' },
                        ].map((tab) => (
                            <button
                                key={tab.key}
                                onClick={() => setActiveTab(tab.key)}
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

                <div className="p-10">
                    {/* Audio Upload Panel */}
                    {activeTab === 'audio' && (
                        <div>
                            <div onClick={() => audioInputRef.current?.click()} className="upload-area bg-white rounded-3xl p-16 text-center cursor-pointer">
                                {audioFile ? (
                                    <div>
                                        <p className="text-2xl text-primary-900 font-medium">{audioFile.name}</p>
                                        <p className="text-base text-primary-500 mt-3">Click to change</p>
                                    </div>
                                ) : (
                                    <div>
                                        <svg className="mx-auto h-20 w-20 text-primary-400" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                                            <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                                        </svg>
                                        <p className="mt-6 text-2xl text-primary-900">Drop audio file here</p>
                                        <p className="mt-3 text-base text-primary-500">Supports WAV, MP3, FLAC</p>
                                    </div>
                                )}
                            </div>
                            <input ref={audioInputRef} type="file" accept=".wav,.mp3,.flac" onChange={handleAudioFileChange} className="hidden" />

                            <div className="mt-8">
                                <label className="block text-xl text-primary-900 mb-4">Participant ID <span className="text-primary-500">(optional)</span></label>
                                <input type="text" value={participantId} onChange={(e) => setParticipantId(e.target.value)} placeholder="CHI"
                                    className="w-full px-6 py-4 bg-white rounded-2xl text-base focus:outline-none focus:bg-primary-100 transition-all" />
                            </div>

                            <FusionCheckbox id="audioUseFusion" />
                            <ModelSelect />

                            <button onClick={() => handlePredict('audio')} disabled={!audioFile || loading}
                                className="mt-8 w-full px-8 py-5 bg-primary-900 text-white rounded-2xl text-xl hover:bg-primary-800 transition-all disabled:opacity-40 disabled:cursor-not-allowed">
                                {loading ? 'Analyzing...' : 'Analyze Audio'}
                            </button>
                        </div>
                    )}

                    {/* Text Input Panel */}
                    {activeTab === 'text' && (
                        <div>
                            <label className="block text-xl text-primary-900 mb-4">Speech Transcript</label>
                            <textarea value={textInput} onChange={(e) => setTextInput(e.target.value)}
                                placeholder="Enter or paste the speech transcript here..." rows={10}
                                className="w-full px-6 py-4 bg-white rounded-2xl text-base focus:outline-none focus:bg-primary-100 transition-all resize-none" />

                            <FusionCheckbox id="textUseFusion" />
                            <ModelSelect />

                            <button onClick={() => handlePredict('text')} disabled={!textInput.trim() || loading}
                                className="mt-8 w-full px-8 py-5 bg-primary-900 text-white rounded-2xl text-xl hover:bg-primary-800 transition-all disabled:opacity-40 disabled:cursor-not-allowed">
                                {loading ? 'Analyzing...' : 'Analyze Text'}
                            </button>
                        </div>
                    )}

                    {/* CHAT File Panel */}
                    {activeTab === 'chat' && (
                        <div>
                            <div onClick={() => chatInputRef.current?.click()} className="upload-area bg-white rounded-3xl p-16 text-center cursor-pointer">
                                {chatFile ? (
                                    <div>
                                        <p className="text-2xl text-primary-900 font-medium">{chatFile.name}</p>
                                        <p className="text-base text-primary-500 mt-3">Click to change</p>
                                    </div>
                                ) : (
                                    <div>
                                        <svg className="mx-auto h-20 w-20 text-primary-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                        </svg>
                                        <p className="mt-6 text-2xl text-primary-900">Drop CHAT file here</p>
                                        <p className="mt-3 text-base text-primary-500">Supports .cha files</p>
                                    </div>
                                )}
                            </div>
                            <input ref={chatInputRef} type="file" accept=".cha,.CHA,text/plain" onChange={handleChatFileChange} className="hidden" />

                            <FusionCheckbox id="chaUseFusion" />
                            <ModelSelect />

                            <button onClick={() => handlePredict('chat')} disabled={!chatFile || loading}
                                className="mt-8 w-full px-8 py-5 bg-primary-900 text-white rounded-2xl text-xl hover:bg-primary-800 transition-all disabled:opacity-40 disabled:cursor-not-allowed">
                                {loading ? 'Analyzing...' : 'Analyze File'}
                            </button>
                        </div>
                    )}
                </div>
            </div>

            {/* Results Section */}
            <div className="bg-primary-50 border-t-2 border-black rounded-b-3xl overflow-hidden">
                <div className="px-10 py-8 bg-white">
                    <h2 className="text-5xl text-primary-900">Analysis Results</h2>
                </div>

                <div className="p-10">
                    {error && (
                        <div className="bg-red-50 border-l-4 border-red-500 p-6 rounded mb-6">
                            <p className="text-red-700 text-base">{error}</p>
                        </div>
                    )}

                    {!result && !error && !loading && (
                        <div className="text-center py-24 text-primary-400 text-xl">
                            Upload an audio file or enter text to see results
                        </div>
                    )}

                    {loading && (
                        <div className="text-center py-24">
                            <div className="spinner w-16 h-16 mx-auto mb-4"></div>
                            <p className="text-primary-700 text-xl">Analyzing...</p>
                        </div>
                    )}

                    {result && (
                        <div>
                            {/* Prediction Result */}
                            <div className={`p-8 rounded-2xl mb-6 ${result.prediction === 'ASD' ? 'bg-red-50 border-2 border-red-200' : 'bg-green-50 border-2 border-green-200'}`}>
                                <div className="text-center">
                                    <p className="text-base text-primary-600 mb-2">Prediction</p>
                                    <p className={`text-5xl font-medium ${result.prediction === 'ASD' ? 'text-red-700' : 'text-green-700'}`}>
                                        {result.prediction}
                                    </p>
                                    <p className="text-2xl mt-4 text-primary-700">
                                        Confidence: {(result.confidence * 100).toFixed(1)}%
                                    </p>
                                </div>
                            </div>

                            {/* Probabilities */}
                            {result.probabilities && (
                                <div className="mb-6">
                                    <h3 className="text-2xl font-medium text-primary-900 mb-4">Class Probabilities</h3>
                                    {Object.entries(result.probabilities).map(([cls, prob]) => (
                                        <div key={cls} className="mb-4">
                                            <div className="flex justify-between mb-2">
                                                <span className="text-lg text-primary-700">{cls}</span>
                                                <span className="text-lg text-primary-900 font-medium">{(prob * 100).toFixed(1)}%</span>
                                            </div>
                                            <div className="w-full bg-primary-200 rounded-full h-3">
                                                <div className="bg-primary-700 h-3 rounded-full transition-all" style={{ width: `${prob * 100}%` }}></div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}

                            {/* Model Used */}
                            <div className="text-base text-primary-600 space-y-1">
                                <p>Model: {result.model_used || result.models_used?.join(', ')}</p>
                                {result.features_extracted && <p>Features Extracted: {result.features_extracted}</p>}
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};
