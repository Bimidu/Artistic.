import React, { useState } from 'react';
import { useAuth } from '@hooks/useAuth';
import { useNavigate } from 'react-router-dom';
import { PredictionPage } from '@pages/UserMode/PredictionPage';
import { TrainingPage } from '@pages/Training/TrainingPage';

export const HomePage = () => {
    const { user, logout } = useAuth();
    const navigate = useNavigate();
    const [mode, setMode] = useState('user'); // 'user' or 'training'
    const [apiStatus, setApiStatus] = useState('Connected');
    const [statusColor, setStatusColor] = useState('bg-green-400');

    const handleLogout = async () => {
        await logout();
        navigate('/login');
    };

    return (
        <div className="bg-white min-h-screen font-sans antialiased">
            {/* Header */}
            <header className="bg-lime-950">
                <div className="max-w-7xl mx-auto px-12 py-4">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-6">
                            <div className="text-4xl text-white">Artistic</div>
                            <div className="text-lg text-white/70 hidden sm:block">ASD Detection System</div>
                        </div>

                        <div className="flex items-center gap-8">
                            {/* Mode Toggle */}
                            <div className="toggle-switch">
                                <div className={`toggle-option ${mode === 'user' ? 'active' : ''}`} onClick={() => setMode('user')}>
                                    User Mode
                                </div>
                                <div className={`toggle-option ${mode === 'training' ? 'active' : ''}`} onClick={() => setMode('training')}>
                                    Training Mode
                                </div>
                                <div
                                    className="toggle-slider"
                                    style={{
                                        width: 'calc(50% - 4px)',
                                        left: mode === 'user' ? '4px' : '50%',
                                    }}
                                />
                            </div>

                            {/* Status */}
                            <div className="flex items-center gap-3 text-base text-white/80">
                                <span className={`w-2.5 h-2.5 rounded-full ${statusColor} status-connected`}></span>
                                <span>{apiStatus}</span>
                            </div>

                            {/* User Menu */}
                            <div className="flex items-center gap-4">
                                <span className="text-white text-sm">{user?.full_name || 'User'}</span>
                                <button
                                    onClick={handleLogout}
                                    className="px-5 py-2 bg-white text-primary-900 rounded-xl hover:bg-primary-100 transition-colors"
                                >
                                    Logout
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </header>

            {/* Landing Section */}
            <div className="bg-gradient-to-b from-lime-900/20 to-white">
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
                            {/* Pragmatic & Conversational */}
                            <div className="bg-white rounded-2xl p-8 transition-shadow">
                                <div className="w-16 h-16 bg-lime-100 rounded-2xl flex items-center justify-center mb-6">
                                    <svg className="w-10 h-10 text-lime-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"></path>
                                    </svg>
                                </div>
                                <h3 className="text-xl font-medium text-lime-950 mb-3">Pragmatic & Conversational</h3>
                                <p className="text-sm text-lime-700 leading-relaxed">
                                    Analyzes turn-taking, topic maintenance, conversational repairs, and social communication patterns in dialogue
                                </p>
                            </div>

                            {/* Acoustic & Prosodic */}
                            <div className="bg-white rounded-2xl p-8 transition-shadow">
                                <div className="w-16 h-16 bg-lime-100 rounded-2xl flex items-center justify-center mb-6">
                                    <svg className="w-10 h-10 text-lime-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3"></path>
                                    </svg>
                                </div>
                                <h3 className="text-xl font-medium text-lime-950 mb-3">Acoustic & Prosodic</h3>
                                <p className="text-sm text-lime-700 leading-relaxed">
                                    Examines pitch variation, speech rhythm, intonation patterns, and vocal quality characteristics
                                </p>
                            </div>

                            {/* Syntactic & Semantic */}
                            <div className="bg-white rounded-2xl p-8 transition-shadow">
                                <div className="w-16 h-16 bg-lime-100 rounded-2xl flex items-center justify-center mb-6">
                                    <svg className="w-10 h-10 text-lime-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"></path>
                                    </svg>
                                </div>
                                <h3 className="text-xl font-medium text-lime-950 mb-3">Syntactic & Semantic</h3>
                                <p className="text-sm text-lime-700 leading-relaxed">
                                    Evaluates sentence structure complexity, grammatical patterns, and semantic coherence in language use
                                </p>
                            </div>

                            {/* Multi-Modal Fusion */}
                            <div className="bg-white rounded-2xl p-8 transition-shadow">
                                <div className="w-16 h-16 bg-lime-100 rounded-2xl flex items-center justify-center mb-6">
                                    <svg className="w-10 h-10 text-lime-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M4 5a1 1 0 011-1h4a1 1 0 011 1v7a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM14 5a1 1 0 011-1h4a1 1 0 011 1v7a1 1 0 01-1 1h-4a1 1 0 01-1-1V5zM4 16a1 1 0 011-1h4a1 1 0 011 1v3a1 1 0 01-1 1H5a1 1 0 01-1-1v-3zM14 16a1 1 0 011-1h4a1 1 0 011 1v3a1 1 0 01-1 1h-4a1 1 0 01-1-1v-3z"></path>
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
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
                                </svg>
                            </div>
                            <h4 className="text-lg font-medium text-lime-950 mb-2">Real-time Analysis</h4>
                            <p className="text-sm text-lime-700">Instant processing of audio files and text transcripts</p>
                        </div>

                        <div className="text-center">
                            <div className="w-12 h-12 bg-lime-100 rounded-xl flex items-center justify-center mx-auto mb-4">
                                <svg className="w-7 h-7 text-lime-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"></path>
                                </svg>
                            </div>
                            <h4 className="text-lg font-medium text-lime-950 mb-2">Explainable AI</h4>
                            <p className="text-sm text-lime-700">SHAP values and counterfactual explanations</p>
                        </div>

                        <div className="text-center">
                            <div className="w-12 h-12 bg-lime-100 rounded-xl flex items-center justify-center mx-auto mb-4">
                                <svg className="w-7 h-7 text-lime-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4"></path>
                                </svg>
                            </div>
                            <h4 className="text-lg font-medium text-lime-950 mb-2">Flexible Training</h4>
                            <p className="text-sm text-lime-700">Custom model training with multiple ML algorithms</p>
                        </div>
                    </div>
                </div>
            </div>

            {/* Mode Content */}
            <div className="max-w-7xl mx-auto px-12 py-12">
                {mode === 'user' && <PredictionPage />}
                {mode === 'training' && <TrainingPage />}
            </div>
        </div>
    );
};
