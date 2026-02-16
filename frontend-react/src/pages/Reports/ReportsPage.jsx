import React, { useState, useEffect, useRef } from 'react';
import { useAuth } from '@hooks/useAuth';
import { useNavigate } from 'react-router-dom';
import { reportService } from '@services/reportService';

export const ReportsPage = () => {
    const { user } = useAuth();
    const navigate = useNavigate();
    const [reportsByPatient, setReportsByPatient] = useState([]);
    const [expandedPatients, setExpandedPatients] = useState(new Set());
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');
    const [dropdownOpen, setDropdownOpen] = useState(false);
    const dropdownRef = useRef(null);

    // Close dropdown when clicking outside
    useEffect(() => {
        const handleClickOutside = (event) => {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
                setDropdownOpen(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    useEffect(() => {
        loadReports();
    }, []);

    const loadReports = async () => {
        try {
            setLoading(true);
            const data = await reportService.getReportsByPatient();
            setReportsByPatient(data);
            setError('');
        } catch (err) {
            console.error('Failed to load reports:', err);
            setError('Failed to load reports');
        } finally {
            setLoading(false);
        }
    };

    const togglePatient = (patientName) => {
        const newExpanded = new Set(expandedPatients);
        if (newExpanded.has(patientName)) {
            newExpanded.delete(patientName);
        } else {
            newExpanded.add(patientName);
        }
        setExpandedPatients(newExpanded);
    };

    const formatDate = (dateString) => {
        const date = new Date(dateString);
        return date.toLocaleDateString('en-US', {
            month: 'short',
            day: 'numeric',
            year: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    };

    const getPredictionColor = (prediction) => {
        return prediction === 'ASD' ? 'text-red-600' : 'text-green-600';
    };

    const handleLogout = async () => {
        await logout();
        navigate('/login');
    };

    return (
        <div className="bg-white min-h-screen font-sans antialiased">
            {/* Header */}
            <header className="bg-lime-950 border-b border-lime-800">
                <div className="max-w-7xl mx-auto px-12 py-4">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-4">
                            <button
                                onClick={() => navigate('/')}
                                className="text-white hover:text-lime-200 transition-colors"
                            >
                                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                                </svg>
                            </button>
                            <h1 className="text-3xl text-white font-normal">My Reports</h1>
                        </div>

                        {/* Profile Icon Toggle */}
                        <div className="relative" ref={dropdownRef}>
                            <button
                                onClick={() => setDropdownOpen(!dropdownOpen)}
                                className="w-12 h-12 bg-lime-700 rounded-full flex items-center justify-center text-xl font-medium text-white hover:bg-lime-600 transition-colors"
                                aria-label="Toggle profile menu"
                            >
                                {user?.full_name?.charAt(0).toUpperCase() || 'U'}
                            </button>

                            {/* Dropdown Menu */}
                            {dropdownOpen && (
                                <div className="absolute right-0 mt-3 w-80 bg-white rounded-2xl shadow-xl border border-lime-200 overflow-hidden z-50">
                                    {/* Profile Section */}
                                    <div className="p-6 bg-gradient-to-br from-lime-950 to-lime-900 text-white">
                                        <div className="flex items-center gap-4">
                                            <div className="w-14 h-14 bg-lime-700 rounded-full flex items-center justify-center text-2xl font-medium ring-2 ring-white/20">
                                                {user?.full_name?.charAt(0).toUpperCase() || 'U'}
                                            </div>
                                            <div className="flex-1">
                                                <h3 className="font-medium text-lg">{user?.full_name || 'User'}</h3>
                                                <p className="text-sm text-lime-200/90">{user?.email}</p>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Menu Items */}
                                    <div className="p-3">
                                        <button
                                            onClick={() => {
                                                setDropdownOpen(false);
                                            }}
                                            className="w-full px-4 py-3 text-left rounded-xl bg-lime-50 transition-colors flex items-center gap-3 text-lime-950 group cursor-default"
                                        >
                                            <div className="w-10 h-10 rounded-lg bg-lime-200 flex items-center justify-center">
                                                <svg className="w-5 h-5 text-lime-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                                </svg>
                                            </div>
                                            <span className="font-medium">My Reports</span>
                                        </button>

                                        <div className="my-2 border-t border-lime-100"></div>

                                        <button
                                            onClick={handleLogout}
                                            className="w-full px-4 py-3 text-left rounded-xl hover:bg-red-50 transition-colors flex items-center gap-3 text-red-600 group"
                                        >
                                            <div className="w-10 h-10 rounded-lg bg-red-100 group-hover:bg-red-200 flex items-center justify-center transition-colors">
                                                <svg className="w-5 h-5 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                                                </svg>
                                            </div>
                                            <span className="font-medium">Logout</span>
                                        </button>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </header>

            {/* Main Content */}
            <div className="max-w-7xl mx-auto px-12 py-12">
                {loading && (
                    <div className="text-center py-12">
                        <div className="spinner mx-auto mb-4"></div>
                        <p className="text-lime-700">Loading reports...</p>
                    </div>
                )}

                {error && (
                    <div className="bg-red-50 border-l-4 border-red-500 p-6 mb-6">
                        <p className="text-red-700">{error}</p>
                    </div>
                )}

                {!loading && !error && reportsByPatient.length === 0 && (
                    <div className="text-center py-12">
                        <svg className="w-24 h-24 mx-auto mb-6 text-lime-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                        </svg>
                        <h2 className="text-2xl text-lime-950 mb-3">No Reports Yet</h2>
                        <p className="text-lime-700 mb-6">Run an analysis and save it to see it here.</p>
                        <button
                            onClick={() => navigate('/')}
                            className="px-6 py-3 bg-lime-950 text-white rounded-xl hover:bg-lime-800 transition-colors"
                        >
                            Go to Analysis
                        </button>
                    </div>
                )}

                {!loading && !error && reportsByPatient.length > 0 && (
                    <div className="space-y-4">
                        {reportsByPatient.map((patientGroup) => (
                            <div key={patientGroup.patient_name} className="bg-white rounded-2xl overflow-hidden border border-lime-200 shadow-sm">
                                {/* Patient Header - Accordion Toggle */}
                                <button
                                    onClick={() => togglePatient(patientGroup.patient_name)}
                                    className="w-full px-6 py-4 flex items-center justify-between hover:bg-lime-50 transition-colors"
                                >
                                    <div className="flex items-center gap-4">
                                        <svg
                                            className={`w-6 h-6 text-lime-700 transition-transform ${expandedPatients.has(patientGroup.patient_name) ? 'rotate-90' : ''
                                                }`}
                                            fill="none"
                                            stroke="currentColor"
                                            viewBox="0 0 24 24"
                                        >
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                                        </svg>
                                        <span className="text-xl font-medium text-lime-950">{patientGroup.patient_name}</span>
                                    </div>
                                    <span className="text-base text-lime-600 bg-lime-100 px-4 py-2 rounded-full">
                                        {patientGroup.report_count} {patientGroup.report_count === 1 ? 'report' : 'reports'}
                                    </span>
                                </button>

                                {/* Patient Reports - Collapsible */}
                                {expandedPatients.has(patientGroup.patient_name) && (
                                    <div className="border-t border-lime-200 bg-lime-50">
                                        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4 p-6">
                                            {patientGroup.reports.map((report) => (
                                                <div
                                                    key={report.report_id}
                                                    className="bg-white rounded-xl p-5 border border-lime-200 hover:shadow-md transition-shadow"
                                                >
                                                    <div className="flex justify-between items-start mb-3">
                                                        <span className={`text-lg font-semibold ${getPredictionColor(report.prediction)}`}>
                                                            {report.prediction}
                                                        </span>
                                                        <span className="text-xs text-lime-600 bg-lime-100 px-2 py-1 rounded">
                                                            {formatDate(report.analysis_date)}
                                                        </span>
                                                    </div>
                                                    <div className="text-sm text-lime-700 space-y-2">
                                                        <div className="flex justify-between">
                                                            <span>Confidence:</span>
                                                            <span className="font-semibold">{(report.confidence * 100).toFixed(1)}%</span>
                                                        </div>
                                                        <div className="flex justify-between">
                                                            <span>Input Type:</span>
                                                            <span className="font-semibold capitalize">{report.input_type}</span>
                                                        </div>
                                                        <div className="flex justify-between">
                                                            <span>Model:</span>
                                                            <span className="font-semibold text-xs">{report.model_used}</span>
                                                        </div>
                                                        {report.features_extracted && (
                                                            <div className="flex justify-between">
                                                                <span>Features:</span>
                                                                <span className="font-semibold">{report.features_extracted}</span>
                                                            </div>
                                                        )}
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
};
