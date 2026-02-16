import React, { useState, useEffect } from 'react';
import { useAuth } from '@hooks/useAuth';
import { useNavigate } from 'react-router-dom';
import { reportService } from '@services/reportService';

export const Sidebar = ({ isOpen, onToggle }) => {
    const { user, logout } = useAuth();
    const navigate = useNavigate();
    const [reportsByPatient, setReportsByPatient] = useState([]);
    const [expandedPatients, setExpandedPatients] = useState(new Set());
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

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

    const handleLogout = () => {
        logout();
        navigate('/login');
    };

    return (
        <>
            {/* Sidebar */}
            <div
                className={`fixed right-0 top-0 h-full bg-lime-50 border-l border-lime-200 shadow-2xl transform transition-transform duration-300 ease-in-out z-40 ${isOpen ? 'translate-x-0' : 'translate-x-full'
                    } w-96 flex flex-col`}
            >
                {/* Profile Section - Top */}
                <div className="p-6 bg-lime-950 text-white flex-shrink-0 relative">
                    {/* Close Button */}
                    <button
                        onClick={onToggle}
                        className="absolute top-4 right-4 w-8 h-8 flex items-center justify-center rounded-full hover:bg-lime-800 transition-colors"
                        aria-label="Close sidebar"
                    >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                    </button>

                    <div className="flex items-center gap-4">
                        <div className="w-12 h-12 bg-lime-700 rounded-full flex items-center justify-center text-xl font-medium">
                            {user?.full_name?.charAt(0).toUpperCase() || 'U'}
                        </div>
                        <div>
                            <h3 className="font-medium text-lg">{user?.full_name || 'User'}</h3>
                            <p className="text-sm text-lime-200">{user?.email}</p>
                        </div>
                    </div>
                </div>

                {/* My Reports Section - Middle (Scrollable) */}
                <div className="flex-1 overflow-y-auto p-6">
                    <h2 className="text-2xl font-medium text-lime-950 mb-4">My Reports</h2>

                    {loading && (
                        <div className="text-center py-8 text-lime-700">
                            Loading reports...
                        </div>
                    )}

                    {error && (
                        <div className="bg-red-50 border-l-4 border-red-500 p-4 mb-4">
                            <p className="text-red-700 text-sm">{error}</p>
                        </div>
                    )}

                    {!loading && !error && reportsByPatient.length === 0 && (
                        <div className="text-center py-8 text-lime-600">
                            No reports yet. Run an analysis and save it to see it here.
                        </div>
                    )}

                    {!loading && !error && reportsByPatient.length > 0 && (
                        <div className="space-y-3">
                            {reportsByPatient.map((patientGroup) => (
                                <div key={patientGroup.patient_name} className="bg-white rounded-xl overflow-hidden border border-lime-200">
                                    {/* Patient Header - Accordion Toggle */}
                                    <button
                                        onClick={() => togglePatient(patientGroup.patient_name)}
                                        className="w-full px-4 py-3 flex items-center justify-between hover:bg-lime-50 transition-colors"
                                    >
                                        <div className="flex items-center gap-3">
                                            <svg
                                                className={`w-5 h-5 text-lime-700 transition-transform ${expandedPatients.has(patientGroup.patient_name) ? 'rotate-90' : ''
                                                    }`}
                                                fill="none"
                                                stroke="currentColor"
                                                viewBox="0 0 24 24"
                                            >
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                                            </svg>
                                            <span className="font-medium text-lime-950">{patientGroup.patient_name}</span>
                                        </div>
                                        <span className="text-sm text-lime-600 bg-lime-100 px-3 py-1 rounded-full">
                                            {patientGroup.report_count} {patientGroup.report_count === 1 ? 'report' : 'reports'}
                                        </span>
                                    </button>

                                    {/* Patient Reports - Collapsible */}
                                    {expandedPatients.has(patientGroup.patient_name) && (
                                        <div className="border-t border-lime-200">
                                            {patientGroup.reports.map((report) => (
                                                <div
                                                    key={report.report_id}
                                                    className="px-4 py-3 border-b border-lime-100 last:border-b-0 hover:bg-lime-50 transition-colors"
                                                >
                                                    <div className="flex justify-between items-start mb-2">
                                                        <span className={`font-medium ${getPredictionColor(report.prediction)}`}>
                                                            {report.prediction}
                                                        </span>
                                                        <span className="text-xs text-lime-600">
                                                            {formatDate(report.analysis_date)}
                                                        </span>
                                                    </div>
                                                    <div className="text-sm text-lime-700 space-y-1">
                                                        <div className="flex justify-between">
                                                            <span>Confidence:</span>
                                                            <span className="font-medium">{(report.confidence * 100).toFixed(1)}%</span>
                                                        </div>
                                                        <div className="flex justify-between">
                                                            <span>Input:</span>
                                                            <span className="font-medium capitalize">{report.input_type}</span>
                                                        </div>
                                                        {report.features_extracted && (
                                                            <div className="flex justify-between">
                                                                <span>Features:</span>
                                                                <span className="font-medium">{report.features_extracted}</span>
                                                            </div>
                                                        )}
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                {/* Logout Section - Bottom */}
                <div className="p-6 border-t border-lime-200 bg-white flex-shrink-0">
                    <button
                        onClick={handleLogout}
                        className="w-full px-6 py-3 bg-lime-950 text-white rounded-xl hover:bg-lime-800 transition-colors flex items-center justify-center gap-2"
                    >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                        </svg>
                        Logout
                    </button>
                </div>
            </div>
        </>
    );
};
