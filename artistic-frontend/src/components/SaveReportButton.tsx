'use client';

import React, { useState } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { saveReport, type ReportData } from '@/utils/reportService';

interface SaveReportButtonProps {
  reportData: ReportData | null;
  onSaveSuccess?: () => void;
}

export default function SaveReportButton({ reportData, onSaveSuccess }: SaveReportButtonProps) {
  const { user, token } = useAuth();
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState('');
  const [showPatientNameModal, setShowPatientNameModal] = useState(false);
  const [patientName, setPatientName] = useState('');

  if (!reportData) return null;

  const handleSaveClick = () => {
    if (!user) {
      alert('Please log in to save reports');
      return;
    }

    setShowPatientNameModal(true);
  };

  const handleSaveReport = async () => {
    if (!token || !patientName.trim()) {
      setError('Please enter a patient name');
      return;
    }

    setSaving(true);
    setError('');

    try {
      await saveReport(token, {
        ...reportData,
        patient_name: patientName.trim(),
      });

      setSaved(true);
      setShowPatientNameModal(false);
      setPatientName('');

      setTimeout(() => setSaved(false), 3000);

      if (onSaveSuccess) {
        onSaveSuccess();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save report');
    } finally {
      setSaving(false);
    }
  };

  return (
    <>
      <button
        onClick={handleSaveClick}
        disabled={saving || saved}
        className={`px-6 py-3 rounded-xl text-sm font-medium transition-all ${
          saved
            ? 'bg-green-100 text-green-700 border border-green-200'
            : 'bg-primary-900 text-white hover:bg-primary-800 disabled:opacity-50 disabled:cursor-not-allowed'
        }`}
      >
        {saved ? (
          <span className="flex items-center gap-2">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
            </svg>
            Report Saved
          </span>
        ) : (
          <span className="flex items-center gap-2">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3 3m0 0l-3-3m3 3V4" />
            </svg>
            Save Report
          </span>
        )}
      </button>

      {/* Patient Name Modal */}
      {showPatientNameModal && (
        <div
          className="fixed inset-0 backdrop-blur-sm flex items-center justify-center z-50"
          onClick={() => setShowPatientNameModal(false)}
        >
          <div
            className="bg-white rounded-2xl max-w-md w-full mx-4 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="px-8 py-6 border-b border-primary-100">
              <h3 className="text-xl font-normal text-primary-900" style={{ letterSpacing: '-0.02em' }}>
                Save Analysis Report
              </h3>
            </div>

            <div className="px-8 py-6">
              <label htmlFor="patientName" className="block text-sm font-medium text-primary-700 mb-2">
                Patient Name / ID
              </label>
              <input
                type="text"
                id="patientName"
                value={patientName}
                onChange={(e) => setPatientName(e.target.value)}
                className="w-full px-4 py-3 bg-white border border-primary-200 rounded-xl text-sm focus:outline-none focus:border-primary-400 transition-all"
                placeholder="e.g., Patient 001 or John Doe"
                autoFocus
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && patientName.trim()) {
                    handleSaveReport();
                  }
                }}
              />
              <p className="mt-2 text-xs text-primary-500">
                Enter an identifier to help you find this report later
              </p>

              {error && (
                <div className="mt-4 px-4 py-3 bg-red-50 border border-red-200 rounded-xl">
                  <p className="text-sm text-red-600">{error}</p>
                </div>
              )}

              <div className="mt-6 flex gap-3">
                <button
                  onClick={() => setShowPatientNameModal(false)}
                  className="flex-1 px-4 py-3 bg-white border border-primary-200 text-primary-700 rounded-xl text-sm font-medium hover:bg-primary-50 transition-all"
                >
                  Cancel
                </button>
                <button
                  onClick={handleSaveReport}
                  disabled={saving || !patientName.trim()}
                  className="flex-1 px-4 py-3 bg-primary-900 text-white rounded-xl text-sm font-medium hover:bg-primary-800 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {saving ? 'Saving...' : 'Save'}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
