'use client';

import React, { useState, useEffect } from 'react';
import { useAuth } from '@/contexts/AuthContext';

interface Report {
  report_id: string;
  patient_name: string;
  analysis_date: string;
  prediction: string;
  confidence: number;
  model_used: string;
  input_type: string;
}

interface ProfileSidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function ProfileSidebar({ isOpen, onClose }: ProfileSidebarProps) {
  const { user, logout, token } = useAuth();
  const [reports, setReports] = useState<Report[]>([]);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'profile' | 'reports'>('profile');

  const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

  useEffect(() => {
    if (isOpen && token && activeTab === 'reports') {
      fetchReports();
    }
  }, [isOpen, token, activeTab]);

  const fetchReports = async () => {
    if (!token) return;

    setLoading(true);
    try {
      const response = await fetch(`${apiUrl}/api/reports/my-reports`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setReports(data);
      }
    } catch (error) {
      console.error('Failed to fetch reports:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    logout();
    onClose();
  };

  if (!isOpen) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 backdrop-blur-sm z-40 transition-opacity"
        onClick={onClose}
      />

      {/* Sidebar */}
      <div className="fixed right-0 top-0 h-full w-full max-w-md bg-white shadow-2xl z-50 overflow-hidden flex flex-col">
        {/* Header */}
        <div className="px-6 py-5 border-b border-primary-100 flex items-center justify-between bg-white">
          <h2 className="text-xl font-normal text-primary-900" style={{ letterSpacing: '-0.02em' }}>
            {activeTab === 'profile' ? 'Profile' : 'My Reports'}
          </h2>
          <button
            onClick={onClose}
            className="text-primary-400 hover:text-primary-900 transition-colors p-1"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-primary-100 bg-primary-50">
          <button
            onClick={() => setActiveTab('profile')}
            className={`flex-1 px-6 py-3 text-sm font-medium transition-colors ${
              activeTab === 'profile'
                ? 'text-primary-900 border-b-2 border-primary-900 bg-white'
                : 'text-primary-500 hover:text-primary-700'
            }`}
          >
            Profile
          </button>
          <button
            onClick={() => setActiveTab('reports')}
            className={`flex-1 px-6 py-3 text-sm font-medium transition-colors ${
              activeTab === 'reports'
                ? 'text-primary-900 border-b-2 border-primary-900 bg-white'
                : 'text-primary-500 hover:text-primary-700'
            }`}
          >
            Reports ({reports.length})
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto">
          {activeTab === 'profile' ? (
            <div className="p-6 space-y-6">
              {/* User Info */}
              <div className="flex items-center gap-4">
                {user?.avatar_url && (
                  <img
                    src={user.avatar_url}
                    alt={user.full_name}
                    className="w-16 h-16 rounded-full border-2 border-primary-200"
                  />
                )}
                <div className="flex-1">
                  <h3 className="text-lg font-medium text-primary-900">{user?.full_name}</h3>
                  <p className="text-sm text-primary-500">{user?.email}</p>
                </div>
              </div>

              {/* Account Details */}
              <div className="bg-primary-50 border border-primary-100 rounded-xl p-5 space-y-3">
                <h4 className="text-xs font-medium text-primary-400 uppercase tracking-widest mb-3">
                  Account Details
                </h4>

                <div className="flex items-center justify-between">
                  <span className="text-sm text-primary-600">Role</span>
                  <span className="text-sm font-medium text-primary-900 capitalize">{user?.role}</span>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm text-primary-600">Status</span>
                  <span className="inline-flex items-center gap-1.5 px-2 py-1 bg-green-100 text-green-700 text-xs font-medium rounded-full">
                    <span className="w-1.5 h-1.5 bg-green-500 rounded-full"></span>
                    Active
                  </span>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm text-primary-600">Member since</span>
                  <span className="text-sm font-medium text-primary-900">
                    {user?.created_at ? new Date(user.created_at).toLocaleDateString() : 'N/A'}
                  </span>
                </div>
              </div>

            </div>
          ) : (
            <div className="p-6">
              {loading ? (
                <div className="flex items-center justify-center py-12">
                  <div className="w-8 h-8 border-3 border-primary-200 border-t-primary-600 rounded-full animate-spin"></div>
                </div>
              ) : reports.length === 0 ? (
                <div className="text-center py-12">
                  <div className="w-16 h-16 bg-primary-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <svg className="w-8 h-8 text-primary-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                  </div>
                  <p className="text-sm text-primary-500">No reports yet</p>
                  <p className="text-xs text-primary-400 mt-1">Your saved analyses will appear here</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {reports.map((report) => (
                    <div
                      key={report.report_id}
                      className="bg-white border border-primary-200 rounded-xl p-4 hover:border-primary-300 hover:shadow-sm transition-all cursor-pointer"
                    >
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex-1">
                          <h4 className="text-sm font-medium text-primary-900">{report.patient_name}</h4>
                          <p className="text-xs text-primary-500 mt-0.5">
                            {new Date(report.analysis_date).toLocaleDateString('en-US', {
                              month: 'short',
                              day: 'numeric',
                              year: 'numeric',
                            })}
                          </p>
                        </div>
                        <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                          report.prediction === 'ASD'
                            ? 'bg-orange-100 text-orange-700'
                            : 'bg-blue-100 text-blue-700'
                        }`}>
                          {report.prediction}
                        </span>
                      </div>

                      <div className="flex items-center gap-4 text-xs text-primary-600">
                        <div className="flex items-center gap-1">
                          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                          </svg>
                          <span>{Math.round(report.confidence * 100)}% confidence</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" />
                          </svg>
                          <span className="capitalize">{report.input_type.replace('_', ' ')}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Sign Out Button */}
        <div className="border-t border-primary-100 p-6 bg-white">
          <button
            onClick={handleLogout}
            className="w-full px-4 py-3 bg-red-600 text-white rounded-xl text-sm font-black hover:bg-red-700 transition-all"
          >
            Sign Out
          </button>
        </div>
      </div>
    </>
  );
}
