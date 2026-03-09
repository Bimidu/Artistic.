'use client';

import { useEffect } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { saveReport, type ReportData } from '@/utils/reportService';

/**
 * This component exposes report saving functionality to vanilla JS (app.js)
 * It allows the existing app.js code to save reports without refactoring
 */
export default function ReportSaveIntegration() {
  const { user, token } = useAuth();

  useEffect(() => {
    // Expose save report function to global window object
    (window as Window & {
      saveReportToBackend?: (reportData: ReportData, patientName: string) => Promise<void>;
      checkAuthStatus?: () => { isAuthenticated: boolean; user: unknown };
    }).saveReportToBackend = async (reportData: ReportData, patientName: string) => {
      if (!token) {
        throw new Error('User not authenticated');
      }

      await saveReport(token, {
        ...reportData,
        patient_name: patientName,
      });
    };

    // Expose auth status check
    (window as Window & {
      checkAuthStatus?: () => { isAuthenticated: boolean; user: unknown };
    }).checkAuthStatus = () => ({
      isAuthenticated: !!user,
      user: user,
    });

    // Cleanup
    return () => {
      const win = window as Window & {
        saveReportToBackend?: (reportData: ReportData, patientName: string) => Promise<void>;
        checkAuthStatus?: () => { isAuthenticated: boolean; user: unknown };
      };
      delete win.saveReportToBackend;
      delete win.checkAuthStatus;
    };
  }, [user, token]);

  return null; // This component doesn't render anything
}
