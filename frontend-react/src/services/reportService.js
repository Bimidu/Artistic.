import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * Report Service
 * Handles API calls for patient reports
 */
export const reportService = {
    /**
     * Save a new patient analysis report
     */
    async saveReport(reportData) {
        const token = localStorage.getItem('asd_auth_token');
        const response = await axios.post(
            `${API_URL}/api/reports/save`,
            reportData,
            {
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json'
                }
            }
        );
        return response.data;
    },

    /**
     * Get all reports for the current user
     */
    async getMyReports() {
        const token = localStorage.getItem('asd_auth_token');
        const response = await axios.get(
            `${API_URL}/api/reports/my-reports`,
            {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            }
        );
        return response.data;
    },

    /**
     * Get reports grouped by patient
     */
    async getReportsByPatient() {
        const token = localStorage.getItem('asd_auth_token');
        const response = await axios.get(
            `${API_URL}/api/reports/by-patient`,
            {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            }
        );
        return response.data;
    }
};
