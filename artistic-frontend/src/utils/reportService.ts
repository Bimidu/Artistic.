/**
 * Report Service
 *
 * Utility functions for saving and retrieving patient analysis reports
 */

export interface ReportData {
  patient_name: string;
  prediction: string;
  confidence: number;
  probabilities: Record<string, number>;
  model_used: string;
  input_type: 'audio' | 'text' | 'chat_file';
  features_extracted?: number | { count: number };
  transcript?: string;
}

export interface SavedReport extends ReportData {
  report_id: string;
  analysis_date: string;
  created_at: string;
}

const getApiUrl = () => process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

/**
 * Save a new analysis report
 */
export async function saveReport(
  token: string,
  reportData: ReportData
): Promise<SavedReport> {
  // Normalize features_extracted to be a number
  const normalizedData = {
    ...reportData,
    features_extracted: typeof reportData.features_extracted === 'object' && reportData.features_extracted !== null
      ? (reportData.features_extracted as { count: number }).count
      : reportData.features_extracted,
  };

  const response = await fetch(`${getApiUrl()}/api/reports/save`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`,
    },
    body: JSON.stringify(normalizedData),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to save report');
  }

  return response.json();
}

/**
 * Get all reports for the current user
 */
export async function getMyReports(token: string): Promise<SavedReport[]> {
  const response = await fetch(`${getApiUrl()}/api/reports/my-reports`, {
    headers: {
      'Authorization': `Bearer ${token}`,
    },
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to fetch reports');
  }

  return response.json();
}

/**
 * Generate a user-friendly report ID
 */
export function generateReportId(): string {
  const timestamp = Date.now().toString(36);
  const randomStr = Math.random().toString(36).substring(2, 7);
  return `RPT-${timestamp}-${randomStr}`.toUpperCase();
}
