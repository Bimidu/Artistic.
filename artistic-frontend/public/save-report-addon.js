/**
 * Save Report Add-on
 *
 * This script adds save report functionality to the existing app.js
 * It integrates with the React authentication context
 */

// Store the current analysis data globally
let currentAnalysisData = null;

// Override the displayResults function to include save button
(function() {
  // Wait for the original displayResults to be defined
  const originalDisplayResults = window.displayResults;

  window.displayResults = function(data) {
    // Store the analysis data
    currentAnalysisData = {
      prediction: data.prediction,
      confidence: data.confidence,
      probabilities: data.probabilities,
      model_used: data.models_used ? data.models_used.join(', ') : (data.model_used || 'Unknown'),
      input_type: data.input_type,
      features_extracted: data.features_extracted ? { count: data.features_extracted } : undefined,
      transcript: data.transcript || data.annotated_transcript_html || undefined,
    };

    // Call the original function
    if (originalDisplayResults) {
      originalDisplayResults(data);
    }

    // Add save button to results area
    addSaveButtonToResults();
  };
})();

function addSaveButtonToResults() {
  const resultsArea = document.getElementById('resultsArea');
  if (!resultsArea) return;

  // Check if save button already exists
  if (document.getElementById('saveReportBtn')) return;

  // Create save button container
  const saveButtonContainer = document.createElement('div');
  saveButtonContainer.className = 'mt-6 pt-6 border-t border-primary-200';
  saveButtonContainer.innerHTML = `
    <div class="flex items-center justify-between">
      <div>
        <p class="text-sm font-medium text-primary-900 mb-1">Save this analysis</p>
        <p class="text-xs text-primary-500">Store this report for future reference</p>
      </div>
      <button
        id="saveReportBtn"
        class="px-6 py-3 bg-primary-900 text-white rounded-xl text-sm font-medium hover:bg-primary-800 transition-all flex items-center gap-2"
      >
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3 3m0 0l-3-3m3 3V4" />
        </svg>
        Save Report
      </button>
    </div>
  `;

  resultsArea.appendChild(saveButtonContainer);

  // Add click handler
  document.getElementById('saveReportBtn').addEventListener('click', handleSaveReport);
}

async function handleSaveReport() {
  // Check if user is authenticated
  const authStatus = window.checkAuthStatus ? window.checkAuthStatus() : { isAuthenticated: false };

  if (!authStatus.isAuthenticated) {
    showSaveReportModal('Please log in to save reports. Click the Login button in the top right corner.');
    return;
  }

  if (!currentAnalysisData) {
    showSaveReportModal('No analysis data to save');
    return;
  }

  // Show patient name input modal
  showPatientNameModal();
}

function showPatientNameModal() {
  // Create modal
  const modal = document.createElement('div');
  modal.id = 'patientNameModal';
  modal.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
  modal.innerHTML = `
    <div class="bg-white rounded-2xl max-w-md w-full mx-4 shadow-2xl">
      <div class="px-8 py-6 border-b border-primary-100">
        <h3 class="text-xl font-normal text-primary-900" style="letter-spacing: -0.02em;">
          Save Analysis Report
        </h3>
      </div>

      <div class="px-8 py-6">
        <label for="patientNameInput" class="block text-sm font-medium text-primary-700 mb-2">
          Patient Name / ID
        </label>
        <input
          type="text"
          id="patientNameInput"
          class="w-full px-4 py-3 bg-white border border-primary-200 rounded-xl text-sm focus:outline-none focus:border-primary-400 transition-all"
          placeholder="e.g., Patient 001 or John Doe"
          autofocus
        />
        <p class="mt-2 text-xs text-primary-500">
          Enter an identifier to help you find this report later
        </p>

        <div id="saveReportError" class="hidden mt-4 px-4 py-3 bg-red-50 border border-red-200 rounded-xl">
          <p class="text-sm text-red-600"></p>
        </div>

        <div class="mt-6 flex gap-3">
          <button
            id="cancelSaveBtn"
            class="flex-1 px-4 py-3 bg-white border border-primary-200 text-primary-700 rounded-xl text-sm font-medium hover:bg-primary-50 transition-all"
          >
            Cancel
          </button>
          <button
            id="confirmSaveBtn"
            class="flex-1 px-4 py-3 bg-primary-900 text-white rounded-xl text-sm font-medium hover:bg-primary-800 transition-all"
          >
            Save
          </button>
        </div>
      </div>
    </div>
  `;

  document.body.appendChild(modal);

  // Add event listeners
  document.getElementById('cancelSaveBtn').addEventListener('click', () => {
    document.body.removeChild(modal);
  });

  document.getElementById('confirmSaveBtn').addEventListener('click', async () => {
    const patientName = document.getElementById('patientNameInput').value.trim();

    if (!patientName) {
      showSaveError('Please enter a patient name');
      return;
    }

    // Show loading state
    const saveBtn = document.getElementById('confirmSaveBtn');
    saveBtn.disabled = true;
    saveBtn.textContent = 'Saving...';

    try {
      // Call the React-exposed save function
      if (window.saveReportToBackend) {
        await window.saveReportToBackend(currentAnalysisData, patientName);

        // Success - update button appearance
        const mainSaveBtn = document.getElementById('saveReportBtn');
        if (mainSaveBtn) {
          mainSaveBtn.className = 'px-6 py-3 bg-green-100 text-green-700 border border-green-200 rounded-xl text-sm font-medium flex items-center gap-2';
          mainSaveBtn.innerHTML = `
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
            </svg>
            Report Saved
          `;
          mainSaveBtn.disabled = true;
        }

        // Close modal
        document.body.removeChild(modal);

        // Show success message
        showSaveReportModal(`Report saved successfully for ${patientName}`);
      } else {
        throw new Error('Save function not available');
      }
    } catch (error) {
      console.error('Failed to save report:', error);
      showSaveError(error.message || 'Failed to save report');
      saveBtn.disabled = false;
      saveBtn.textContent = 'Save';
    }
  });

  // Allow Enter key to save
  document.getElementById('patientNameInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      document.getElementById('confirmSaveBtn').click();
    }
  });

  // Click outside to close
  modal.addEventListener('click', (e) => {
    if (e.target === modal) {
      document.body.removeChild(modal);
    }
  });
}

function showSaveError(message) {
  const errorDiv = document.getElementById('saveReportError');
  if (errorDiv) {
    errorDiv.querySelector('p').textContent = message;
    errorDiv.classList.remove('hidden');
  }
}

function showSaveReportModal(message) {
  // Simple notification
  const notification = document.createElement('div');
  notification.className = 'fixed top-4 right-4 bg-white border border-primary-200 rounded-xl shadow-lg p-4 z-50 max-w-sm';
  notification.innerHTML = `
    <div class="flex items-start gap-3">
      <div class="w-5 h-5 bg-primary-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
        <svg class="w-3 h-3 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      </div>
      <p class="text-sm text-primary-900 flex-1">${message}</p>
      <button class="text-primary-400 hover:text-primary-900 transition-colors" onclick="this.parentElement.parentElement.remove()">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </div>
  `;

  document.body.appendChild(notification);

  // Auto-remove after 4 seconds
  setTimeout(() => {
    if (notification.parentElement) {
      notification.remove();
    }
  }, 4000);
}

// Initialize on page load
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    console.log('Save Report Add-on loaded');
  });
} else {
  console.log('Save Report Add-on loaded');
}
