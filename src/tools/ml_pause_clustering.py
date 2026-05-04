
"""
ML Pause Threshold Discovery Tool

This script uses unsupervised learning to empirically determine the pause/latency
thresholds used in the PauseLatencyFeatures extractor. Rather than picking thresholds
by hand, it learns them from the distribution of actual inter-turn gaps observed in
the ASDBank corpus.

Approach — Gaussian Mixture Model (GMM):
  A GMM with 3 components is fitted to child-turn inter-turn gaps. The three
  components correspond to three naturally occurring pause types:
    1. Rapid response     — child replies quickly (no processing delay)
    2. Processing pause   — child pauses briefly before responding
    3. Disengagement/long — child takes a very long time or does not respond

  The cluster means and standard deviations inform the thresholds used in
  PauseLatencyFeatures (NORMAL_RESPONSE_TIME, LONG_PAUSE_THRESHOLD, etc.).

Fallback — 1-D K-Means:
  If sklearn is unavailable, a hand-rolled 1-D K-Means is used as an approximation.

Results from this script were used to set the data-driven thresholds in
`src/features/pragmatic_conversational/pause_latency.py`.

Usage:
    python -m src.tools.ml_pause_clustering
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.parsers.chat_parser import CHATParser

def run_clustering():
    """
    Fit a GMM to child response latencies and report discovered cluster boundaries.

    The pipeline is:
      1. Parse all ASDBank transcripts.
      2. Extract every inter-turn gap where the current speaker is the child (CHI).
      3. Filter to gaps in [0, 10s) — discards noise and session boundaries.
      4. Fit a 3-component GMM and sort components by mean duration.
      5. Compute decision boundaries as a spread-weighted midpoint between adjacent means.
      6. Plot the distribution (requires matplotlib/scipy) and save as PNG.

    The resulting boundary values feed directly into the thresholds in
    PauseLatencyFeatures.
    """
    data_dir = project_root / "data/asdbank_aac"
    print(f"Collecting data from: {data_dir}")
    
    parser = CHATParser()
    try:
        # Using the robust subset we identified
        transcripts = parser.parse_directory(data_dir, recursive=True)
    except Exception as e:
        print(f"Error parsing: {e}")
        return

    latencies = []
    
    print("Extracting features...")
    for t in transcripts:
        utterances = t.utterances
        if not utterances:
            continue
        
        for i in range(1, len(utterances)):
            curr = utterances[i]
            prev = utterances[i-1]
            
            if curr.timing is not None and prev.end_timing is not None:
                diff = curr.timing - prev.end_timing
                # Filter for valid conversational pauses (0 to 10s)
                if 0 <= diff < 10.0:
                    if curr.speaker == 'CHI':
                        latencies.append(diff)
                        
    X = np.array(latencies).reshape(-1, 1)
    print(f"\ndataset size: {len(X)} samples")
    
    if len(X) < 50:
        print("Insufficient data for ML clustering.")
        return

    print("\n--- Applying Gaussian Mixture Model (GMM) ---")
    try:
        from sklearn.mixture import GaussianMixture
        
        # Three components reflect three psycholinguistically motivated pause types:
        #   1. Rapid — child has no processing delay
        #   2. Processing — brief planning or retrieval delay
        #   3. Disengaged — very long gap, potential communicative breakdown
        gmm = GaussianMixture(n_components=3, random_state=42)
        gmm.fit(X)
        
        means = gmm.means_.flatten()
        weights = gmm.weights_.flatten()
        covariances = gmm.covariances_.flatten()
        
        # Sort clusters by mean duration so labels stay interpretable
        sorted_indices = np.argsort(means)
        means = means[sorted_indices]
        weights = weights[sorted_indices]
        stds = np.sqrt(covariances[sorted_indices])
        
        print("\nDiscovered Clusters:")
        labels = ['Normal (Fast)', 'Processing (Medium)', 'Long (Disengaged)']
        for i in range(3):
            print(f"Cluster {i+1} [{labels[i]}]: Mean = {means[i]:.2f}s, Std = {stds[i]:.2f}s, Weight = {weights[i]:.2f}")
            
        print("\n--- Calculating Decision Boundaries ---")
        # Boundaries are computed as a spread-weighted midpoint between adjacent cluster means.
        # Weighting by the opposing cluster's std gives a boundary that leans toward
        # the tighter (more certain) distribution, which is more stable than a simple midpoint.

        t1_normal_limit = means[0] + 1.5 * stds[0]  # Aggressive limit of normal
        t2_long_start = means[1]
        
        # Boundary between Cluster 1 (Rapid) and Cluster 2 (Processing)
        boundary_1_2 = (means[0] * stds[1] + means[1] * stds[0]) / (stds[0] + stds[1])
        
        # Boundary between Cluster 2 (Processing) and Cluster 3 (Disengaged)
        boundary_2_3 = (means[1] * stds[2] + means[2] * stds[1]) / (stds[1] + stds[2])
        
        print(f"Proposed Boundary (Normal -> Processing): {boundary_1_2:.2f} s")
        print(f"Proposed Boundary (Processing -> Long):   {boundary_2_3:.2f} s")
        
        # Plotting
        try:
            import matplotlib.pyplot as plt
            x = np.linspace(0, 10, 1000).reshape(-1, 1)
            logprob = gmm.score_samples(x)
            pdf = np.exp(logprob)
            
            plt.figure(figsize=(10, 6))
            
            # Histogram of data
            plt.hist(X, bins=50, density=True, alpha=0.5, color='gray', label='Observed Latency')
            
            # Individual components
            for i in range(3):
                # We need to construct individual PDFs manually or using scipy
                # Simple approximation using weights, mean, std
                import scipy.stats as stats
                component_pdf = weights[i] * stats.norm.pdf(x, means[i], stds[i])
                plt.plot(x, component_pdf, '--', linewidth=2, label=f'{labels[i]} Component')
            
            # Total PDF
            plt.plot(x, pdf, '-k', linewidth=2, label='Total GMM Density')
            
            # Decision Boundaries
            plt.axvline(boundary_1_2, color='r', linestyle=':', label=f'Threshold 1 ({boundary_1_2:.2f}s)')
            plt.axvline(boundary_2_3, color='r', linestyle=':', label=f'Threshold 2 ({boundary_2_3:.2f}s)')
            
            plt.title('Pause Latency Distribution & ML Clusters (ASD Cohort)')
            plt.xlabel('Response Latency (seconds)')
            plt.ylabel('Density')
            plt.legend()
            plt.grid(alpha=0.3)
            
            output_path = project_root / "pause_clustering_results.png"
            plt.savefig(output_path)
            print(f"\nGraph saved to: {output_path}")
            
        except ImportError as e:
            print(f"Could not plot: {e}. Install matplotlib and scipy.")
            
    except ImportError:
        # sklearn unavailable — fall back to a hand-rolled 1-D K-Means.
        # Initial centroids are chosen to match the expected cluster centres so
        # convergence is fast and stable even without a proper seeding strategy.
        print("sklearn not found. Using simple 1D K-Means implementation.")
        centroids = np.array([0.5, 2.0, 5.0])
        for _ in range(20):
            distances = np.abs(X - centroids)
            labels = np.argmin(distances, axis=1)
            new_centroids = np.array([X[labels == k].mean() for k in range(3)])
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
            
        centroids.sort()
        print(f"Converged Centroids: {centroids}")
        
        # Simple midpoint boundary — less precise than the GMM spread-weighted version
        # but adequate when the cluster distributions are well-separated
        boundary_1_2 = (centroids[0] + centroids[1]) / 2
        boundary_2_3 = (centroids[1] + centroids[2]) / 2
        
        print(f"Proposed Boundary (Normal -> Processing): {boundary_1_2:.2f} s")
        print(f"Proposed Boundary (Processing -> Long):   {boundary_2_3:.2f} s")

if __name__ == "__main__":
    run_clustering()
