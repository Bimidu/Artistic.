
import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.parsers.chat_parser import CHATParser

def run_clustering():
    """
    Apply Unsupervised ML (GMM/K-Means) to find natural pause thresholds.
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
        if not utterances: continue
        
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
        
        # We assume 3 components: 
        # 1. Normal/Fast response 
        # 2. Thinking/Processing pause
        # 3. Disengagement/Long pause
        gmm = GaussianMixture(n_components=3, random_state=42)
        gmm.fit(X)
        
        means = gmm.means_.flatten()
        weights = gmm.weights_.flatten()
        covariances = gmm.covariances_.flatten()
        
        # Sort clusters by mean duration
        sorted_indices = np.argsort(means)
        means = means[sorted_indices]
        weights = weights[sorted_indices]
        stds = np.sqrt(covariances[sorted_indices])
        
        print("\nDiscovered Clusters:")
        labels = ['Normal (Fast)', 'Processing (Medium)', 'Long (Disengaged)']
        for i in range(3):
            print(f"Cluster {i+1} [{labels[i]}]: Mean = {means[i]:.2f}s, Std = {stds[i]:.2f}s, Weight = {weights[i]:.2f}")
            
        print("\n--- Calculating Decision Boundaries ---")
        # Boundary between 1 and 2 (Normal vs Long)
        # Simple approximation: equidistant or weighted intersection.
        # Let's use the midpoint between means for simplicity, or 2 sigma from mean 1.
        
        # Better: Solve for intersection of PDF, or just take (Mean1 + Mean2)/2
        # Usually, "Threshold" is where you switch from Class 1 to Class 2.
        
        t1_normal_limit = means[0] + 1.5 * stds[0] # Aggressive limit of normal
        t2_long_start = means[1]
        
        # A conservative boundary between Cluster 1 and 2
        boundary_1_2 = (means[0] * stds[1] + means[1] * stds[0]) / (stds[0] + stds[1]) # Weighted by spread
        
        # Boundary between 2 and 3 (Long vs Very Long)
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
        print("sklearn not found. Using simple 1D K-Means implementation.")
        # Simple 1D K-Means
        # Init centroids
        centroids = np.array([0.5, 2.0, 5.0])
        for _ in range(20):
            # Assign
            distances = np.abs(X - centroids)
            labels = np.argmin(distances, axis=1)
            # Update
            new_centroids = np.array([X[labels == k].mean() for k in range(3)])
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
            
        centroids.sort()
        print(f"Converged Centroids: {centroids}")
        
        boundary_1_2 = (centroids[0] + centroids[1]) / 2
        boundary_2_3 = (centroids[1] + centroids[2]) / 2
        
        print(f"Proposed Boundary (Normal -> Processing): {boundary_1_2:.2f} s")
        print(f"Proposed Boundary (Processing -> Long):   {boundary_2_3:.2f} s")

if __name__ == "__main__":
    run_clustering()
