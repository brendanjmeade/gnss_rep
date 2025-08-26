"""
GPS Time Series Dimensionality Reduction Analysis
==================================================
Comprehensive analysis of GPS displacement data using various
dimensionality reduction techniques for earthquake detection and
tectonic motion analysis.

Author: Time Series Expert
Dataset: 400 GPS stations × 7300 daily observations in Japan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# Optional imports (install as needed)
try:
    from pydmd import DMD

    DMD_AVAILABLE = True
except ImportError:
    DMD_AVAILABLE = False
    print("PyDMD not installed. Run: pip install pydmd")

try:
    from pyts.decomposition import SingularSpectrumAnalysis

    SSA_AVAILABLE = True
except ImportError:
    SSA_AVAILABLE = False
    print("pyts not installed. Run: pip install pyts")

try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not installed. Run: pip install umap-learn")

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not installed. Run: pip install torch")

try:
    from eofs.standard import Eof

    EOF_AVAILABLE = True
except ImportError:
    EOF_AVAILABLE = False
    print("eofs not installed. Run: pip install eofs")


class GPSDataGenerator:
    """Generate synthetic GPS data for demonstration"""

    @staticmethod
    def generate_synthetic_data(n_stations=400, n_timesteps=7300, n_components=3):
        """
        Generate synthetic GPS data with:
        - Slow tectonic drift
        - Seasonal signals
        - Earthquake events
        - Noise
        """
        np.random.seed(42)

        # Initialize data array
        data = np.zeros((n_stations * n_components, n_timesteps))

        # Time array (in days)
        t = np.arange(n_timesteps)

        for i in range(n_stations):
            for j in range(n_components):
                idx = i * n_components + j

                # Slow tectonic drift (mm/year converted to mm/day)
                drift_rate = np.random.uniform(-20, 20) / 365.25
                drift = drift_rate * t

                # Seasonal signal (annual + semi-annual)
                annual = 5 * np.sin(
                    2 * np.pi * t / 365.25 + np.random.uniform(0, 2 * np.pi)
                )
                semiannual = 2 * np.sin(
                    4 * np.pi * t / 365.25 + np.random.uniform(0, 2 * np.pi)
                )

                # Add earthquake events (step functions + exponential decay)
                n_events = np.random.poisson(3)  # Average 3 events per station
                earthquake_signal = np.zeros(n_timesteps)

                for _ in range(n_events):
                    event_time = np.random.randint(100, n_timesteps - 100)
                    magnitude = np.random.exponential(10)

                    # Coseismic offset
                    earthquake_signal[event_time:] += magnitude

                    # Postseismic decay
                    decay_time = np.arange(n_timesteps - event_time)
                    earthquake_signal[event_time:] += (
                        magnitude * 0.3 * (1 - np.exp(-decay_time / 100))
                    )

                # Combine all signals
                data[idx, :] = drift + annual + semiannual + earthquake_signal

                # Add white noise
                data[idx, :] += np.random.normal(0, 0.5, n_timesteps)

        return data, t


class DimensionalityReductionAnalysis:
    """Main class for dimensionality reduction analysis of GPS data"""

    def __init__(self, data, station_names=None, verbose=True):
        """
        Initialize with GPS data

        Parameters:
        -----------
        data : numpy array
            Shape: (n_stations * n_components, n_timesteps)
        station_names : list, optional
            Names of GPS stations
        verbose : bool
            Print progress messages
        """
        self.data = data
        self.n_features, self.n_timesteps = data.shape
        self.n_stations = self.n_features // 3  # Assuming 3 components (E, N, U)
        self.station_names = station_names or [
            f"Station_{i}" for i in range(self.n_stations)
        ]
        self.verbose = verbose
        self.results = {}

        # Standardize data
        self.scaler = StandardScaler()
        self.data_scaled = self.scaler.fit_transform(data.T).T

    def pca_analysis(self, n_components=20):
        """
        Principal Component Analysis
        """
        if self.verbose:
            print("\n" + "=" * 50)
            print("Running PCA Analysis...")
            print("=" * 50)

        pca = PCA(n_components=n_components)

        # Fit on transposed data (time × features)
        pca_scores = pca.fit_transform(self.data_scaled.T)

        self.results["pca"] = {
            "model": pca,
            "scores": pca_scores,
            "components": pca.components_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
        }

        if self.verbose:
            print(
                f"PCA captured {self.results['pca']['cumulative_variance'][-1]:.2%} of variance"
            )
            print(
                f"First 5 components explain: {self.results['pca']['cumulative_variance'][4]:.2%}"
            )

        return self.results["pca"]

    def ica_analysis(self, n_components=20):
        """
        Independent Component Analysis
        """
        if self.verbose:
            print("\n" + "=" * 50)
            print("Running ICA Analysis...")
            print("=" * 50)

        ica = FastICA(n_components=n_components, random_state=42, max_iter=1000)

        # Fit on transposed data
        ica_sources = ica.fit_transform(self.data_scaled.T)

        self.results["ica"] = {
            "model": ica,
            "sources": ica_sources,
            "mixing_matrix": ica.mixing_,
            "components": ica.components_,
        }

        if self.verbose:
            print(f"ICA extracted {n_components} independent components")
            print(f"Convergence achieved: {ica.n_iter_ < ica.max_iter}")

        return self.results["ica"]

    def dmd_analysis(self, svd_rank=20):
        """
        Dynamic Mode Decomposition
        """
        if not DMD_AVAILABLE:
            print("DMD analysis skipped (PyDMD not installed)")
            return None

        if self.verbose:
            print("\n" + "=" * 50)
            print("Running DMD Analysis...")
            print("=" * 50)

        dmd = DMD(svd_rank=svd_rank)
        dmd.fit(self.data_scaled)

        self.results["dmd"] = {
            "model": dmd,
            "modes": dmd.modes,
            "dynamics": dmd.dynamics,
            "eigenvalues": dmd.eigs,
            "amplitudes": dmd.amplitudes,
        }

        if self.verbose:
            print(f"DMD extracted {svd_rank} dynamic modes")
            print(
                f"Dominant frequency: {np.abs(np.angle(dmd.eigs[0])) / (2 * np.pi):.4f} cycles/day"
            )

        return self.results["dmd"]

    def ssa_analysis(self, window_size=365, n_groups=10):
        """
        Singular Spectrum Analysis
        """
        if not SSA_AVAILABLE:
            print("SSA analysis skipped (pyts not installed)")
            return None

        if self.verbose:
            print("\n" + "=" * 50)
            print("Running SSA Analysis...")
            print("=" * 50)

        # Apply SSA to each station/component separately
        ssa_components = []

        for i in range(min(5, self.n_features)):  # Process first 5 for demonstration
            ssa = SingularSpectrumAnalysis(window_size=window_size, groups=n_groups)
            # Reshape for SSA (needs 2D input with shape (n_samples, n_timestamps))
            component = ssa.fit_transform(self.data_scaled[i : i + 1, :])
            ssa_components.append(component)

        self.results["ssa"] = {
            "components": np.array(ssa_components),
            "window_size": window_size,
            "n_groups": n_groups,
        }

        if self.verbose:
            print(f"SSA decomposed signals with window size {window_size}")
            print(f"Extracted {n_groups} groups per signal")

        return self.results["ssa"]

    def nmf_analysis(self, n_components=15):
        """
        Non-negative Matrix Factorization
        """
        if self.verbose:
            print("\n" + "=" * 50)
            print("Running NMF Analysis...")
            print("=" * 50)

        # Ensure non-negative data (use absolute values or shift)
        data_positive = self.data - self.data.min() + 1e-10

        nmf = NMF(n_components=n_components, init="nndsvd", random_state=42)
        W = nmf.fit_transform(data_positive.T)
        H = nmf.components_

        self.results["nmf"] = {
            "model": nmf,
            "W": W,  # Temporal patterns
            "H": H,  # Spatial patterns
            "reconstruction_error": nmf.reconstruction_err_,
        }

        if self.verbose:
            print(f"NMF extracted {n_components} non-negative components")
            print(f"Reconstruction error: {nmf.reconstruction_err_:.4f}")

        return self.results["nmf"]

    def umap_analysis(self, n_components=3, n_neighbors=30):
        """
        UMAP (Uniform Manifold Approximation and Projection)
        """
        if not UMAP_AVAILABLE:
            print("UMAP analysis skipped (umap-learn not installed)")
            return None

        if self.verbose:
            print("\n" + "=" * 50)
            print("Running UMAP Analysis...")
            print("=" * 50)

        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            random_state=42,
        )

        # Apply UMAP to time points
        embedding = reducer.fit_transform(self.data_scaled.T)

        self.results["umap"] = {
            "model": reducer,
            "embedding": embedding,
            "n_components": n_components,
        }

        if self.verbose:
            print(f"UMAP reduced to {n_components}D manifold")
            print(f"Used {n_neighbors} nearest neighbors")

        return self.results["umap"]

    def eof_analysis(self, n_eofs=20):
        """
        Empirical Orthogonal Functions
        """
        if not EOF_AVAILABLE:
            print("EOF analysis skipped (eofs not installed)")
            return None

        if self.verbose:
            print("\n" + "=" * 50)
            print("Running EOF Analysis...")
            print("=" * 50)

        # Reshape data for EOF analysis (time, space)
        solver = Eof(self.data_scaled.T)

        eofs = solver.eofs(neofs=n_eofs)
        pcs = solver.pcs(npcs=n_eofs)
        variance_fractions = solver.varianceFraction(neigs=n_eofs)

        self.results["eof"] = {
            "solver": solver,
            "eofs": eofs,
            "pcs": pcs,
            "variance_fractions": variance_fractions,
        }

        if self.verbose:
            print(f"EOF extracted {n_eofs} modes")
            print(
                f"First 5 EOFs explain: {variance_fractions[:5].sum():.2%} of variance"
            )

        return self.results["eof"]

    def common_mode_analysis(self, n_components=5):
        """
        Common Mode Component Analysis for GPS networks
        """
        if self.verbose:
            print("\n" + "=" * 50)
            print("Running Common Mode Analysis...")
            print("=" * 50)

        # Reshape to (n_stations, n_components, n_times)
        data_reshaped = self.data.reshape(self.n_stations, 3, self.n_timesteps)

        # Extract common mode for each component (E, N, U)
        common_modes = []
        spatial_responses = []

        for comp in range(3):
            comp_data = data_reshaped[:, comp, :]

            # Remove station-specific means
            data_centered = comp_data - comp_data.mean(axis=1, keepdims=True)

            # Weight by station variance
            weights = 1.0 / (data_centered.var(axis=1, keepdims=True) + 1e-10)
            data_weighted = data_centered * np.sqrt(weights)

            # PCA on weighted data
            pca = PCA(n_components=n_components)
            modes = pca.fit_transform(data_weighted.T)

            common_modes.append(modes)
            spatial_responses.append(pca.components_)

        self.results["common_mode"] = {
            "temporal_modes": common_modes,
            "spatial_responses": spatial_responses,
            "n_components": n_components,
        }

        if self.verbose:
            print(f"Extracted {n_components} common modes per component")
            print("Common modes represent network-wide signals")

        return self.results["common_mode"]

    def vae_analysis(self, latent_dim=10, n_epochs=50):
        """
        Variational Autoencoder for nonlinear dimensionality reduction
        """
        if not TORCH_AVAILABLE:
            print("VAE analysis skipped (PyTorch not installed)")
            return None

        if self.verbose:
            print("\n" + "=" * 50)
            print("Running VAE Analysis...")
            print("=" * 50)

        class GPS_VAE(nn.Module):
            def __init__(self, input_dim, latent_dim):
                super().__init__()
                # Encoder
                self.fc1 = nn.Linear(input_dim, 512)
                self.fc2 = nn.Linear(512, 256)
                self.fc_mu = nn.Linear(256, latent_dim)
                self.fc_var = nn.Linear(256, latent_dim)

                # Decoder
                self.fc3 = nn.Linear(latent_dim, 256)
                self.fc4 = nn.Linear(256, 512)
                self.fc5 = nn.Linear(512, input_dim)

            def encode(self, x):
                h = torch.relu(self.fc1(x))
                h = torch.relu(self.fc2(h))
                return self.fc_mu(h), self.fc_var(h)

            def reparameterize(self, mu, log_var):
                std = torch.exp(0.5 * log_var)
                eps = torch.randn_like(std)
                return mu + eps * std

            def decode(self, z):
                h = torch.relu(self.fc3(z))
                h = torch.relu(self.fc4(h))
                return self.fc5(h)

            def forward(self, x):
                mu, log_var = self.encode(x)
                z = self.reparameterize(mu, log_var)
                return self.decode(z), mu, log_var

        # Prepare data
        data_tensor = torch.FloatTensor(self.data_scaled.T)

        # Initialize model
        model = GPS_VAE(self.n_features, latent_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Training loop (simplified)
        model.train()
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            recon_batch, mu, log_var = model(data_tensor)

            # Loss = reconstruction loss + KL divergence
            recon_loss = nn.functional.mse_loss(recon_batch, data_tensor)
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = recon_loss + 0.001 * kl_loss

            loss.backward()
            optimizer.step()

            if epoch % 10 == 0 and self.verbose:
                print(f"  Epoch {epoch}: Loss = {loss.item():.4f}")

        # Extract latent representation
        model.eval()
        with torch.no_grad():
            mu, _ = model.encode(data_tensor)
            latent = mu.numpy()

        self.results["vae"] = {
            "model": model,
            "latent_representation": latent,
            "latent_dim": latent_dim,
        }

        if self.verbose:
            print(f"VAE compressed to {latent_dim}D latent space")

        return self.results["vae"]

    def plot_results(self):
        """
        Visualize results from different methods
        """
        n_methods = len([k for k in self.results.keys() if self.results[k] is not None])

        if n_methods == 0:
            print("No results to plot")
            return

        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()

        plot_idx = 0

        # PCA variance explained
        if "pca" in self.results and self.results["pca"] is not None:
            ax = axes[plot_idx]
            variance = self.results["pca"]["explained_variance_ratio"][:10]
            ax.bar(range(1, len(variance) + 1), variance)
            ax.set_xlabel("PC Number")
            ax.set_ylabel("Variance Explained")
            ax.set_title("PCA: Variance Explained")
            plot_idx += 1

        # ICA sources (first 3)
        if "ica" in self.results and self.results["ica"] is not None:
            ax = axes[plot_idx]
            sources = self.results["ica"]["sources"]
            for i in range(min(3, sources.shape[1])):
                ax.plot(sources[:1000, i], alpha=0.7, label=f"IC{i + 1}")
            ax.set_xlabel("Time (days)")
            ax.set_ylabel("Amplitude")
            ax.set_title("ICA: First 3 Independent Components")
            ax.legend()
            plot_idx += 1

        # DMD eigenvalues
        if "dmd" in self.results and self.results["dmd"] is not None:
            ax = axes[plot_idx]
            eigs = self.results["dmd"]["eigenvalues"]
            ax.scatter(eigs.real, eigs.imag, alpha=0.6)
            circle = plt.Circle((0, 0), 1, fill=False, edgecolor="r", linestyle="--")
            ax.add_patch(circle)
            ax.set_xlabel("Real")
            ax.set_ylabel("Imaginary")
            ax.set_title("DMD: Eigenvalues")
            ax.set_aspect("equal")
            plot_idx += 1

        # NMF temporal patterns
        if "nmf" in self.results and self.results["nmf"] is not None:
            ax = axes[plot_idx]
            W = self.results["nmf"]["W"]
            for i in range(min(3, W.shape[1])):
                ax.plot(W[:1000, i], alpha=0.7, label=f"NMF{i + 1}")
            ax.set_xlabel("Time (days)")
            ax.set_ylabel("Weight")
            ax.set_title("NMF: Temporal Patterns")
            ax.legend()
            plot_idx += 1

        # UMAP embedding
        if "umap" in self.results and self.results["umap"] is not None:
            ax = axes[plot_idx]
            embedding = self.results["umap"]["embedding"]
            if embedding.shape[1] >= 2:
                scatter = ax.scatter(
                    embedding[:, 0],
                    embedding[:, 1],
                    c=np.arange(len(embedding)),
                    cmap="viridis",
                    alpha=0.5,
                    s=1,
                )
                plt.colorbar(scatter, ax=ax, label="Time")
                ax.set_xlabel("UMAP 1")
                ax.set_ylabel("UMAP 2")
                ax.set_title("UMAP: 2D Embedding of Time Series")
            plot_idx += 1

        # Common mode
        if "common_mode" in self.results and self.results["common_mode"] is not None:
            ax = axes[plot_idx]
            # Plot first common mode for East component
            modes = self.results["common_mode"]["temporal_modes"][0]
            ax.plot(modes[:, 0], label="Common Mode 1")
            ax.set_xlabel("Time (days)")
            ax.set_ylabel("Amplitude")
            ax.set_title("Common Mode: East Component")
            ax.legend()
            plot_idx += 1

        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.show()

    def summarize_results(self):
        """
        Print summary of all analyses
        """
        print("\n" + "=" * 60)
        print("SUMMARY OF DIMENSIONALITY REDUCTION RESULTS")
        print("=" * 60)

        for method, result in self.results.items():
            if result is not None:
                print(f"\n{method.upper()}:")

                if method == "pca":
                    print(
                        f"  - Components needed for 95% variance: {np.argmax(result['cumulative_variance'] >= 0.95) + 1}"
                    )
                    print(
                        f"  - First component explains: {result['explained_variance_ratio'][0]:.2%}"
                    )

                elif method == "ica":
                    print(
                        f"  - Number of independent components: {result['sources'].shape[1]}"
                    )

                elif method == "dmd":
                    print(f"  - Number of dynamic modes: {len(result['eigenvalues'])}")
                    print(
                        f"  - Most stable mode growth rate: {np.max(np.abs(result['eigenvalues'])):.4f}"
                    )

                elif method == "nmf":
                    print(
                        f"  - Reconstruction error: {result['reconstruction_error']:.4f}"
                    )

                elif method == "umap":
                    print(f"  - Embedding dimension: {result['embedding'].shape[1]}")

                elif method == "vae":
                    print(f"  - Latent dimension: {result['latent_dim']}")

                elif method == "common_mode":
                    print(f"  - Common modes per component: {result['n_components']}")


def main():
    """
    Main execution function
    """
    print("=" * 60)
    print("GPS TIME SERIES DIMENSIONALITY REDUCTION ANALYSIS")
    print("=" * 60)

    # Generate synthetic data (replace with your real data)
    print("\nGenerating synthetic GPS data...")
    data_generator = GPSDataGenerator()
    gps_data, time_array = data_generator.generate_synthetic_data(
        n_stations=400, n_timesteps=7300, n_components=3
    )

    print(f"Data shape: {gps_data.shape}")
    print(f"  - {gps_data.shape[0] // 3} stations with 3 components each")
    print(f"  - {gps_data.shape[1]} daily observations (~20 years)")

    # Initialize analysis
    analyzer = DimensionalityReductionAnalysis(gps_data, verbose=True)

    # Run all analyses
    print("\nStarting comprehensive analysis...")

    # Linear methods
    analyzer.pca_analysis(n_components=20)
    analyzer.ica_analysis(n_components=20)

    # Time series specific methods
    analyzer.dmd_analysis(svd_rank=20)
    analyzer.ssa_analysis(window_size=365, n_groups=10)

    # Matrix factorization
    analyzer.nmf_analysis(n_components=15)

    # Nonlinear methods
    analyzer.umap_analysis(n_components=3)
    analyzer.vae_analysis(latent_dim=10, n_epochs=30)

    # Geophysical methods
    analyzer.eof_analysis(n_eofs=20)
    analyzer.common_mode_analysis(n_components=5)

    # Summarize results
    analyzer.summarize_results()

    # Plot results
    print("\nGenerating visualizations...")
    analyzer.plot_results()

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

    # Export results example
    print("\nTo access specific results:")
    print("  - PCA scores: analyzer.results['pca']['scores']")
    print("  - ICA sources: analyzer.results['ica']['sources']")
    print("  - DMD modes: analyzer.results['dmd']['modes']")
    print("  - etc.")

    return analyzer


if __name__ == "__main__":
    # Run the analysis
    analyzer = main()

    # To load your own data, replace the synthetic data generation with:
    # gps_data = np.load('your_gps_data.npy')  # Shape: (1200, 7300)
    # analyzer = DimensionalityReductionAnalysis(gps_data)
