# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains analysis tools for GPS/GNSS time series data from Japan, focusing on dimensionality reduction techniques for earthquake detection and tectonic motion analysis. The codebase processes 400 GPS stations with 7300 daily observations (~20 years) to identify seismic events and ground deformation patterns.

## Key Data Files

- `gps_arrays_contiguous_N400_K5.npz`: NumPy archive containing GPS station data arrays (names, dates, coordinates, displacement measurements)
- `gnss_dataclass_contiguous_N400_K5.pkl`: Pickle file with GNSS dataclass objects
- `cmt_dataclass.pkl`: Pickle file with CMT (Centroid Moment Tensor) earthquake catalog data
- `F3_offset_var221231.csv`: Equipment maintenance offset records

## Common Development Commands

### Running Analysis
```bash
python japan_time_approaches.py  # Run comprehensive dimensionality reduction analysis
```

### Jupyter Notebooks
```bash
jupyter notebook analyze_gps_time_series.ipynb  # GPS time series visualization
jupyter notebook make_gnss_offset.ipynb  # Generate offset matrices for earthquakes/maintenance
```

### Required Dependencies Installation
```bash
# Core packages
pip install numpy pandas matplotlib scikit-learn cloudpickle

# Optional packages for advanced methods
pip install pydmd  # Dynamic Mode Decomposition
pip install pyts  # Singular Spectrum Analysis
pip install umap-learn  # UMAP dimensionality reduction
pip install torch  # Variational Autoencoder
pip install eofs  # Empirical Orthogonal Functions
```

## Code Architecture

### Core Analysis Pipeline
1. **Data Loading**: GPS data from `.npz` files, dataclasses from pickles
2. **Preprocessing**: Standardization, offset detection (earthquakes + maintenance)
3. **Dimensionality Reduction**: Multiple techniques applied to identify patterns:
   - Linear: PCA, ICA, EOF
   - Time-series specific: DMD, SSA
   - Nonlinear: UMAP, VAE
   - Geophysical: Common Mode Analysis
4. **Visualization**: Time series plots with event markers, dimensionality reduction results

### Key Classes
- `GPSDataGenerator`: Creates synthetic GPS data for testing
- `DimensionalityReductionAnalysis`: Main analysis class implementing all reduction methods
- `GPS_VAE`: PyTorch model for variational autoencoder analysis

### Earthquake Offset Detection
Uses distance threshold formula: $10^{0.36M_W - 0.15}$ km from epicenter
- M6.0: 102 km radius
- M7.0: 234 km radius  
- M8.0: 537 km radius
- M9.0: 1230 km radius

### Data Structure
GPS arrays shape: `(n_stations × n_components, n_timesteps)`
- 400 stations × 3 components (East, North, Up) = 1200 features
- 7300 timesteps (daily observations)