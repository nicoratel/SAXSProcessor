# SAXS Analysis Toolkit

This repository contains complementary Python modules designed for Small-Angle X-ray Scattering (SAXS) data analysis.  
Together, they provide tools for detector-corrected preprocessing, azimuthal/radial integration, nematic order parameter extraction, and correlation distance analysis.

### Note to the user: The library is designed to unwrap images contained in a h5 file, and to average them. In case you do not wish to average your data (e.g. spatial scans), please refer to µSAXS case, described in Workflow_microSAXS.ipynb.
---

# Module Overview

## 1. `saxsprocessor.py` — Core SAXS Data Processing Engine  
The `SAXSProcessor` class provides a unified interface to load and process 2D SAXS detector images from several beamlines (ID02, SWING, LGC).  
It handles:

### Key Features
- Reading EIGER / HDF5 / EDF data via dedicated file readers.
- Automatic background subtraction with optimized scaling factor *k*.
- Pixel binning.
- Beam-stop / dead-pixel masking and mask-based “caving”.
- Generation of 2D reciprocal-space (Qx, Qz) images.
- Extraction of:
  - Azimuthal profiles at constant q.
  - Radial profiles within specific azimuthal sectors.
- Export of SasView-compatible 2D ASCII files.
- Computation of full Qx, Qy, Qz detector grids based on geometry.




### Main Methods
- `export_sasview()`
- `extract_azimuthal_profile(qvalue, threshold, ...)`
- `extract_radial_profile(azimuth, width, ...)`
- `plot2d_vsq()`

---

## 2. `correlationdistancecalculator.py` — Peak Detection & Correlation Distances  
The `CorrelationDistanceCalculator` analyzes radial profiles to extract structural distances, typically from Bragg or correlation peaks.

### Key Features
- Savitzky–Golay smoothing and second-derivative peak detection.
- Automatic prominence filtering and peak ranking.
- Detection of the strongest q-peaks in the profile.
- Conversion of q-peaks into real-space distances (d = 2π/q).
- Structural anisotropy analysis using azimuth-dependent radial scans.
- Optional visualization of all steps.

### Main Methods
- `detect_peaks(q, I, ...)`
- `compute_correlation_distances(nb_peaks, azimuth, ...)`
- `analyze_anisotropy(azimuth_list, ...)`

Used for lamellar spacing, correlation lengths, periodic structures, and directional anisotropy.

---

## 3. `nematicordercalculator.py` — Nematic Order Parameter from Azimuthal Profiles  
The `NematicOrderCalculator` computes the nematic order parameter **S** using a Maier–Saupe angular distribution, optionally convolved with a cylinder form factor via sasmodels.

### Key Features
- Pure Maier–Saupe model fitting (no form factor).
- Optional convolution with a 2D cylinder form factor
- Automatic extraction of form-factor azimuthal profiles at each q.
- Nonlinear curve fitting with confidence metrics (R², covariance).
- Extraction of:
  - S (nematic order parameter)
  - m (distribution sharpness)
  - x₀ (director angle)
  - Fit amplitude, background parameters assumed linear)

### Included Classes

#### `NematicOrderCalculator`
- `ms_distribution()`
- `compute_S()`
- `fit_azimuthal_profile(theta_exp, I_exp, ...)`
- `convolve_with_form_factor()`
- `mirror_profile()`

#### `CylinderFormFactor`
- Computes 2D cylinder form factor using sasmodels.
- Inherits detector geometry from a `SAXSProcessor`.
- Provides azimuthal form-factor profiles for convolution.

#### `BatchNematic`
High-level batch processor:
- Iterates over all SAXS files in a folder.
- Extracts azimuthal profiles for each q.
- Fits the nematic order parameter S.
- Produces:
  - A full CSV table of results
  - A log file
  - Per-profile data files

---



