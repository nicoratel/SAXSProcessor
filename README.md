---

# 🧪 SAXSProcessor

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyFAI](https://img.shields.io/badge/Powered%20by-PyFAI-orange)](https://pyfai.readthedocs.io)

A Python toolkit for processing and analyzing **Small-Angle X-ray Scattering (SAXS)** data from multiple beamlines and detectors — including **ESRF ID02**, **SOLEIL SWING**, and **LGC (EDF format)**.

This package supports:

* Unified file readers for EDF and HDF5-based formats
* 2D/1D SAXS visualization
* Azimuthal and radial profile extraction using **PyFAI**
* Peak detection and correlation distance calculation
* Orientation order parameter determination
* Batch processing of SAXS datasets

---

## 📦 Supported Instruments

| Instrument     | Format          | Class          |
| -------------- | --------------- | -------------- |
| ESRF – ID02    | `.h5` (Eiger2)  | `h5File_ID02`  |
| SOLEIL – SWING | `.h5` or `.nxs` | `h5File_SWING` |
| LGC            | `.edf`          | `EdfFile`      |

---

## 🧩 Class Overview

### `EdfFile`

Handles `.edf` files from LGC.
Extracts metadata such as wavelength, pixel size, detector center, and sample distance.
Automatically detects “line eraser” dual-frame files.

---

### `h5File_ID02`

Reads **HDF5 data from ESRF ID02** beamline.

* Extracts metadata from HDF5 groups (wavelength, detector geometry, binning, etc.)
* Loads single or multi-frame data arrays
* Parses sample name and magnetic field (`B` value)

---

### `h5File_SWING`

Reads **SWING beamline data (SOLEIL)**.

* Retrieves beamline parameters and detector setup
* Averages multiple frames if requested
* Converts HDF5 datasets to EDF format with `convert2edf()`

---

## ⚙️ `SAXSDataProcessor`

The core class that performs SAXS **data extraction, processing, visualization, and structural analysis**.
It integrates all instrument-specific readers seamlessly.

### Initialization

```python
SAXSDataProcessor(
    file: str,
    instrument: str = 'ID02',
    qvalues: np.ndarray = [0.034, 0.068],
    threshold: float = 0.0001,
    binning: int = 1,
    mask: str = None
)
```

| Parameter    | Description                                       |
| ------------ | ------------------------------------------------- |
| `file`       | Path to the SAXS data file                        |
| `instrument` | `'ID02'`, `'SWING'`, or `'LGC'`                   |
| `qvalues`    | List of q-values for azimuthal profile extraction |
| `mask`       | Optional EDF mask file                            |
| `binning`    | Downsampling factor                               |
| `threshold`  | Relative q-value tolerance                        |

---

## 🧠 Key Methods

### 🖼 Visualization

#### `plot2d_vsq(q_range=[0,0.2], cmap='jet', log=True, q_circles=None, ...)`

Generates a **2D SAXS intensity map (Qx, Qz)** with optional log scale and q-circle overlays.

#### `downsample_image()`

Downsamples image data by block averaging.

#### `caving()` / `caving2(max_iter=10)`

Symmetry-based correction of masked pixels (“caving”) using beam center reflection.

---

### 📈 Data Extraction

#### `pyFAI_extract_azimprofiles(qvalue)`

Extracts **azimuthal intensity profiles** at a given q-value using PyFAI.
Outputs data in `.dat` format (`chi`, `I(chi)`).

#### `pyFAI_extract_radialprofiles(azimuth=90, width=40, caving=False)`

Extracts **radial profiles I(q)** along a specified azimuthal sector.

---

### 🔍 Structural Analysis

#### `detect_all_peaks_by_second_derivative(q, I, nb_peaks=1, ...)`

Detects one or multiple **diffraction peaks** in `I(q)` using second derivative curvature analysis.

#### `compute_correlation_distance(nb_peaks=1, azimuth=90, width=40, plot=False)`

Computes **correlation distances**:
[
d = \frac{2\pi}{q_\text{peak}}
]
based on detected peak positions.

---

### 📐 Orientation Analysis

#### `compute_S()`

Fits azimuthal profiles with a **pseudo-Voigt function** and calculates the **orientational order parameter S** for each q-value.

Returns a dictionary:

```python
{
  qvalue: [y0, I, x0, x0_S, gamma, eta, slope, S, R²]
}
```

Associated functions:

* `pseudo_voigt()` – mixed Lorentz-Gauss peak model
* `calc_S()` – integrates over P₂ to compute S
* `pv_nobckgd()`, `P2_nobckgd()` – auxiliary fitting functions

---

### 📉 Power-law Slope

#### `slope_determination(qmin=0.01, qmax=0.1)`

Fits `I(q)` in log–log space to estimate the scattering **slope exponent**, useful for Porod or fractal regimes.

---

### 📊 Profile Plotting

#### `plot_azim_profiles()`

Plots azimuthal intensity profiles for all q-values defined in the processor.

---

## 🧮 `BatchSAXSDataProcessor`

Performs **automated batch processing** of multiple SAXS files.

Key features:

* Loops through a directory of `.h5` or `.edf` files
* Runs `compute_S()` and `compute_correlation_distance()` for each sample
* Exports results to a CSV summary
* Logs failed fits and errors

Outputs:

* `SAXS_processed.csv`
* `BatchAzimProfileExtraction.log`
* 2D image PNGs (`png_images/*.png`)

---

## 💾 Typical Output Files

| Folder                           | Content                    |
| -------------------------------- | -------------------------- |
| `azimuthal_profiles/`            | `*.dat` azimuthal profiles |
| `radial_profiles/`               | `*.dat` radial profiles    |
| `png_images/`                    | 2D scattering maps         |
| `SAXS_processed.csv`             | Summary of all results     |
| `BatchAzimProfileExtraction.log` | Processing log             |

---

## 🚀 Example Usage

```python
from SAXSProcessor import SAXSDataProcessor

# Initialize processor
proc = SAXSDataProcessor(
    "sample_eiger2_raw.h5",
    instrument="ID02",
    qvalues=[0.03, 0.06],
    mask="mask.edf"
)

# Plot 2D scattering with q-circles
proc.plot2d_vsq(q_range=[0, 0.2], q_circles=[0.03, 0.06])

# Compute orientation order and correlation distances
S_results = proc.compute_S()
d_corr = proc.compute_correlation_distance(plot=True)
```

---

## 🧭 Batch Processing Example

```python
from SAXSProcessor import BatchSAXSDataProcessor

batch = BatchSAXSDataProcessor(
    path="data/ID02/",
    instrument="ID02",
    azimqvalues=[0.034, 0.068],
    mask="mask.edf"
)
df = batch.create_dataframe()
print(df.head())
```

---

## 📚 Dependencies

* `numpy`, `scipy`, `matplotlib`, `pandas`
* `h5py`, `fabio`, `pyFAI`, `skimage`
* `ase` (for structural data import)


```

---

Would you like me to **add a short “Quick Start” section** with installation + usage commands (for users cloning this from GitHub)? It would make it look even more polished.
