# SAXS Nematic Order Parameter Analysis

A Python toolkit for processing Small-Angle X-ray Scattering (SAXS) data and computing nematic order parameters from synchrotron micro-SAXS experiments, specifically designed for the SWING beamline at SOLEIL.

## Overview

This package provides tools to:
- Process SAXS diffraction patterns from HDF5 and EDF files
- Extract azimuthal intensity profiles at specific q-values
- Calculate cylinder form factors with polydispersity
- Fit experimental data to determine nematic order parameters
- Generate 2D orientation maps from linescan experiments
- Visualize nematic order and nanorods alignment

## Features

### Core Capabilities
- **Automated workflow** for micro-SAXS linescan data processing
- **Nematic order parameter (S)** computation with customizable fitting
- **Orientation mapping** with quiver plots showing local alignment
- **Quality filtering** based on R² goodness-of-fit thresholds


### Analysis Pipeline
1. Extract azimuthal profiles from 2D SAXS patterns
2. Determine main nanorod orientation
3. Calculate theoretical cylinder form factors
4. Fit experimental profiles to extract nematic order parameters using Maier Sapue distribution
5. Generate spatially-resolved maps

## Installation

### Prerequisites
```bash
pip install numpy scipy matplotlib pandas tqdm ipython
```

### Required Custom Modules
This package depends on the following modules (should be in your Python path):
- `filereaders` - HDF5 file handling for SWING beamline
- `saxsprocessor` - SAXS data processing and analysis
- `nematicordercalculator` - Form factor and order parameter calculations

## Usage

### Basic Nematic Order Calculation

```python
from saxsprocessor import SAXSProcessor
from utilities import compute_nematic_parameter

# Initialize processor with SAXS image
processor = SAXSProcessor(file='your_saxs_image.edf', mask=mask, instrument='LGC')

# Compute nematic order parameter
S, results = compute_nematic_parameter(
    processor=processor,
    qvalue=0.034,        # q-value for analysis (Å⁻¹)
    threshold=0.05,      # relative q-range threshold
    radius=78,           # cylinder radius (Å)
    L=840,               # cylinder length (Å)
    radius_pd=0.3,       # radius polydispersity
    L_pd=0.75,           # length polydispersity
    plot=True,           # show diagnostic plots
    apply_mirror=False,  # mirror incomplete profiles
    verbose=True
)

print(f"Nematic order parameter S = {S:.3f}")
print(f"Goodness of fit R² = {results['R2']:.3f}")
```

### Processing Complete Linescan (SWING Beamline)

```python
from utilities import compute_nematic_order_assembly_SWING

# Process entire linescan dataset
x_2d, z_2d, orientation_2d, S_2d, R2_2d = compute_nematic_order_assembly_SWING(
    h5path='/path/to/h5/files/',
    mask=your_mask_array,
    qvalue=0.034,
    threshold=0.05,
    radius=78,
    L=840,
    radius_pd=0.3,
    L_pd=0.75,
    plot=False,
    apply_mirror=False,
    verbose=True
)
```

This function will:
1. Convert all HDF5 files to EDF format
2. Process each SAXS pattern sequentially
3. Compute nematic order at each position
4. Export results to CSV
5. Generate an orientation map with quiver overlay

### Visualization from Saved Results

```python
from utilities import plot_from_csv, plot_nematic_order_map

# Generate map from CSV file
plot_from_csv(
    csvpath='/path/to/nematic_order_results.csv',
    R2_threshold=0.9  # filter low-quality fits
)

# Or plot from computed arrays
plot_nematic_order_map(
    x_2d, z_2d, orientation_2d, S_2d, R2_2d,
    R2_threshold=0.9,
    outputpath='/output/directory/'
)
```

## Function Reference

### `compute_nematic_parameter()`

Compute nematic order parameter from a single SAXS pattern.

**Parameters:**
- `processor` (SAXSProcessor): Initialized processor instance
- `qvalue` (float): q-value for azimuthal profile extraction (Å⁻¹)
- `threshold` (float): relative threshold for q-range (default: 0.05)
- `radius` (float): cylinder radius in Ångstroms (default: 78)
- `L` (float): cylinder length in Ångstroms (default: 840)
- `radius_pd` (float): radius polydispersity ratio (default: 0.3)
- `L_pd` (float): length polydispersity ratio (default: 0.75)
- `plot` (bool): show diagnostic plots (default: False)
- `apply_mirror` (bool): apply mirror symmetry to profiles (default: False)
- `verbose` (bool): print progress messages (default: True)

**Returns:**
- `S` (float): nematic order parameter
- `results` (dict): full fitting results including I₀, m, x₀, a, b, S, R², I_model

### `compute_nematic_order_assembly_SWING()`

Process complete micro-SAXS linescan from SWING beamline.

**Parameters:**
- `h5path` (str): path to directory containing HDF5 files
- `mask` (array): detector mask array
- `qvalue`, `threshold`, `radius`, `L`, `radius_pd`, `L_pd`: same as above
- `plot`, `apply_mirror`, `verbose`: same as above

**Returns:**
- `x_2d` (array): 2D array of x positions (mm)
- `z_2d` (array): 2D array of z positions (mm)
- `orientation_2d` (array): 2D array of orientations (degrees)
- `S_2d` (array): 2D array of nematic order parameters
- `R2_2d` (array): 2D array of fit quality (R²)

**Output Files:**
- `nematic_order_results.csv`: tabulated results
- `nematic_orientation_map.png`: visualization with orientation vectors

### `plot_nematic_order_map()`

Generate interpolated nematic order map with quality filtering.

**Parameters:**
- `x_2d`, `z_2d`, `orientation_2d`, `S_2d`, `R2_2d`: 2D arrays from processing
- `R2_threshold` (float): minimum R² for valid data points (default: 0.9)
- `outputpath` (str): directory for output figure (optional)

**Features:**
- Cubic interpolation for smooth visualization
- Quality-based masking of unreliable fits
- Quiver overlay showing local orientation
- Customizable colormaps and scaling

### `plot_from_csv()`

Generate map directly from saved CSV results.

**Parameters:**
- `csvpath` (str): path to CSV results file
- `R2_threshold` (float): minimum R² threshold (default: 0.9)

## Data Format

### Input Requirements
- **HDF5 files**: SWING beamline format with naming pattern `*lacroix_XXX.h5`
- **EDF files**: Processed frames with naming pattern `*File_XXX_Img_YYY.edf`
- **Mask**: 2D numpy array matching detector dimensions

### Output CSV Structure
```
File number, samplename, B (mT), x (mm), z (mm), orientation (°), R2, I0, m, x0, a, b, S
```

## Physical Model

The analysis is based on a **cylindrical particle model** with:
- Form factor calculation including size polydispersity
- Nematic order parameter S ranging from 0 (isotropic) to 1 (perfect alignment)
- Orientation angle χ measured from horizontal axis

The fitting function combines:
- Theoretical cylinder form factor
- Nematic order weighting: P₂(cos θ) = (3cos²θ - 1)/2
- Background and scaling parameters

## Workflow Example

see Workflow_microSAXS.ipynb notebook

## Tips and Best Practices

1. **q-value selection**: Choose q corresponding to the first-order peak of your structure
2. **Threshold tuning**: Adjust to capture sufficient azimuthal range without noise
3. **Form factor parameters**: Measure or estimate cylinder dimensions from TEM
4. **Polydispersity**: Include realistic size distributions for better fits
5. **Quality control**: Always check R² values; filter out R² < 0.9 for reliable maps
6. **Mirror symmetry**: Use when beam geometry creates incomplete azimuthal coverage

## Troubleshooting

**Low R² values:**
- Check form factor parameters match your system
- Verify q-value selection
- Adjust polydispersity parameters
- Ensure proper background subtraction

**Missing data in maps:**
- Increase interpolation density
- Lower R² threshold (with caution)
- Check for detector artifacts or beam shadows

**File conversion errors:**
- Verify HDF5 file structure matches SWING format
- Ensure sufficient disk space for EDF files
- Check file naming patterns



For questions or issues, please contact [your contact information]

## Acknowledgments

Developed for analysis of nematic liquid crystal systems and aligned soft materials using synchrotron SAXS data from the SWING beamline at SOLEIL.
