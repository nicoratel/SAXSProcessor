from filereaders import h5File_SWING
from saxsprocessor import SAXSProcessor
import glob
import os
import re
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from IPython.display import clear_output, display, Markdown
from scipy.interpolate import griddata
from utilities import compute_nematic_parameter
import multiprocessing
from joblib import Parallel, delayed
from correlationdistancecalculator import CorrelationDistanceCalculator
import ast




#####################################################################################################################

        # FUNCTIONS DEDICATED TO THE WORKFLOW FOR MICRO-SAXS DATA PROCESSING

# a single file corresponds to a linescan along x (multiple frames along x for a given z position)

#####################################################################################################################
def view_position_grid(
    h5path: str,
    prefix: str = 'lacroix',
    basler_coords: tuple = (186, 309),
    basler_calibration: tuple = (3.7, 3.7),
    reference_file: str = None,
    marker_size: float = 3,
    marker_color: str = 'yellow',
    marker_style: str = 'o',
    figsize: tuple = (12, 12),
    cmap: str = 'gray',
    contrast_adjust: bool = False,
    contrast_percentiles: tuple = (5, 95),
    plot: bool = True,
    save_image: bool = False,
    output_filename: str = None,
    verbose: bool = False
):
    """
    Visualize all acquisition positions on a Basler reference image.
    
    Parameters
    ----------
    h5path : str
        Path to folder containing HDF5 files    basler_coords : tuple
        Reference position (x, y) in pixels on Basler image
    basler_calibration : tuple
        Calibration in µm/pixel (x, z)
    prefix : str
        Prefix for file selection (default: 'lacroix'). Files matching '{prefix}_*.h5' will be processed
    reference_file : str, optional
        Path to specific reference file. If None, uses first file
    marker_size : float
        Size of position markers (default: 3)
    marker_color : str
        Color of position markers (default: 'yellow')
    marker_style : str
        Matplotlib marker style (default: 'o')
    figsize : tuple
        Figure size in inches (default: (12, 12))
    cmap : str
        Colormap for Basler image (default: 'gray')
    contrast_adjust : bool
        Apply contrast adjustment using percentiles (default: False)
    contrast_percentiles : tuple
        Percentiles for contrast adjustment (default: (5, 95))
    plot : bool
        Display the plot (default: True)
    save_image : bool
        Save the figure to file (default: False)
    output_filename : str, optional
        Custom output filename. If None, auto-generates name
    verbose : bool
        Print detailed information (default: True)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'positions_x': list of x positions in pixels
        - 'positions_z': list of z positions in pixels
        - 'file_numbers': list of file numbers
        - 'total_frames': total number of frames processed
        - 'figure': matplotlib figure object (if plot=True)
        - 'stats': dictionary with position statistics
    """    
    # Find all files
    file_pattern = f'{prefix}_*.h5'
    all_files = sorted(glob.glob(os.path.join(h5path, file_pattern)))
    
    if len(all_files) == 0:
        raise FileNotFoundError(f"No files found matching pattern '{file_pattern}' in {h5path}")
    
    if verbose:
        print(f"{'='*60}")
        print(f"VIEW POSITION GRID")
        print(f"{'='*60}")
        print(f"Data folder: {h5path}")
        print(f"Prefix: {prefix}")
        print(f"File pattern: {file_pattern}")
        print(f"Files found: {len(all_files)}")
    
    # Load reference Basler image
    if reference_file is None:
        reference_file = all_files[0]
    
    if verbose:
        print(f"\nReference file: {os.path.basename(reference_file)}")
    
    swing_ref = h5File_SWING(file=reference_file, mean=True)
    basler_ref = swing_ref.basler_image
    
    # Extract positions from all files and frames
    positions_x = []
    positions_z = []
    file_numbers = []
    
    x_ref_mm = swing_ref.position_x_start[0]
    z_ref_mm = swing_ref.position_z_start[0]
    
    if verbose:
        print(f"\nReference position: X={x_ref_mm:.4f} mm, Z={z_ref_mm:.4f} mm")
        print(f"Basler calibration: {basler_calibration[0]} µm/pixel, {basler_calibration[1]} µm/pixel")
        print(f"Basler reference coords: {basler_coords}")
        print(f"\nExtracting positions from all frames...")
    
    total_frames = 0
    for i, file in enumerate(all_files):
        try:
            # Load without averaging to get frame count
            swing = h5File_SWING(file=file, mean=False)
            nb_frames = swing.nb_frames
            
            # Start and end positions
            x_start = swing.position_x_start[0]
            z_start = swing.position_z_start[0]
            x_end = swing.position_x_end[0]
            z_end = swing.position_z_end[0]
            
            # Calculate step per frame (in mm)
            step_x_mm = (x_end - x_start) / nb_frames if nb_frames > 1 else 0
            step_z_mm = (z_end - z_start) / nb_frames if nb_frames > 1 else 0
            
            # Process each frame
            for frame_idx in range(nb_frames):
                # Position of this frame
                x_pos = x_start + frame_idx * step_x_mm
                z_pos = z_start + frame_idx * step_z_mm
                
                # Calculate displacement from reference (mm)
                delta_x_mm = x_pos - x_ref_mm
                delta_z_mm = z_pos - z_ref_mm
                
                # Convert to pixels: mm -> µm -> pixels
                delta_x_pixels = (delta_x_mm * 1000) / basler_calibration[0]
                delta_z_pixels = (delta_z_mm * 1000) / basler_calibration[1]
                
                # Calculate position in pixels on Basler image
                x_pixel = basler_coords[0] + int(delta_x_pixels)
                z_pixel = basler_coords[1] + int(delta_z_pixels)
                
                positions_x.append(x_pixel)
                positions_z.append(z_pixel)
                file_numbers.append(swing.file_number)
                
                total_frames += 1
                
                # Display some examples
                if verbose and ((i < 2 and frame_idx < 3) or (i >= len(all_files) - 1 and frame_idx < 3)):
                    print(f"  File #{swing.file_number}, frame {frame_idx}: "
                          f"X={x_pos:.4f} mm, Z={z_pos:.4f} mm | Pixel=({x_pixel}, {z_pixel})")
            
            if verbose and i == 2:
                print(f"  ...")
                
        except Exception as e:
            print(f"  Error with {os.path.basename(file)}: {e}")
    
    # Calculate statistics
    stats = {
        'x_min': min(positions_x),
        'x_max': max(positions_x),
        'x_range': max(positions_x) - min(positions_x),
        'z_min': min(positions_z),
        'z_max': max(positions_z),
        'z_range': max(positions_z) - min(positions_z),
        'total_points': len(positions_x),
        'file_range': (min(file_numbers), max(file_numbers))
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"STATISTICS")
        print(f"{'='*60}")
        print(f"Total positions extracted: {stats['total_points']} ({total_frames} frames)")
        print(f"Position X: min={stats['x_min']}, max={stats['x_max']}, range={stats['x_range']} px")
        print(f"Position Z: min={stats['z_min']}, max={stats['z_max']}, range={stats['z_range']} px")
        print(f"File numbers: #{stats['file_range'][0]} to #{stats['file_range'][1]}")
        print(f"{'='*60}")
    
    # Create figure
    fig = None
    if plot or save_image:
        fig = plt.figure(figsize=figsize)
        
        # Apply contrast adjustment if requested
        if contrast_adjust:
            vmin = np.percentile(basler_ref, contrast_percentiles[0])
            vmax = np.percentile(basler_ref, contrast_percentiles[1])
            plt.imshow(basler_ref, cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            plt.imshow(basler_ref, cmap=cmap)
        
        # Plot all positions
        plt.plot(positions_x, positions_z, 
                marker=marker_style, 
                color=marker_color, 
                markersize=marker_size, 
                markeredgewidth=0,
                linestyle='none')
        
        plt.title(f'Position Grid Mapping\n'
                  f'Reference: {swing_ref.samplename} (#{swing_ref.file_number})\n'
                  f'{stats["total_points"]} positions')
        plt.xlabel('Pixel X')
        plt.ylabel('Pixel Y')
        plt.tight_layout()
        
        # Save if requested
        if save_image:
            if output_filename is None:
                output_filename = os.path.join(
                    h5path, 
                    f"position_grid_{swing_ref.samplename}.png"
                )
            fig.savefig(output_filename, dpi=300, bbox_inches='tight')
            if verbose:
                print(f"\nImage saved: {output_filename}")
        
        # Show if requested
        if plot:
            plt.show()
        else:
            plt.close(fig)
    
    # Return results
    return {
        'positions_x': positions_x,
        'positions_z': positions_z,
        'file_numbers': file_numbers,
        'total_frames': total_frames,
        'figure': fig,
        'stats': stats,
        'reference_image': basler_ref,
        'reference_sample': swing_ref.samplename
    }

def sort_h5(file, prefix='lacroix'):
    """
    Docstring pour sort_h5
    
    :param file: path to file
    :param prefix: h5 file prefix (e.g., 'lacroix_00001.h5' -> prefix = 'lacroix')
    """
    # Pattern plus flexible : accepte prefix_XXXXX suivi de n'importe quoi (dates, etc.)
    pattern = rf'{re.escape(prefix)}_(\d+)'
    match = re.search(pattern, file)
    if not match:
        raise ValueError(f"Format de fichier non reconnu: {file}\nAttendant format: {prefix}_XXXXX")
    return int(match.group(1))

def sort_edf(file):
    match = re.search(r'File_(\d+)_Img_(\d+)', file)
    if not match:
        raise ValueError("Format de fichier non reconnu")

    return int(match.group(1)), int(match.group(2))

def _process_one_edf(
    i,
    file,
    reference_file,
    k,
    autosubstract,
    mask,
    qvalue,
    threshold,
    radius,
    L,
    radius_pd,
    L_pd,
    apply_mirror,
    verbose
):
    proc = SAXSProcessor(
        file=file,
        reference_file=reference_file,
        k=k,
        autosubstract=autosubstract,
        mask=mask,
        instrument='LGC'
    )

    S, orientation, results = compute_nematic_parameter(
        processor=proc,
        qvalue=qvalue,
        threshold=threshold,
        radius=radius,
        L=L,
        radius_pd=radius_pd,
        L_pd=L_pd,
        plot=False,               # <<< IMPORTANT
        apply_mirror=apply_mirror,
        verbose=verbose
    )

    row_data = {
        'File number': proc.file_number,
        'samplename': proc.samplename,
        'B (mT)': proc.B,
        'x (mm)': proc.x,
        'z (mm)': proc.z,
        'orientation (°)': orientation,
        'S': S,
        'R2': results['R2']
    }

    for key, value in results.items():
        if key != 'I_model':
            row_data[key] = value

    return (
        i,
        proc.x,
        proc.z,
        orientation,
        S,
        results['R2'],
        row_data
    )

def compute_nematic_order_assembly_SWING(
    h5path,
    prefix='lacroix',
    reference_file=None,
    k=1,
    autosubstract=True,
    mask=None,
    qvalue=0.034,
    threshold=0.05,
    radius=78,
    L=840,
    radius_pd=0.3,
    L_pd=0.75,
    plot=False,
    apply_mirror=None,
    verbose=True
):
    """
    SAXS linescan processing with nematic order parameter computation
    (PARALLELIZED VERSION)
    """

    def log_step(step, total, message):
        clear_output(wait=True)
        display(Markdown(f"### Step {step}/{total}\n**{message}**"))

    print("⚠️ WARNING: This function assumes the data correspond to a linescan.")

    # === STEP 1: h5 files
    log_step(1, 7, "Searching and sorting h5 files")
    h5_filelist = sorted(
        glob.glob(os.path.join(h5path, '*.h5')),
        key=lambda f: sort_h5(f, prefix=prefix)
    )

    # === STEP 2: h5 → EDF
    log_step(2, 7, "Extracting frames and converting to EDF files")
    edfpath = os.path.join(h5path, 'edf_files')
    os.makedirs(edfpath, exist_ok=True)

    for h5file in tqdm(h5_filelist, desc="Converting h5 → EDF"):
        SWING_file = h5File_SWING(h5file, mean=False)
        SWING_file.convert2edf(outputdir=edfpath)

    # === STEP 3: geometry
    log_step(3, 7, "Determining scan geometry")
    number_of_lines = len(h5_filelist)
    number_of_columns = SWING_file.nb_frames

    # === STEP 4: EDF list
    log_step(4, 7, "Building EDF file list")
    edf_filelist = sorted(
        glob.glob(os.path.join(edfpath, '*Img*.edf')),
        key=sort_edf
    )

    # === STEP 5: parallel nematic computation
    log_step(5, 7, "Computing nematic order parameter (parallel)")

    nfiles = len(edf_filelist)

    x_array = np.zeros(nfiles)
    z_array = np.zeros(nfiles)
    orientation_array = np.zeros(nfiles)
    S_array = np.zeros(nfiles)
    R2_array = np.zeros(nfiles)
    data_list = []

    n_jobs = max(1, multiprocessing.cpu_count() - 1)

    results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        verbose=10
    )(
        delayed(_process_one_edf)(
            i,
            file,
            reference_file,
            k,
            autosubstract,
            mask,
            qvalue,
            threshold,
            radius,
            L,
            radius_pd,
            L_pd,
            apply_mirror,
            verbose
        )
        for i, file in enumerate(edf_filelist)
    )

    for i, x, z, orientation, S, R2, row_data in results:
        x_array[i] = x
        z_array[i] = z
        orientation_array[i] = orientation
        S_array[i] = S
        R2_array[i] = R2
        data_list.append(row_data)

    # === STEP 6: CSV export
    log_step(6, 7, "Exporting results to CSV")
    df = pd.DataFrame(data_list)

    outputpath = os.path.join(h5path, 'nematic_processing_results')
    os.makedirs(outputpath, exist_ok=True)

    csv_filename = os.path.join(outputpath, 'nematic_order_results.csv')
    df.to_csv(csv_filename, index=False)

    # === STEP 7: final map
    log_step(7, 7, "Generating final nematic order map")

    orientation_2d = orientation_array.reshape(number_of_lines, number_of_columns)
    S_2d = S_array.reshape(number_of_lines, number_of_columns)
    x_2d = x_array.reshape(number_of_lines, number_of_columns)
    z_2d = z_array.reshape(number_of_lines, number_of_columns)
    R2_2d = R2_array.reshape(number_of_lines, number_of_columns)

    # Points originaux
    x_orig = x_2d.flatten()
    z_orig = z_2d.flatten()
    
    # Étendre légèrement les limites (5% de marge)
    x_margin = (x_orig.max() - x_orig.min()) * 0.05
    z_margin = (z_orig.max() - z_orig.min()) * 0.05
    
    x_min_ext = x_orig.min() - x_margin
    x_max_ext = x_orig.max() + x_margin
    z_min_ext = z_orig.min() - z_margin
    z_max_ext = z_orig.max() + z_margin
    
    # Grille interpolée (2x plus fine)
    x_interp = np.linspace(x_min_ext, x_max_ext, number_of_columns * 2)
    z_interp = np.linspace(z_min_ext, z_max_ext, number_of_lines * 2)
    x_grid, z_grid = np.meshgrid(x_interp, z_interp)
    
    # Interpoler S_2d (seulement les points valides)
    points = np.column_stack((x_orig, z_orig))
    values_S = S_2d.flatten()
    
    #print(f"Interpolation avec {len(values_S)} points")
    
    # Essayer d'abord avec 'linear', puis remplir les NaN avec 'nearest'
    S_interp = griddata(points, values_S, (x_grid, z_grid), method='linear')
    
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(
        S_2d,
        extent=[x_min_ext, x_max_ext, z_min_ext, z_max_ext],
        origin='lower',
        aspect='auto',
        cmap='jet',
        interpolation='bicubic'
    )

    u = np.cos(np.radians(orientation_2d))
    v = np.sin(np.radians(orientation_2d))

    ax.quiver(x_2d, z_2d, u, v, color='black', scale=15, width=0.005)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('S', rotation=270, labelpad=20)

    ax.set_xlabel('X position (mm)')
    ax.set_ylabel('Z position (mm)')
    ax.set_title('Nematic order parameter map')
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(outputpath, 'nematic_orientation_map.png'))
    plt.show()

    return x_2d, z_2d, orientation_2d, S_2d, R2_2d




def plot_nematic_order_map(x_2d, z_2d, orientation_2d, S_2d, R2_2d, R2_threshold=0.9, outputpath=None):
    """
    Generate nematic order parameter map with quiver plot for orientation.
    """
    
    
    if outputpath is None:
        outputpath = os.path.join(os.getcwd(), 'nematic_processing_results')
    os.makedirs(outputpath, exist_ok=True)
    
    # Masquer les valeurs avec R2 < seuil
    number_of_lines, number_of_columns = S_2d.shape
    mask_valid = R2_2d >= R2_threshold
    
    
    # Points originaux
    x_orig = x_2d.flatten()
    z_orig = z_2d.flatten()
    
    # Étendre légèrement les limites (5% de marge)
    x_margin = (x_orig.max() - x_orig.min()) * 0.05
    z_margin = (z_orig.max() - z_orig.min()) * 0.05
    
    x_min_ext = x_orig.min() - x_margin
    x_max_ext = x_orig.max() + x_margin
    z_min_ext = z_orig.min() - z_margin
    z_max_ext = z_orig.max() + z_margin
    
    # Grille interpolée (2x plus fine)
    x_interp = np.linspace(x_min_ext, x_max_ext, number_of_columns * 2)
    z_interp = np.linspace(z_min_ext, z_max_ext, number_of_lines * 2)
    x_grid, z_grid = np.meshgrid(x_interp, z_interp)
    
    # Interpoler S_2d (seulement les points valides)
    points = np.column_stack((x_orig[mask_valid.flatten()], z_orig[mask_valid.flatten()]))
    values_S = S_2d.flatten()[mask_valid.flatten()]
    
    #print(f"Interpolation avec {len(values_S)} points")
    
    # Essayer d'abord avec 'linear', puis remplir les NaN avec 'nearest'
    S_interp = griddata(points, values_S, (x_grid, z_grid), method='linear')
    
    # Remplir les NaN avec la méthode 'nearest' pour éviter les trous
    nan_mask = np.isnan(S_interp)
    if nan_mask.any():
        #print(f"Remplissage de {nan_mask.sum()} valeurs NaN avec la méthode 'nearest'")
        S_interp_nearest = griddata(points, values_S, (x_grid, z_grid), method='nearest')
        S_interp[nan_mask] = S_interp_nearest[nan_mask]
    
    #print(f"S_interp après remplissage: min={np.nanmin(S_interp):.4f}, max={np.nanmax(S_interp):.4f}, NaN count={np.isnan(S_interp).sum()}")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(
        S_interp,
        extent=[x_min_ext, x_max_ext, z_min_ext, z_max_ext],
        origin='lower',
        aspect='auto',
        cmap='jet',
        interpolation='bilinear'
    )
    
    # Calculer u et v seulement pour les points non masqués
    orientation_2d_masked = np.ma.masked_where(~mask_valid, orientation_2d)
    u = np.cos(np.radians(orientation_2d_masked))
    v = np.sin(np.radians(orientation_2d_masked))
    
    # Filtrer les positions pour quiver
    x_valid = x_2d[mask_valid]
    z_valid = z_2d[mask_valid]
    u_valid = u[mask_valid]
    v_valid = v[mask_valid]
    
    if len(x_valid) > 0:
        ax.quiver(
            x_valid, z_valid, u_valid, v_valid,
            color='black',
            scale=15,
            width=0.005,
            alpha=0.8,
            headwidth=3,
            headlength=4
        )
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('S', rotation=270, labelpad=20, fontsize=12)
    
    ax.set_xlabel('X position (mm)', fontsize=14)
    ax.set_ylabel('Z position (mm)', fontsize=14)
    ax.set_title(f'Nematic order parameter map (R² > {R2_threshold})', fontsize=16)
    
    ax.invert_yaxis()
        
    plt.tight_layout()
    figname = os.path.join(outputpath, 'nematic_orientation_map_filtered.png')
    plt.savefig(figname, dpi=300, bbox_inches='tight')
    plt.show()



def plot_from_csv(csvpath, R2_threshold=0.9):
    """
    Generate nematic order parameter map from CSV results file.
    """
    df = pd.read_csv(csvpath)
    
    x_2d = df['x (mm)'].values.reshape(-1, int(df['z (mm)'].nunique()))
    z_2d = df['z (mm)'].values.reshape(-1, int(df['z (mm)'].nunique()))
    orientation_2d = (df['orientation (°)'].values + 90).reshape(-1, int(df['z (mm)'].nunique()))-90
    S_2d = df['S'].values.reshape(-1, int(df['z (mm)'].nunique()))
    R2_2d = df['R2'].values.reshape(-1, int(df['z (mm)'].nunique()))
    
    outputpath = os.path.dirname(csvpath)
    
    plot_nematic_order_map(x_2d, z_2d, orientation_2d, S_2d, R2_2d, R2_threshold=R2_threshold, outputpath=outputpath)



def plot_transmission_map(h5path,prefix='lacroix'):
    """
    Generate transmission map from SWING h5 files.
    """
    h5_filelist = sorted(
        glob.glob(os.path.join(h5path, '*.h5')),
        key=lambda f: sort_h5(f, prefix=prefix)
    )

    number_of_lines = len(h5_filelist)
    number_of_columns = h5File_SWING(h5_filelist[0], mean=False).nb_frames

    x_array = np.zeros((number_of_lines, number_of_columns))
    z_array = np.zeros((number_of_lines, number_of_columns))
    transmission_array = np.zeros((number_of_lines, number_of_columns))

    for i, h5file in enumerate(tqdm(h5_filelist, desc="Extracting transmission data")):
        SWING_file = h5File_SWING(h5file, mean=False)
        SWING_file._extract_from_h5()
        SWING_file._extract_scatteringdata()

        for j in range(SWING_file.nb_frames):
            x_array[i, j] = SWING_file.position_x_start + j * SWING_file.step_x
            z_array[i, j] = SWING_file.position_z_start
            transmission_array[i, j] = SWING_file.transmission[j]
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        transmission_array,
        extent=[x_array.min(), x_array.max(), z_array.min(), z_array.max()],
        origin='lower',
        aspect='auto',
        cmap='jet',
        interpolation='bicubic'
    )
    ax.set_xlabel('X position (mm)', fontsize=14)
    ax.set_ylabel('Z position (mm)', fontsize=14)
    ax.set_title(f'Transmission map', fontsize=16)
    ax.invert_yaxis()
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Transmission', rotation=270, labelpad=20, fontsize=14)
    plt.tight_layout()
    figname = os.path.join(h5path, 'transmission_map.png')
    plt.savefig(figname, dpi=300, bbox_inches='tight')

def average_h5_intensity(
        h5path,
        prefix='lacroix',
        reference_file=None,
        k=1,
        autosubstract=True,
        mask=None,
        verbose=True):
    """
    Calculate average SAXS intensity from multiple h5 files.
    
    This function loads all h5 files matching a prefix pattern, extracts the 
    scattering intensity from each file, and computes the mean intensity across 
    all files.
    
    Parameters
    ----------
    h5path : str
        Path to folder containing h5 files
    prefix : str
        h5 file prefix (default: 'lacroix')
    reference_file : str, optional
        Path to h5 file for reference measurement (default: None)
    k : float
        Coefficient for reference subtraction (default: 1)
    autosubstract : bool
        Use optimized reference subtraction (default: True)
    mask : str, optional
        Path to mask file
    verbose : bool
        Print progress information (default: True)
    
    Returns
    -------
    mean_intensity : ndarray
        Average intensity across all files
    nb_files : int
        Number of files processed
    """
    
    # 1. Create h5_file list
    h5_filelist = sorted(
        glob.glob(os.path.join(h5path, '*.h5')),
        key=lambda f: sort_h5(f, prefix=prefix)
    )
    nb_files = len(h5_filelist)

    # 2. Retrieve mean intensity in each file
    total_intensity = 0
    for file in h5_filelist:
        print(f'Processing {file}')
        proc = SAXSProcessor(file,
                             instrument='SWING',
                             reference_file = reference_file,
                             k = k,
                             autosubstract = autosubstract,
                             mask = mask,                          
                             )
        total_intensity += proc.data
    # 3. Calculate mean intensity
    mean_intensity = total_intensity / nb_files
    
    if verbose:
        print(f"✓ Average intensity computed from {nb_files} files")
    
    return mean_intensity, nb_files


def average_h5_processor(
        h5path,
        prefix='lacroix',
        reference_file=None,
        k=1,
        autosubstract=True,
        mask=None,
        verbose=True):
    """
    Calculate average SAXS intensity from multiple h5 files.
    
    This function loads all h5 files matching a prefix pattern, extracts the 
    scattering intensity from each file, and computes the mean intensity across 
    all files.
    
    Parameters
    ----------
    h5path : str
        Path to folder containing h5 files
    prefix : str
        h5 file prefix (default: 'lacroix')
    reference_file : str, optional
        Path to h5 file for reference measurement (default: None)
    k : float
        Coefficient for reference subtraction (default: 1)
    autosubstract : bool
        Use optimized reference subtraction (default: True)
    mask : str, optional
        Path to mask file
    verbose : bool
        Print progress information (default: True)
    
    Returns
    -------
    SAXSProcessor
        SAXSProcessor instance with average intensity data
    """

       # 1. Create h5_file list
    h5_filelist = sorted(
        glob.glob(os.path.join(h5path, '*.h5')),
        key=lambda f: sort_h5(f, prefix=prefix)
    )
    nb_files = len(h5_filelist)

    # 2. Retrieve mean intensity in each file
    total_intensity = 0
    for file in h5_filelist:
        print(f'Processing {file}')
        proc = SAXSProcessor(file,
                             instrument='SWING',
                             reference_file = reference_file,
                             k = k,
                             autosubstract = autosubstract,
                             mask = mask,                          
                             )
        total_intensity += proc.data
    # 3. Calculate mean intensity
    mean_intensity = total_intensity / nb_files
    
    if verbose:
        print(f"✓ Average intensity computed from {nb_files} files")

    proc = SAXSProcessor(h5_filelist[0],
                         instrument='SWING',
                         reference_file = reference_file,
                         k = k,
                         autosubstract = autosubstract,
                         mask = mask,                          
                         )
    proc.data = mean_intensity
    
    return proc




def compute_nematic_parameter_linescan_SWING(
    h5file,
    reference_file=None,
    k=1,
    autosubstract=True,
    mask=None,
    qvalue=0.034,
    threshold=0.05,
    radius=78,
    L=840,
    radius_pd=0.3,
    L_pd=0.75,
    plot=True,
    apply_mirror=None,
    verbose=True
):
    """
    SAXS linescan processing with nematic order parameter computation for a single h5 file
    (PARALLELIZED VERSION)
    
    Parameters
    ----------
    h5file : str
        Path to the single h5 file containing the linescan data
    ... (autres paramètres identiques à compute_nematic_order_assembly_SWING)
    
    Returns
    -------
    x_array : ndarray
        X positions
    z_array : ndarray
        Z positions (constant for a linescan)
    orientation_array : ndarray
        Orientation angles
    S_array : ndarray
        Nematic order parameters
    R2_array : ndarray
        R² values
    df : DataFrame
        Complete results as DataFrame
    """
    
    def log_step(step, total, message):
        clear_output(wait=True)
        display(Markdown(f"### Step {step}/{total}\n**{message}**"))
    
    print("⚠️ WARNING: Processing single linescan file")
    
    # === STEP 1: Vérifier le fichier h5
    log_step(1, 6, "Checking h5 file")
    if not os.path.exists(h5file):
        raise FileNotFoundError(f"File not found: {h5file}")
    
    # === STEP 2: h5 → EDF
    log_step(2, 6, "Extracting frames and converting to EDF files")
    h5dir = os.path.dirname(h5file)
    edfpath = os.path.join(h5dir, 'edf_files_linescan')
    os.makedirs(edfpath, exist_ok=True)
    
    SWING_file = h5File_SWING(h5file, mean=False)
    SWING_file.convert2edf(outputdir=edfpath)
    
    # === STEP 3: Geometry (1 ligne × N colonnes)
    log_step(3, 6, "Determining scan geometry")
    number_of_columns = SWING_file.nb_frames
    number_of_lines = 1
    
    # === STEP 4: EDF list
    log_step(4, 6, "Building EDF file list")
    edf_filelist = sorted(
        glob.glob(os.path.join(edfpath, '*Img*.edf')),
        key=sort_edf
    )
    
    # === STEP 5: Parallel nematic computation
    log_step(5, 6, "Computing nematic order parameter (parallel)")
    
    nfiles = len(edf_filelist)
    
    x_array = np.zeros(nfiles)
    z_array = np.zeros(nfiles)
    orientation_array = np.zeros(nfiles)
    S_array = np.zeros(nfiles)
    R2_array = np.zeros(nfiles)
    data_list = []
    
    n_jobs = max(1, multiprocessing.cpu_count() - 1)
    
    results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        verbose=10
    )(
        delayed(_process_one_edf)(
            i,
            file,
            reference_file,
            k,
            autosubstract,
            mask,
            qvalue,
            threshold,
            radius,
            L,
            radius_pd,
            L_pd,
            apply_mirror,
            verbose
        )
        for i, file in enumerate(edf_filelist)
    )
    
    for i, x, z, orientation, S, R2, row_data in results:
        x_array[i] = x
        z_array[i] = z
        orientation_array[i] = orientation
        S_array[i] = S
        R2_array[i] = R2
        data_list.append(row_data)
    
    # === STEP 6: CSV export et visualisation
    log_step(6, 6, "Exporting results and generating plots")
    df = pd.DataFrame(data_list)
    
    outputpath = os.path.join(h5dir, 'nematic_linescan_results')
    os.makedirs(outputpath, exist_ok=True)
    
    csv_filename = os.path.join(outputpath, 'nematic_order_linescan.csv')
    df.to_csv(csv_filename, index=False)
    
    if plot:
        # Plot simple: S(x) avec flèches d'orientation
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Tracer S(x)
        ax.plot(x_array, S_array, 'o-', linewidth=2, markersize=8, color='blue', label='S')
        
        # Ajouter les flèches d'orientation à chaque point
        u = np.cos(np.radians(orientation_array))
        v = np.sin(np.radians(orientation_array))
        
        # Échelle des flèches proportionnelle à la plage de S
        S_range = S_array.max() - S_array.min()
        arrow_scale = S_range * 0.15 if S_range > 0 else 0.1
        
        ax.quiver(x_array, S_array, u, v, 
                  color='black',scale=15,width=0.005,alpha=0.8,headwidth=3,headlength=4, label='Orientation')
        
        ax.set_xlabel('X position (mm)', fontsize=12)
        ax.set_ylabel('Nematic order parameter S', fontsize=12)
        ax.set_title('Nematic order parameter and orientation along linescan', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(outputpath, 'nematic_linescan_profile.png'), dpi=150)
        plt.show()
    
    print(f"\n✓ Processing complete!")
    print(f"  - Results saved to: {csv_filename}")
    print(f"  - Number of points: {nfiles}")
    
    return x_array, z_array, orientation_array, S_array, R2_array, df


def detect_peaks_hybrid_interactive(proc,
                                    nb_peaks_init=6,
                                    qmin_init=0.03,
                                    qmax_init=0.15,
                                    fit_window_width_init=1,
                                    prominence=0.8,
                                    distance_pts=20,
                                    power_law_order=None,
                                    subtract_power_law_init=True,
									power_law_method = 'cancel'):
    """
    ÉTAPE 1 (Interactive) : Détection des pics avec méthode hybride et ajustement en temps réel.
    
    Permet d'ajuster nb_peaks, qmin, qmax et fit_window_width avec des sliders,
    et de relancer la détection en temps réel.
    
    Parameters:
    -----------
    proc : SAXSProcessor
        Instance du processeur SAXS
    nb_peaks_init : int
        Nombre initial de pics à détecter
    qmin_init, qmax_init : float
        Limites initiales en q (Å⁻¹)
    fit_window_width_init : float
        Largeur initiale de la fenêtre de fit SPV
    prominence : float
        Prominence minimale des pics (dérivée)
    distance_pts : int
        Distance minimale entre pics (points)
    power_law_order : float, optional
        Ordre de la loi de puissance pour le fond. Si None, sera estimé automatiquement
    subtract_power_law_init : bool
        Si True, soustrait la loi de puissance avant le fitting
        
    Returns:
    --------
    get_results : function
        Fonction pour récupérer les résultats après ajustement
    """
    from ipywidgets import IntSlider, FloatSlider, Checkbox, Output, VBox, Label, Dropdown
    from IPython.display import display
    
    output = Output()
    last_results = {'data': None}  # Variable pour stocker les résultats
    
    # Créer les sliders
    nb_peaks_slider = IntSlider(
        value=nb_peaks_init, min=1, max=15, step=1,
        description='nb_peaks:', style={'description_width': '120px'}
    )
    
    qmin_slider = FloatSlider(
        value=qmin_init, min=0.01, max=0.2, step=0.005,
        description='qmin (Å⁻¹):', style={'description_width': '120px'}
    )
    
    qmax_slider = FloatSlider(
        value=qmax_init, min=0.05, max=0.3, step=0.005,
        description='qmax (Å⁻¹):', style={'description_width': '120px'}
    )
    
    fit_window_slider = FloatSlider(
        value=fit_window_width_init, min=0.5, max=3.0, step=0.1,
        description='fit_window_width:', style={'description_width': '120px'}
    )
    
    subtract_power_law_checkbox = Checkbox(
        value=subtract_power_law_init,
        description='Subtract power law:',
        style={'description_width': '120px'}
    )
    
    power_law_method_dropdown = Dropdown(
        options=['feat', 'cancel'],
        value=power_law_method,
        description='Power law method:',
        style={'description_width': '120px'}
    )
    
    def update_plot(nb_peaks, qmin, qmax, fit_window_width, subtract_power_law, power_law_method_val):
        output.clear_output(wait=True)
        
        with output:
            print("=" * 70)
            print("ÉTAPE 1 : Détection des pics (Méthode HYBRIDE : Dérivée + SPV)")
            print("=" * 70)
            print(f"Paramètres : nb_peaks={nb_peaks}, q_range=[{qmin:.4f}, {qmax:.4f}], "
                  f"fit_window_width={fit_window_width:.1f}, subtract_power_law={subtract_power_law}, "
                  f"power_law_method={power_law_method_val}")
            
            corr = CorrelationDistanceCalculator(proc)
            distances, distances_std, hybrid_results = corr.compute_correlation_distances(
                nb_peaks=nb_peaks,
                azimuth=90,
                width=360,
                method='hybrid',
                qmin=qmin,
                qmax=qmax,
                prominence=prominence,
                distance_pts=distance_pts,
                power_law_order=power_law_order,
				power_law_method = power_law_method_val,
                power_law_range=[1,5],
                fit_window_width=fit_window_width,
                subtract_power_law=subtract_power_law,
                verbose=False,
                plot=True
            )
        
        # Stocker les résultats
        last_results['data'] = {
            'distances': distances,
            'distances_std': distances_std,
            'hybrid_results': hybrid_results,
            'peaklist': 2*np.pi/distances
        }
        
        return last_results['data']
    
    # Connecter les sliders à la fonction
    from ipywidgets import interactive
    interactive_plot = interactive(
        update_plot,
        nb_peaks=nb_peaks_slider,
        qmin=qmin_slider,
        qmax=qmax_slider,
        fit_window_width=fit_window_slider,
        subtract_power_law=subtract_power_law_checkbox,
        power_law_method_val=power_law_method_dropdown
    )
    
    # Afficher les contrôles et la sortie
    controls = VBox([
        Label('Ajuster les paramètres pour relancer la détection :'),
        nb_peaks_slider,
        qmin_slider,
        qmax_slider,
        fit_window_slider,
        subtract_power_law_checkbox,
        power_law_method_dropdown,
    ])
    
    # Fonction pour retourner les résultats
    def get_results():
        if last_results['data'] is None:
            print("❌ Aucun résultat disponible. Veuillez d'abord exécuter l'analyse.")
            return None
        return last_results['data']
    
    # Afficher
    display(controls)
    display(output)
    
    # Exécuter une première fois
    update_plot(nb_peaks_init, qmin_init, qmax_init, fit_window_width_init, subtract_power_law_init, power_law_method)
    
    # Créer fonction pour accéder aux résultats
    get_results.last_results = last_results
    
    return get_results


def save_azimuthal_profiles_to_csv(
        azimuthal_results,
        proc=None, 
        output_file=None, 
        tolerance=5.0,
        hybrid_results=None,
        verbose=True):
    """
    Exporte les résultats azimutaux en CSV avec arrondi des angles proches de 0, 45, 90°.
    
    Parameters:
    -----------
    azimuthal_results : dict
        Résultats de extract_and_plot_azimuthal_profiles() contenant :
        - 'q_values' : liste des positions des pics en q (Å⁻¹)
        - 'phi_maxima' : liste de listes, angles azimutaux des maxima pour chaque pic (°)
    proc : SAXSProcessor, optional
        Instance du processeur SAXS pour déterminer le chemin de sortie
    output_file : str, optional
        Chemin du fichier CSV. Si None, génère un nom auto avec timestamp
    tolerance : float, default=5.0
        Tolérance en degrés pour arrondir aux valeurs nominales (0°, 45°, 90°)
    hybrid_results : dict, optional
        Résultats de compute_correlation_distances contenant :
        - 'q_peaks' : positions des pics affinées
        - 'distances' : distances de corrélation
        - 'distances_std' : incertitudes sur les distances
        Si fourni, les distances seront incluses dans le CSV
    verbose : bool
        Afficher les informations de sauvegarde
        
    Returns:
    --------
    output_file : str
        Chemin du fichier sauvegardé
    """
    from datetime import datetime
    
    # Validation des entrées
    if not azimuthal_results or 'q_values' not in azimuthal_results or 'phi_maxima' not in azimuthal_results:
        raise ValueError("azimuthal_results doit contenir 'q_values' et 'phi_maxima'")
    
    # Générer un nom de fichier si nécessaire
    if output_file is None:
        if proc is not None:
            output_file = os.path.join(proc.path, f'{proc.samplename}_azimuthal_profiles.csv')
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"azimuthal_profiles_{timestamp}.csv"
    
    # Fonction pour arrondir les angles
    def round_angle_to_nominal(angle, tolerance=5.0):
        """Arrondit l'angle aux valeurs nominales (0°, 45°, 90°) si proche de ±tolerance"""
        nominal_angles = [-180,-135, -90, -45, 0, 45, 90, 135, 180]
        for nominal in nominal_angles:
            if abs(angle - nominal) <= tolerance:
                return nominal
        return round(angle, 2)  # Retourner l'angle réel arrondi à 2 décimales
    
    # Créer un mapping q_rounded -> (distance, distance_std) si hybrid_results fourni
    q_to_distance = {}
    if hybrid_results is not None:
        q_peaks = hybrid_results.get('q_peaks', [])
        distances = hybrid_results.get('distances', [])
        distances_std = hybrid_results.get('distances_std', [])
        
        for i, q_peak in enumerate(q_peaks):
            q_key = round(q_peak, 6)
            distance = distances[i] if i < len(distances) else None
            distance_std = distances_std[i] if i < len(distances_std) else None
            q_to_distance[q_key] = (distance, distance_std)
    
    # Construire le DataFrame
    rows = []
    for q_value, angles_list in zip(azimuthal_results['q_values'], azimuthal_results['phi_maxima']):
        q_rounded = round(q_value, 6)
        
        # Chercher les distances correspondantes
        distance = None
        distance_std = None
        if q_rounded in q_to_distance:
            distance, distance_std = q_to_distance[q_rounded]
        
        # Créer une ligne par angle détecté
        for angle in angles_list:
            row = {
                'q_peak_Angstrom-1': q_value,
                'azimuthal_angle_degrees': round(angle, 2),
                'angle_rounded_degrees': round_angle_to_nominal(angle, tolerance)
            }
            
            # Ajouter les distances si disponibles
            if distance is not None:
                row['distance_Angstrom'] = distance
            if distance_std is not None:
                row['distance_std_Angstrom'] = distance_std
            
            rows.append(row)
    
    # Créer le DataFrame
    df = pd.DataFrame(rows)
    
    # Trier par q puis par angle
    df = df.sort_values(['q_peak_Angstrom-1', 'azimuthal_angle_degrees']).reset_index(drop=True)
    
    # Sauvegarder
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    df.to_csv(output_file, index=False)
    
    if verbose:
        print("=" * 70)
        print("✓ RÉSULTATS AZIMUTAUX SAUVEGARDÉS")
        print("=" * 70)
        print(f"Fichier : {output_file}")
        print(f"Nombre de lignes : {len(df)}")
        print(f"Colonnes : {list(df.columns)}")
        print(f"\nAperçu :")
        print(df.to_string(index=False))
        print("=" * 70)
        print(f"\nAperçu :")
        print(df.to_string(index=False))
    
    return output_file


def extract_and_plot_azimuthal_profiles_interactive(proc, 
                                                    peaklist,
                                                    threshold=0.005,
                                                    apply_mirror=True,
                                                    output_dir=None):
    """
    Version interactive de extract_and_plot_azimuthal_profiles avec widgets.
    
    Permet d'ajuster smooth_sigma, peak_prominence et peak_distance
    et de relancer la détection en temps réel.
    
    Parameters:
    -----------
    proc : SAXSProcessor
        Instance du processeur SAXS
    peaklist : array
        Positions des pics (en Å⁻¹)
    threshold : float
        Tolérance relative en q pour l'extraction
    apply_mirror : bool
        Appliquer la symétrie miroir (180°)
    output_dir : str, optional
        Répertoire de sauvegarde des figures
        
    Returns:
    --------
    dict
        Dictionnaire contenant 'q_values' et 'phi_maxima' après la première exécution
    """
    from ipywidgets import FloatSlider, IntSlider, Output, VBox, Label
    from IPython.display import display
    
    output = Output()
    last_results = {'data': None}  # Variable pour stocker les résultats
    
    # Créer les sliders
    smooth_slider = FloatSlider(
        value=1.5, min=0.1, max=5.0, step=0.1,
        description='Smooth σ:', style={'description_width': '100px'}
    )
    
    prominence_slider = FloatSlider(
        value=0.1, min=0.01, max=0.3, step=0.01,
        description='Prominence:', style={'description_width': '100px'}
    )
    
    distance_slider = IntSlider(
        value=20, min=1, max=50, step=1,
        description='Distance (pts):', style={'description_width': '100px'}
    )
    
    def update_plot(smooth_sigma, peak_prominence, peak_distance):
        output.clear_output(wait=True)
        
        with output:
            results = extract_and_plot_azimuthal_profiles(
                proc,
                peaklist,
                threshold=threshold,
                apply_mirror=apply_mirror,
                output_dir=output_dir,
                smooth_sigma=smooth_sigma,
                peak_prominence=peak_prominence,
                peak_distance=peak_distance if peak_distance > 0 else None
            )
        
        # Stocker les résultats
        last_results['data'] = results
        return results
    
    # Connecter les sliders à la fonction
    from ipywidgets import interactive
    interactive_plot = interactive(
        update_plot,
        smooth_sigma=smooth_slider,
        peak_prominence=prominence_slider,
        peak_distance=distance_slider
    )
    
    # Afficher les contrôles et la sortie
    controls = VBox([
        Label('Ajuster les paramètres de détection :'),
        smooth_slider,
        prominence_slider,
        distance_slider
    ])
    
    display(controls)
    display(output)
    
    # Exécuter une première fois avec les valeurs par défaut
    print("Extraction des profils azimutaux en cours...")
    update_plot(smooth_slider.value, prominence_slider.value, distance_slider.value)
    
    # S'assurer que les résultats sont disponibles
    if last_results['data'] is None:
        print("⚠️ Aucun résultat disponible après l'exécution initiale")
        return {
            'q_values': [],
            'phi_maxima': []
        }
    
    # Retourner les résultats de la première exécution
    return last_results['data']


def extract_and_plot_azimuthal_profiles(proc,
                                        peaklist,
                                        threshold=0.02,
                                        apply_mirror=True,
                                        output_dir=None,
                                        smooth_sigma=1.5,
                                        peak_prominence=0.1,
                                        peak_distance=None):
    """
    ÉTAPE 2 (Non-interactive) : Extraction et tracé des profils azimutaux.
    
    Parameters:
    -----------
    proc : SAXSProcessor
        Instance du processeur SAXS
    peaklist : array
        Positions des pics (en Å⁻¹)
    threshold : float
        Tolérance relative en q pour l'extraction
    apply_mirror : bool
        Appliquer la symétrie miroir (180°)
    output_dir : str, optional
        Répertoire de sauvegarde des figures
    smooth_sigma : float
        Sigma pour le lissage Gaussien (default: 1.5)
    peak_prominence : float
        Prominence minimale pour la détection des maxima (default: 0.1)
    peak_distance : int, optional
        Distance minimale entre pics (points) (default: None)
        
    Returns:
    --------
    dict
        Dictionnaire contenant :
        - 'q_values' : liste des positions des pics en q (Å⁻¹)
        - 'phi_maxima' : liste de listes, angles azimutaux des maxima pour chaque pic (°)
    """
    from scipy.signal import find_peaks
    from scipy.ndimage import gaussian_filter1d
    import matplotlib.pyplot as plt
    
    q_values = []
    phi_maxima = []
    
    for q_peak in peaklist:
        # Extraire le profil azimutal pour ce q avec apply_mirror
        azimuthal_angles, intensity_profile = proc.extract_azimuthal_profile(
            qvalue=q_peak,
            threshold=threshold,
            save=False,
            apply_mirror=apply_mirror
        )
        
        # Lisser le profil
        if smooth_sigma > 0:
            smoothed_profile = gaussian_filter1d(intensity_profile, sigma=smooth_sigma)
        else:
            smoothed_profile = intensity_profile
        
        # Normaliser le profil
        normalized_profile = (smoothed_profile - np.min(smoothed_profile)) / (np.max(smoothed_profile) - np.min(smoothed_profile) + 1e-10)
        
        # Détecter les maxima
        peaks, _ = find_peaks(
            normalized_profile,
            prominence=peak_prominence,
            distance=peak_distance
        )
        
        # Extraire les angles des maxima et les convertir en liste
        angles_maxima = azimuthal_angles[peaks].tolist() if len(peaks) > 0 else []
        
        # Stocker les résultats
        q_values.append(q_peak)
        phi_maxima.append(angles_maxima)
        
        # Tracer
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Profil azimutal original et lissé
        ax1.plot(azimuthal_angles, intensity_profile, 'b-', alpha=0.5, label='Original')
        ax1.plot(azimuthal_angles, smoothed_profile, 'r-', linewidth=2, label='Lissé')
        ax1.set_xlabel('Azimuthal angle (°)')
        ax1.set_ylabel('Intensity (a.u.)')
        ax1.set_title(f'Profil azimutal à q = {q_peak:.4f} Å⁻¹')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Profil normalisé avec détection des pics
        ax2.plot(azimuthal_angles, normalized_profile, 'g-', linewidth=2, label='Normalisé')
        if len(peaks) > 0:
            ax2.plot(azimuthal_angles[peaks], normalized_profile[peaks], 'ro', markersize=8, label=f'Pics détectés ({len(peaks)})')
        ax2.set_xlabel('Azimuthal angle (°)')
        ax2.set_ylabel('Intensity (normalized)')
        ax2.set_title(f'Détection des maxima (q = {q_peak:.4f} Å⁻¹)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f'azimuthal_q_{q_peak:.4f}.png')
            plt.savefig(filename, dpi=100, bbox_inches='tight')
        
        plt.show()
    
    return {
        'q_values': q_values,
        'phi_maxima': phi_maxima
    }


def plot_2d_with_peaks(proc, peaklist, q_range=None):
    """
    ÉTAPE 3 : Affichage 2D avec les pics marqués en cercles.
    
    Parameters:
    -----------
    proc : SAXSProcessor
        Instance du processeur SAXS
    peaklist : array
        Positions des pics (en Å⁻¹)
    q_range : tuple, optional
        Plage de q pour l'affichage [qmin, qmax]
        
    Returns:
    --------
    None (affiche la figure 2D)
    """
    from IPython.display import clear_output
    
    print("=" * 70)
    print("ÉTAPE 3 : Affichage 2D avec pics")
    print("=" * 70)
    
    proc.plot2d_vsq(q_circles=peaklist, q_range=q_range)
    print(f"✓ Figure 2D affichée avec {len(peaklist)} cercle(s)\n")


def extract_relevant_azimuthal_profiles(proc, 
                                        threshold = 0.005,                                       
                                        plot=True,
                                        output_dir=None,
                                        verbose=True,
                                        apply_mirror=True,
                                        nb_peaks=1,
                                        qmin=0,
                                        qmax=0.2,
                                        manual_input=False):
    """
    Extract and plot relevant azimuthal profiles (wrapper version).
    
    ⚠️ PRÉFÉRER L'UTILISATION EN 3 ÉTAPES :
    
    # Étape 1 : Détection des pics
    peaklist = detect_peaks_interactive(proc, nb_peaks=6, qmin=0.03, qmax=0.15, manual_input=True)
    
    # Étape 2 : Extraction des profils azimutaux
    extract_and_plot_azimuthal_profiles(proc, peaklist, threshold=0.01, apply_mirror=True)
    
    # Étape 3 : Affichage 2D
    plot_2d_with_peaks(proc, peaklist)

    Parameters:
    ----------
    proc : SAXSProcessor
        SAXSProcessor instance with loaded and averaged data
    threshold : float
        Width of tolerance for azimuthal profile extraction (%q)
    plot : bool
        Whether to plot the azimuthal profiles (default: True)
    output_dir : str, optional
        Directory to save plots 
    verbose : bool
        Print progress information (default: True)
    apply_mirror : bool 
        Apply mirror symmetry to complete incomplete azimuthal profiles (default: True)
    manual_input : bool
        If True, allows manual input of q values instead of automatic peak detection (default: False)
    """
    # Exécuter les 3 étapes
    peaklist = detect_peaks_interactive(proc, nb_peaks, qmin, qmax, manual_input)
    extract_and_plot_azimuthal_profiles(proc, peaklist, threshold, apply_mirror, output_dir)
    plot_2d_with_peaks(proc, peaklist)



def compute_average_correlation_distances(
        # SAXSProcessor instance
        proc,
        # CorrelationDistanceCalculator arguments
        nb_peaks=2,
        azimuth=90,
        width=360,
        method='hybrid',
        qmin=0.01,
        qmax=0.125,
        # Détection initiale (dérivée seconde)
        window_length=15,
        polyorder=3,
        prominence=0.5,
        distance_pts=20,
        # Raffinement SPV
        subtract_power_law=True,
        power_law_method='cancel',
        power_law_order=None,
        power_law_range=(2.5, 5.0),
        smooth=False,
        smooth_sigma=2,
        fit_window_width=1, 
        verbose=False,
        plot=True):
    """ 
    Compute average correlation distance from multiple SWING h5 files.
    
    Parameters
    ----------
    --------------- File definitions --------------
    proc : SAXSProcessor
        SAXSProcessor instance with average intensity data
    --------------- CorrelationDistanceCalculator parameters --------------
    nb_peaks : int
        Number of peaks to consider for correlation distance calculation (default: 6)
    azimuth : float
        Azimuthal angle for profile extraction (default: 90°)
    width : float
        Width for azimuthal profile extraction (default: 360°)
    method : str
        Method for correlation distance calculation ('hybrid', 'spv', 'derivative') (default: 'hybrid')
    qmin : float
        Minimum q value for analysis (default: 0.03 Å⁻¹)
    qmax : float
        Maximum q value for analysis (default: 0.125 Å⁻¹)
    --------------- Peak detection parameters (derivative method) --------------
    window_length : int
        Window length for Savitzky-Golay filter (default: 15)
    polyorder : int
        Polynomial order for Savitzky-Golay filter (default: 3)
    prominence : float
        Prominence for peak detection (default: 0.5)
    distance_pts : int
        Minimum distance between peaks in data points (default: 20)
    --------------- SPV refinement parameters --------------
    subtract_power_law : bool
        Whether to subtract power-law background (default: True)
    power_law_method : str
        Method for power-law subtraction ('cancel', 'feat') (default: 'cancel')
    power_law_order : int, optional
        Order of the power-law fit (default: None)
    power_law_range : tuple of float
        Range of q values for power-law fitting (default: (2.5, 5.0))
    smooth : bool
        Whether to apply smoothing (default: False)
    smooth_sigma : float
        Sigma for Gaussian smoothing (default: 2)
    fit_window_width : float
        Width of the fitting window around each peak (default: 3)
    verbose : bool
        Whether to print detailed information (default: False)
    plot : bool
        Whether to generate plots (default: True)
    """

    # Create CorrelationDistanceCalculator
    corr = CorrelationDistanceCalculator(proc)
     # Compute average correlation distance
    avg_distance, err, results = corr.compute_correlation_distances(
        nb_peaks=nb_peaks,
        azimuth=azimuth,
        width=width,
        method=method,
        qmin=qmin,
        qmax=qmax,
        # Peak detection parameters
        window_length=window_length,
        polyorder=polyorder,
        prominence=prominence,
        distance_pts=distance_pts,
        # SPV refinement parameters
        subtract_power_law=subtract_power_law,
        power_law_method=power_law_method,
        power_law_order=power_law_order,
        power_law_range=power_law_range,
        smooth=smooth,
        smooth_sigma=smooth_sigma,
        fit_window_width=fit_window_width,
        verbose=verbose,
        plot=plot
    )
    return avg_distance, err, results

def global_analysis(
        csv_file,
        h5path,
        prefix='lacroix',
        #SAXSProcessor arguments
        reference_file=None,
        k=1,
        autosubstract=True,
        mask=None,
        #Nematic order parameters arguments
        threshold=0.05,
        radius=78,
        L=840,
        radius_pd=0.3,
        L_pd=0.75,        
        # CorrelationDistanceCalculator arguments
        nb_peaks=2,
        azimuth=90,
        width=360,
        method='hybrid', 
        qmin=0.01,
        qmax=0.15,       
        # Détection initiale (dérivée seconde)
        window_length=15,
        polyorder=3,
        prominence=0.5,
        distance_pts=20,
        # Raffinement SPV
        subtract_power_law=True,
        power_law_method='cancel',
        power_law_order=None,
        power_law_range=(2.5, 5.0),
        smooth=False,
        smooth_sigma=2,
        fit_window_width=1, 
        verbose=False,
        plot=True):
    """
    Perform global analysis on SWING h5 files including nematic order parameter determination 
    and average correlation distance computation.

    Parameters:
    ----------
    csv_file : str
        Path to csv_file where to append data.
    --------------- File definitions --------------
    h5path : str
        Path to folder containing h5 files
    prefix : str
        h5 file prefix (default: 'lacroix')
    --------------- SAXSProcessor parameters --------------
    reference_file : str, optional
        Path to h5 file for reference measurement (default: None)
    k : float
        Coefficient for reference subtraction (default: 1)
    autosubstract : bool
        Use optimized reference subtraction (default: True)
    mask : str, optional
        Path to mask file
    --------------- Nematic order parameters arguments --------------
    threshold : float
        Width of tolerance for azimuthal profile extraction (%q)
    radius : float
        CylinderFormFactor parameter - cylinder radius
    L : float
        CylinderFormFactor parameter - cylinder length
    radius_pd : float
        CylinderFormFactor parameter - cylinder radius polydispersity ratio
    L_pd : float
        CylinderFormFactor parameter - cylinder length polydispersity ratio
    R2_threshold : float
        R² threshold for nematic order parameter map visualization (default: 0.9)
    --------------- CorrelationDistanceCalculator parameters --------------
    nb_peaks : int
        Number of peaks to consider for correlation distance calculation (default: 2)
    azimuth : float 
        Azimuthal angle for profile extraction (default: 90°)
    width : float   
        Width for azimuthal profile extraction (default: 360°)
    method : str
        Method for correlation distance calculation ('hybrid', 'spv', 'derivative') (default: 'hybrid')
    
    --------------- Peak detection parameters (derivative method) --------------
    window_length : int
        Window length for Savitzky-Golay filter (default: 15)
    polyorder : int
        Polynomial order for Savitzky-Golay filter (default: 3)
    prominence : float 
        Prominence for peak detection (default: 0.5)
    distance_pts : int
        Minimum distance between peaks in data points (default: 20)
    --------------- SPV refinement parameters --------------
    subtract_power_law : bool
        Whether to subtract power-law background (default: True)
    power_law_method : str
        Method for power-law subtraction ('cancel', 'feat') (default: 'cancel')
    power_law_order : int, optional
        Order of the power-law fit (default: None)
    power_law_range : tuple of float
        Range of q values for power-law fitting (default: (2.5, 5.0))
    smooth : bool
        Whether to apply smoothing (default: False)
    smooth_sigma : float
        Sigma for Gaussian smoothing (default: 2)
    fit_window_width : float
        Width of the fitting window around each peak (default: 3)
    verbose : bool
        Whether to print detailed information (default: False)
    plot : bool
        Whether to generate plots (default: True)   

    
    Returns:
    --------
    dict
        Dictionary containing sample info, nematic order parameter map and average correlation distance results.
    """

    # Create SAXSProcessor with average intensity
    proc = average_h5_processor(
        h5path=h5path,
        prefix=prefix,
        reference_file=reference_file,
        k=k,
        autosubstract=autosubstract,
        mask=mask,
        verbose=verbose
    )
    
    # plot 2D intensity map
    proc.plot2d_vsq()
    
    # Create CorrelationDistanceCalculator
    corr = CorrelationDistanceCalculator(proc)
    # Compute average correlation distance
    distances, err, results_corr = corr.compute_correlation_distances(
        nb_peaks=nb_peaks,
        azimuth=azimuth,
        width=width,
        method=method,
        qmin=qmin,
        qmax=qmax,
        # Peak detection parameters
        window_length=window_length,
        polyorder=polyorder,
        prominence=prominence,
        distance_pts=distance_pts,
        # SPV refinement parameters
        subtract_power_law=subtract_power_law,
        power_law_method=power_law_method,
        power_law_order=power_law_order,
        power_law_range=power_law_range,
        smooth=smooth,
        smooth_sigma=smooth_sigma,
        fit_window_width=fit_window_width,
        verbose=verbose,
        plot=plot
    )
    qvalues = results_corr['q_peaks']
    
    # Compute global nematic order parameter map
    S_array = np.zeros(len(qvalues))
    mean_orientation_array = np.zeros(len(qvalues))
    
    for i,qvalue in enumerate(qvalues):
        S, mean_orientation, res = compute_nematic_parameter(
            proc,
            threshold=threshold,
            radius=radius,
            L=L,
            radius_pd=radius_pd,
            L_pd=L_pd,
            plot=False,
            apply_mirror=None,
            verbose=True)
        S_array[i]=S
        mean_orientation_array[i]=mean_orientation
    
    # build results dictionary
    results={
    'samplename': proc.samplename,
    'B (mT)': proc.B,
    'q_peak (A⁻¹)': qvalues, # array of q values
    'distance (A)': distances, # array of distances
    'distance_error (A)': err, # array of errors
    'mean_orientation (°)': mean_orientation_array, # array of mean orientations
    'nematic_parameter': S_array # array of nematic order parameters
    }       

    # add dictionary to existing csv or create new one
    
    if os.path.exists(csv_file):
        df_existing = pd.read_csv(csv_file)
        df_new = pd.DataFrame([results])
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(csv_file, index=False)
        print(f'The following line {df_new} has been added to file {csv_file}')
        
    else:
        df = pd.DataFrame([results])
        df.to_csv(csv_file, index=False)
        print(f'The folloing line {df} has been added to new file {csv_file}')
    print(f"✓ Global analysis results saved to: {csv_file}")
    return results


