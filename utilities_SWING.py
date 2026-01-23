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


#####################################################################################################################

        # FUNCTIONS DEDICATED TO THE WORKFLOW FOR MICRO-SAXS DATA PROCESSING

# a single file corresponds to a linescan along x (multiple frames along x for a given z position)

#####################################################################################################################



def view_position_grid(
    data_folder: str,
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
    verbose: bool = True
):
    """
    Visualize all acquisition positions on a Basler reference image.
    
    Parameters
    ----------
    data_folder : str
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
    #file_pattern = f'{prefix}_*.h5'
    #all_files = sorted(glob.glob(os.path.join(data_folder, file_pattern)))
    file_pattern = f'{prefix}_*.h5'
    all_files = sorted(
        glob.glob(os.path.join(data_folder, file_pattern)),
        key=lambda f: sort_h5(f, prefix=prefix)
    )
    
    if len(all_files) == 0:
        raise FileNotFoundError(f"No files found matching pattern '{file_pattern}' in {data_folder}")
    
    if verbose:
        print(f"{'='*60}")
        print(f"VIEW POSITION GRID")
        print(f"{'='*60}")
        print(f"Data folder: {data_folder}")
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
                    data_folder, 
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
    pattern = rf'{re.escape(prefix)}_(\d+)'
    match = re.search(pattern, file)
    if not match:
        raise ValueError("Format de fichier non reconnu")
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
    apply_mirror=False,
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

def compute_global_nematic_parameter(
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
        apply_mirror=False,
        verbose=True):
    """
    Compute_global_nematic_parameter for average intensity measured in an assembly
    
    :param h5path: str path to folder containing h5 files
    :param prefix: str prefix for h5files (default = 'lacroix')
    :param reference_file: str path to h5 file corresponding to reference measurement (default= None)
    :param k: float Coefficient for reference substraction (default = 1)
    :param autosubstract: bool Use optimized reference substraction (default = True)
    :param mask: str path to mask file
    :param qvalue: float q value used for azimuthal profile extraction (default = 0.034)
    :param threshold: float Width of tolerance for azimuthal profile extraction (%q)
    :param radius: float CylinderFormFactor parameter - cylinder radius
    :param L: float CylinderFormFactor parameter - cylinder length
    :param radius_pd: float CylinderFormFactor parameter - cylinder radius polydispersity ratio
    :param L_pd: float CylinderFormFactor parameter - cylinder length polydispersity ratio
    :param plot: bool Plot results
    :param apply_mirror: bool apply mirror symetry to complete incomplete azimuthal profiles (default = False)
    :param verbose: bool print outputs
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
    # Assign mean intensity to SAXSProcessor instance
    proc.data = mean_intensity
    S, mean_angle, dict = compute_nematic_parameter(
        processor = proc,
        qvalue = qvalue,
        threshold = threshold,
        radius = radius,
        L = L,
        radius_pd = radius_pd,
        L_pd= L_pd,
        plot = plot,
        apply_mirror = apply_mirror,
        verbose = verbose
    )
    print(f'The global nematic order parameter is S={S:.4f}')
    return S

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
    apply_mirror=False,
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
