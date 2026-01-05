from filereaders import h5File_SWING
from saxsprocessor import SAXSProcessor
import glob
import os
import re
from matplotlib import pyplot as plt
import numpy as np
from nematicordercalculator import CylinderFormFactor,NematicOrderCalculator
import pandas as pd
from tqdm.notebook import tqdm
from IPython.display import clear_output, display, Markdown
from scipy.interpolate import griddata
    

def compute_nematic_parameter(
        processor = None,
        qvalue = 0.034,
        threshold = 0.05,
        radius = 78,
        L = 840,
        radius_pd = 0.3,
        L_pd = 0.75,
        plot=False,
        apply_mirror=False,
        verbose=True
        ):
    
    """
    processor: SAXSProceesor instance
        SAXSProcessor instance (mandatory)
    qvalue: float
        q value of interest at which azimuthal profile is extracted
    threshold: float
        relative value to defin the q range for azimuthal profile extraction
    radius: float
        Cylinder radius for form factor caclulation
    L: float
        Cylinder length for form factor calculation
    radius_pd, L_pd: float
        Polydispersity ratio for radius and length
    plot : bool
        Plot form factor and fitting results
    apply_mirror: bool
        Apply mirror symmetry to incomplete exp azimuthal profile

    *** Returns ****
    results : dict
        {'I0': I0_opt,
        'm': m_opt,
        'x0': x0_opt,
        'a': a_opt,
        'b': b_opt,
        'S': S_opt,
        'R2': r_squared,
        'I_model': I_model}
    """
    # 1. Extract azimuthal profile
    chi_exp, I_az_exp = processor.extract_azimuthal_profile(qvalue=0.034, threshold = 0.05)
    # 2. determine main orientation
    chi = processor.find_main_orientation(qvalue=qvalue, threshold= threshold)
    if verbose:
        print('Main orientation is ', chi)
    # 3. Cylinder form factor calculation
    if verbose:
        print(f"\n{'=' * 60}")
        print('Computing cylinder form factor...')
    form_factor = CylinderFormFactor(
        processor=processor, 
        radius=radius,
        L=L,
        theta=90,
        phi=chi-90,
        radius_pd=radius_pd,
        L_pd=L_pd,
        phi_pd=0,
        theta_pd=0,
        background=0.0001,
        scale=1,
        plot = plot)
    # 4. Nematic order calculation
    if verbose:
        print(f"\n{'=' * 60}")
        print('Fitting azimuthal profile to extract nematic order parameter...')
    nematic_calc = NematicOrderCalculator(form_factor=form_factor)
    results = nematic_calc.fit_azimuthal_profile(
        chi_exp,
        I_az_exp,
        qvalue_ff=qvalue,
        threshold_ff=threshold,
        plot=plot,
        target=chi,
        apply_mirror=apply_mirror,
        processor = processor,
        verbose=verbose
        )
    return results["S"], results


#####################################################################################################################

        # FUNCTIONS DEDICATED TO THE WORKFLOW FOR MICRO-SAXS DATA PROCESSING

# a single file corresponds to a linescan along x (multiple frames along x for a given z position)

#####################################################################################################################


def sort_h5(file):
    match = re.search(r'lacroix_(\d+)', file)
    if not match:
        raise ValueError("Format de fichier non reconnu")

    return int(match.group(1))

def sort_edf(file):
    match = re.search(r'File_(\d+)_Img_(\d+)', file)
    if not match:
        raise ValueError("Format de fichier non reconnu")

    return int(match.group(1)), int(match.group(2))


def compute_nematic_order_assembly_SWING(
        h5path,
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
    SAXS linescan processing with nematic order parameter computation
    and live progress display in a Jupyter notebook.
    """

    def log_step(step, total, message):
        clear_output(wait=True)
        display(Markdown(f"### Step {step}/{total}\n**{message}**"))

    print("⚠️ WARNING: This function assumes the data correspond to a linescan.")

    # === STEP 1: h5 file list
    log_step(1, 7, "Searching and sorting h5 files")
    h5_filelist = sorted(
        glob.glob(os.path.join(h5path, '*.h5')),
        key=sort_h5
    )

    # === STEP 2: conversion to EDF
    log_step(2, 7, "Extracting frames and converting to EDF files")
    edfpath = os.path.join(h5path, 'edf_files')
    os.makedirs(edfpath, exist_ok=True)

    for h5file in tqdm(h5_filelist, desc="Converting h5 → EDF"):
        SWING_file = h5File_SWING(h5file, mean=False)
        SWING_file.convert2edf(outputdir=edfpath)

    # === STEP 3: scan geometry
    log_step(3, 7, "Determining scan geometry")
    number_of_lines = len(h5_filelist)
    number_of_columns = SWING_file.nb_frames

    # === STEP 4: EDF file list
    log_step(4, 7, "Building EDF file list")
    edf_filelist = sorted(
        glob.glob(os.path.join(edfpath, '*Img*.edf')),
        key=sort_edf
    )

    # === STEP 5: nematic order computation
    log_step(5, 7, "Computing nematic order parameter")

    nfiles = len(edf_filelist)
    x_array = np.zeros(nfiles)
    z_array = np.zeros(nfiles)
    orientation_array = np.zeros(nfiles)
    S_array = np.zeros(nfiles)
    R2_array = np.zeros(nfiles)
    data_list = []

    for i, file in enumerate(tqdm(edf_filelist, desc="Processing SAXS images", unit="image")):
        print('\n' + '*'*120)
        print(f'\n Processing file {i+1} / {len(edf_filelist)}: {file}')
        print('\n' + '*'*120)
        proc = SAXSProcessor(file=file, mask=mask, instrument='LGC')

        chi = proc.find_main_orientation(
            qvalue=qvalue,
            threshold=threshold
        )

        S, results = compute_nematic_parameter(
            processor=proc,
            qvalue=qvalue,
            threshold=threshold,
            plot=plot,
            apply_mirror=apply_mirror,
            verbose=verbose
        )

        x_array[i] = proc.x
        z_array[i] = proc.z
        orientation_array[i] = chi
        S_array[i] = S
        R2_array[i] = results['R2']

        row_data = {
            'File number': proc.file_number,
            'samplename': proc.samplename,
            'B (mT)': proc.B,
            'x (mm)': proc.x,
            'z (mm)': proc.z,
            'orientation (°)': chi - 90,
            'R2': results['R2']
        }

        for key, value in results.items():
            if key != 'I_model':
                row_data[key] = value

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

    orientation_2d = orientation_array.reshape(number_of_lines, number_of_columns) - 90
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
    
    S_interp = griddata(points, values_S, (x_grid, z_grid), method='cubic')

    
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

    ax.quiver(
        x_2d, z_2d, u, v,
        color='black',
        scale=15,
        width=0.005,
        alpha=0.8
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('S', rotation=270, labelpad=20)

    ax.set_xlabel('X position (mm)',fontsize=14)
    ax.set_ylabel('Z position (mm)',fontsize=14)
    ax.set_title(f'Nematic order parameter map',fontsize=16)

    plt.tight_layout()
    figname = os.path.join(outputpath, 'nematic_orientation_map.png')
    plt.savefig(figname)
    plt.show()

    return x_2d, z_2d, orientation_2d, S_2d, R2_2d


def plot_nematic_order_map(x_2d, z_2d, orientation_2d, S_2d, R2_2d, R2_threshold=0.9, outputpath=None):
    """
    Generate nematic order parameter map with quiver plot for orientation.
    """
    from scipy.interpolate import griddata
    
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