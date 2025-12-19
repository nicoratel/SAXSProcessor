from filereaders import h5File_SWING
from saxsprocessor import SAXSProcessor
import glob
import os
import re
from matplotlib import pyplot as plt
import numpy as np
from nematicordercalculator import CylinderFormFactor,NematicOrderCalculator
import pandas as pd

def compute_nematic_parameter(
        processor = None,
        qvalue = 0.034,
        threshold = 0.05,
        radius = 78,
        L = 840,
        radius_pd = 0.3,
        L_pd = 0.75,
        plot=False,
        apply_mirror=False
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
    print('Main orientation is ', chi)
    # 3. Cylinder form factor calculationa
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
    nematic_calc = NematicOrderCalculator(form_factor=form_factor)
    results = nematic_calc.fit_azimuthal_profile(
        chi_exp,
        I_az_exp,
        qvalue_ff=qvalue,
        threshold_ff=threshold,
        plot=plot,
        target=chi,
        apply_mirror=apply_mirror)
    return results["S"], results

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


def plot_nematicorder_assembly_SWING(
        h5path,
        mask=None,
        qvalue = 0.034,
        threshold = 0.05,
        radius = 78,
        L = 840,
        radius_pd = 0.3,
        L_pd = 0.75,
        plot=False,
        apply_mirror=False):
    """
    h5path: string
        Path to directory where data is stored
    mask:
        Path to mask file
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
    """
    print('======= WARNING: This function assumes that the file correspond to a linescan. =======')
    # 1. Create list
    h5_filelist=glob.glob(os.path.join(h5path,'*.h5'))
    h5_filelist=sorted (h5_filelist, key = sort_h5) # liste triée par N° d'image
    print('======= h5 file list =======')
    for file in h5_filelist:
        print(file)

    # 2. Extract individual frames and convert files to edf format
    edfpath = os.path.join(h5path,'edf_files/')
    print('======= Extraction and conversion to edf files =======')
    for h5file in h5_filelist:
        SWING_file = h5File_SWING(h5file, mean = False)
        SWING_file.convert2edf(outputdir=edfpath)
        

    # 3. Define number of scans in the map: lines * columns
    number_of_lines = len(h5_filelist)
    number_of_columns = SWING_file.nb_frames

    # 4. create edf file list
    edf_filelist=glob.glob(os.path.join(edfpath, '*Img*.edf'))
    edf_filelist = sorted (edf_filelist, key=sort_edf) # sorted list by line (h5 file) and position

    # 5. Compute nematic order parameter for each file
    x_array=np.zeros(number_of_columns*number_of_lines)
    z_array= np.zeros(number_of_columns*number_of_lines)
    orientation_array=np.zeros(number_of_columns*number_of_lines)
    S_array = np.zeros_like(x_array)
    data_list = []
    for i,file in enumerate(edf_filelist):
        print(f'======= Processing file {i}/{len(edf_filelist)} =======')
        proc = SAXSProcessor(file=file,mask=mask,instrument='LGC')
        chi = proc.find_main_orientation(qvalue=qvalue,threshold=threshold)
        S, results = compute_nematic_parameter(proc,qvalue=qvalue,threshold=threshold,plot=plot,apply_mirror=apply_mirror)
        S_array[i]=S
        x_array[i]=proc.x
        z_array[i]=proc.z
        orientation_array[i]=chi
        

    # 6 Export csv
    # Add data for csv export
        row_data = {
            'File number': proc.file_number,
            'samplename': proc.samplename,
            'B (mT)': proc.B,
            'x (mm)': proc.x,
            'z (mm)': proc.z,
            'orientation (°)': chi-90            
        }
        for key, value in results.items():
            if key != 'I_model':  # Exclure les arrays
                row_data[key] = value
        
        data_list.append(row_data)
    df = pd.DataFrame(data_list)
    outputpath = os.path.join(h5path,'nematic_processing_results')
    os.makedirs(outputpath,exist_ok=True)
    csv_filename = os.path.join(outputpath, 'nematic_order_results.csv')
    df.to_csv(csv_filename, index=False)
    print(f'======= Results exported to {csv_filename} =======')
    


    # 7. Plot results

    # Reshape des tableaux en 2D
    orientation_2d = orientation_array.reshape(number_of_lines, number_of_columns)-90
    S_2d = S_array.reshape(number_of_lines, number_of_columns)
    x_2d = x_array.reshape(number_of_lines, number_of_columns)
    z_2d = z_array.reshape(number_of_lines, number_of_columns)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Carte de couleur pour S
    im = ax.imshow(S_2d, 
                extent=[130, 500,z_2d.min(), z_2d.max()],
                origin='lower', 
                aspect='auto', 
                cmap='jet',
                interpolation='bicubic')

    # Calcul des composantes des flèches à partir de l'orientation
    # orientation en degrés -> conversion en radians
    u = np.cos(np.radians(orientation_2d))
    v = np.sin(np.radians(orientation_2d))

    # Ajout des flèches d'orientation
    quiver = ax.quiver(x_2d, z_2d, u, v, 
                    color='white',           # couleur des flèches
                    scale=20,                # ajuster pour la taille des flèches
                    width=0.003,             # épaisseur des flèches
                    headwidth=3,             # taille de la tête
                    headlength=4,            # longueur de la tête
                    alpha=0.8)               # transparence

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('S', rotation=270, labelpad=20)
    ax.set_xlabel('X position')
    ax.set_ylabel('Z position')
    ax.set_title('Cartographie de S avec orientation')
    plt.tight_layout()
    plt.show()

    figname = os.path.join(outputpath,'nematic_orientation_map.png')
    plt.savefig(figname)
    print(f'======= Plot saved in {figname} =======')

    
    return x_2d, z_2d, orientation_2d, S_2d