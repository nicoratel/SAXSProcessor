from .filereaders import h5File_ID02, h5File_SWING, EdfFile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from ase.io import read
import fabio
import os
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d, griddata

import pandas as pd

# PyFAI imports
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
import pyFAI.detectors


class CrystalStructure:
    """
    Class to handle crystal structure information from CIF files.
    """
    def __init__(self, cif_file: str, reflections: list, threshold: float = 0.001,verbose=True):
        """
        Initialize CrystalStructure with CIF file and reflections.
        
        Parameters:
        -----------
        cif_file : str
            Path to CIF file
        reflections : list
            List of Miller indices (h, k, l)
        threshold : float
            Tolerance for Q value matching
        """
        self.cif_file = cif_file
        self.reflections = reflections
        self.threshold = threshold
        self.verbose = verbose
        # Parse CIF file to extract lattice parameters
        self.parse_cif()
        
        

    def parse_cif(self):
        """Parse CIF file to extract lattice parameters."""
        with open(self.cif_file, 'r') as f:
            lines = f.readlines()
        
        atoms = read(self.cif_file)
        self.lattice_parameters = atoms.get_cell()
        self.a, self.b, self.c = self.lattice_parameters.lengths()
        self.alpha, self.beta_lattice, self.gamma = self.lattice_parameters.angles()
        self.atom_positions = atoms.get_scaled_positions()
        self.atom_elements = atoms.get_chemical_symbols()
        if self.verbose:
            # Print lattice parameters
            print('Crystal structure loaded from CIF:')
            print('Lattice parameters: a=%.4f, b=%.4f, c=%4f, alpha=%d, beta=%d, gamma=%d' %
                    (self.a, self.b, self.c, round(self.alpha), round(self.beta_lattice), round(self.gamma)))
            
            # Print atomic positions
            for i, frac_coord in enumerate(self.atom_positions):
                print(f"Atom {self.atom_elements[i]}: {frac_coord}")
        

class WAXSProcessor(CrystalStructure):
    """
    Unified WAXS data processor compatible with multiple beamlines.
    Handles azimuthal and radial profile extraction, and visualization.
    """
    
    def __init__(self,
                 file: str,                 
                 instrument='ID02',
                 structure = None,
                 binning: int = 1,
                 mask=None,
                 mapping = True,
                 output_dir=None):
        """
        Initialize WAXS data processor.
        
        Parameters:
        -----------
        file : str
            Path to data file
        instrument : str
            'ID02', 'SWING', or 'LGC'
        structure : CrystalStructure or None
            Crystal structure instance
        binning : int
            Downsampling factor
        mask : str
            Path to mask file (EDF format)
        mapping : bool
            project data onto reciprocal space (Q//, Q⊥)
        output_dir : str or None
            Directory to save outputs
        """
        if structure is not None:
            super().__init__(cif_file=structure.cif_file,                             
                             reflections=structure.reflections,
                             threshold=structure.threshold,
                             verbose=False)
        else:
            print('Please create a CrystalStructure instance to use reflection-based methods.')
        
        self.filepath = file
        self.output_dir = output_dir if output_dir else os.path.dirname(file) # set output directory

        self.mapping = mapping
        
        # Load data using appropriate reader
        if instrument == 'ID02':
            self.file = h5File_ID02(file)
            
        elif instrument == 'LGC':
            self.file = EdfFile(file)
            
        elif instrument == 'SWING':
            self.file = h5File_SWING(file)
            
        else:
            raise ValueError(f"Unknown instrument: {instrument}")

        self.instrument = instrument
        self.file_number = self.file.file_number
       
        self.binning = binning
        self.data = self.file.data.copy()
        
        
        
        # Extract metadata
        self.nb_frames = self.file.nb_frames
        self.num_pixel_x = self.file.num_pixel_x
        self.num_pixel_z = self.file.num_pixel_z
        self.pixel_size_x = self.file.pixel_size_x
        self.pixel_size_z = self.file.pixel_size_z
        self.wl = self.file.wl
        self.x_center = self.file.x_center
        self.z_center = self.file.z_center
        self.D = self.file.D
        self.bin_x = self.file.bin_x
        self.bin_y = self.file.bin_y
        self.samplename = self.file.samplename.replace(' ', '_')
        self.B = self.file.B
       
        # Apply binning if needed
        if self.binning != 1:
            print(f'Applying binning factor: {self.binning}')
            self.pixel_size_x *= self.binning
            self.pixel_size_z *= self.binning
        # Load and apply mask
        if mask is not None:
            if mask.split('.')[-1] == 'edf':
                maskimage = fabio.open(mask)
                self.mask = mask
                self.maskdata = maskimage.data
                
            else:
                raise ValueError('Mask must be in EDF format')
        else:
            self.maskdata = np.zeros_like(self.data)   
        # Average frames if needed
        if len(self.data.shape) == 3:
            self.data = np.mean(self.data, axis=0)

        # build pyFAI azimuthal integrator
        detector = pyFAI.detectors.Detector(pixel1=self.pixel_size_x, pixel2=self.pixel_size_z)
        self.ai = AzimuthalIntegrator(dist=self.D, detector=detector)
        self.ai.setFit2D(self.D * 1000, self.x_center, self.z_center, wavelength=self.wl * 1e10)

        # Compute Q grids if mapping is enabled
        if self.mapping:
            print('Computing Q components...')
            # Compute Q_parallel and Q_perpendicular
            self.q_parr = np.zeros(self.data.shape, dtype='float')
            self.q_perp = np.zeros(self.data.shape, dtype='float')
            self.qx = np.zeros(self.data.shape,dtype='float')
            self.qy = np.zeros(self.data.shape,dtype='float')
            self.qz = np.zeros(self.data.shape,dtype='float')
            self.norm_Q = np.zeros(self.data.shape, dtype='float')
            self.thetaB = np.zeros(self.data.shape, dtype='float')
            self.beta = np.zeros(self.data.shape, dtype='float')
            self.phi=np.zeros(self.data.shape, dtype='float')
            
            for i in range(self.num_pixel_z):
                delta_i = (i - self.x_center) * self.pixel_size_x
                for j in range(self.num_pixel_x):
                    delta_j = (j - self.z_center) * self.pixel_size_z
                    denom = (self.D ** 2 + delta_i ** 2 + delta_j ** 2) ** (1/2)
                    a = 2 * np.pi / self.wl
                    self.q_parr[j,i] = a * delta_i / denom  # qx
                    self.q_perp[j,i] = (a / denom) * (delta_j ** 2 + (self.D - denom) ** 2) ** (1/2)
                    self.qy[j,i]=(a/denom)*(self.D-denom)
                    self.qz[j,i]=(a/denom)*delta_j
                    self.norm_Q[j,i] = np.sqrt(self.q_parr[j,i] ** 2 + self.q_perp[j,i] ** 2)
                    self.thetaB[j,i] = np.arcsin(self.q_parr[j,i] / self.norm_Q[j,i]) * 180 / np.pi  # in degrees
                    self.beta[j,i] = np.arccos(self.q_parr[j,i] / self.norm_Q[j,i]) * 180 / np.pi  # in degrees
                    # build phi_array
                    try:
                        self.phi[j,i]=90-(np.arctan(((i-self.z_center)/(j-self.x_center)))*(180/np.pi))
                        phi_error=False
                    except:
                        phi_error=True
                        pass
            if phi_error:
                print("phi values could not be computed (divide by zero)")
                print("Azimuthal profiles should be plotted against beta only")
            self.qx = self.q_parr

    def plot2D(self, cmap='jet', vmin=-5, vmax=0, add_rings=False,save=True):

        if not self.mapping:
            print("Use mapping=True to use this method")
            return

        # ----------------------------------------------------
        # Normalize data (0–1) then clip for LogNorm
        # ----------------------------------------------------
        data = self.data.astype(float)
        data = data / np.max(data)
        data = np.clip(data, 1e-12, None)

        # ----------------------------------------------------
        # Handle vmin/vmax given as log10 values
        # ----------------------------------------------------
        if vmin is None:
            vmin_val = data.min()
        else:
            vmin_val = 10 ** vmin

        if vmax is None:
            vmax_val = data.max()
        else:
            vmax_val = 10 ** vmax

        # Make sure values are valid
        if vmin_val <= 0:
            vmin_val = 1e-12
        if vmax_val <= vmin_val:
            vmax_val = data.max()

        norm = LogNorm(vmin=vmin_val, vmax=vmax_val)

        # ----------------------------------------------------
        # Prepare figure (single axis)
        # ----------------------------------------------------
        fig, ax = plt.subplots(figsize=(7, 6), dpi=200)

        # ----------------------------------------------------
        # q-space mesh
        # ----------------------------------------------------
        q_parr_scaled = 1e-10 * self.q_parr
        q_perp_scaled = 1e-10 * self.q_perp

        q_parr_edges = np.zeros((q_parr_scaled.shape[0] + 1,
                                q_parr_scaled.shape[1] + 1))
        q_perp_edges = np.zeros((q_perp_scaled.shape[0] + 1,
                                q_perp_scaled.shape[1] + 1))

        q_parr_edges[:-1, :-1] = q_parr_scaled
        q_parr_edges[:-1, -1]  = q_parr_scaled[:, -1]
        q_parr_edges[-1, :-1]  = q_parr_scaled[-1, :]
        q_parr_edges[-1, -1]   = q_parr_scaled[-1, -1]

        q_perp_edges[:-1, :-1] = q_perp_scaled
        q_perp_edges[:-1, -1]  = q_perp_scaled[:, -1]
        q_perp_edges[-1, :-1]  = q_perp_scaled[-1, :]
        q_perp_edges[-1, -1]   = q_perp_scaled[-1, -1]

        # ----------------------------------------------------
        # Plot q-space intensity
        # ----------------------------------------------------
        mesh1 = ax.pcolormesh(
            q_parr_edges,
            q_perp_edges,
            data,
            shading='flat',
            cmap=cmap,
            norm=norm
        )

        cbar = fig.colorbar(mesh1, ax=ax)
        cbar.set_label('Intensity (log scale)', fontsize=14)

        ax.set_xlabel(r'$q_{\parallel}$ (Å$^{-1}$)',fontsize=14)
        ax.set_ylabel(r'$q_{\perp}$ (Å$^{-1}$)', fontsize=14)
        ax.set_title(f'Img{self.file_number:05d}_{self.samplename}_{self.B}mT')
        ax.set_aspect('equal')

        # ----------------------------------------------------
        # Add q-rings for reflections
        # ----------------------------------------------------
        if add_rings:
            colors = ['black', 'red', 'blue', 'green', 'white']
            theta = np.linspace(0, np.pi, 200)

            for n, reflection in enumerate(self.reflections):
                q = self.q_hkl(reflection)
                qx = q * np.cos(theta)
                qy = q * np.sin(theta)

                ax.plot(qx, qy,
                        color=colors[n % len(colors)],
                        linestyle='dashed', linewidth=1)

                # Label
                angle_text = (10 + 10 * n) * np.pi / 180
                ax.text(
                    0.9 * q * np.cos(angle_text),
                    0.9 * q * np.sin(angle_text),
                    f'{reflection}',
                    color='white',
                    fontsize=9,
                    ha='right',
                    va='top',
                    bbox=dict(
                        facecolor=colors[n % len(colors)],
                        alpha=0.8,
                        edgecolor=colors[n % len(colors)]
                    ))

        # ----------------------------------------------------
        # Save & show
        # ----------------------------------------------------
        plt.tight_layout()

        if save:
            path = os.path.join(self.output_dir, 'Images')
            os.makedirs(path, exist_ok=True)
            figname = f'{path}/Img{self.file_number:05d}_{self.samplename}_{self.B}mT_WAXS.png'
            plt.savefig(figname, dpi=200)

        plt.show()


   # Functions below have been introduced to extract azimuthal profiles from diffraction image, based on peak indexing

    def d_hkl(self,reflection):
        
        """Compute interplanar spacing d_hkl for given Miller indices."""
        h=reflection[0];k=reflection[1];l=reflection[2]
        # Convert angles from degrees to radians
        if self.alpha!=90 or self.beta_lattice!=90:
            raise Exception("Triclinic systems not implemented")
        
        else:
            alpha = np.radians(self.alpha)
            beta = np.radians(self.beta_lattice)
            gamma = np.radians(self.gamma)
            
            # Compute the denominator of the general formula for d_hkl
                
            term1 = h**2 / ((self.a**2)*(np.sin(gamma))**2) + k**2 / ((self.b**2)*(np.sin(gamma))**2) + l**2 / (self.c**2)-2 * (h * k * np.cos(gamma) / (self.a * self.b*(np.sin(gamma))**2))
            

            # Calculate d_hkl
            d_hkl = np.sqrt(1 / (term1))
        return d_hkl
        
    
    def theta_hkl(self,reflection):
        """Compute Bragg angle"""        
        return np.arcsin(self.wl/(2*self.d_hkl(reflection)))
        
    
    def q_hkl(self,reflection):
        """ Computes the q (norm of Q vector) value for a given reflexion"""
        return 4*np.pi*np.sin(self.theta_hkl(reflection))/self.wl
        
    
    def pixelindexes_constantq(self, reflection=None, qvalue=None):
        """Find detector pixels corresponding to a given Q value."""
        if self.mapping:            
            if reflection is None:
                raise ValueError("Reflection must be provided when cif file is specified.")
            q = self.q_hkl(reflection)
            constantq_pixel_indexes = np.argwhere((np.abs(1e-10 * self.norm_Q[:, :] - q) / q) <= self.threshold)
            return  constantq_pixel_indexes
        else:
            print('Use mapping=True to use this method')
        
    
    
    def extract_azimuthal_profile(self,reflection,plot=False,save=True):
        if self.mapping:
            
            poi = self.pixelindexes_constantq(reflection=reflection)
            
            beta = []
            data = []
            for pixel in poi:
                i, j = pixel
                beta.append(self.beta[i, j])
                data.append(self.data[i, j])
            results = list(zip(beta, data))
            results = sorted(results)
            # Remove data (=0) and corresponding beta induced by detector gaps
            results = [(b, d) for b, d in results if d != 0]
            if results:
                beta, data = zip(*results)
                beta = np.array(beta)
                data = np.array(data)
            else:
                beta = np.array([])
                data = np.array([])
            # Save azimuthal profile data
            
            if save:
                path=os.path.join(self.output_dir,'Azimuthal_Profiles')            
                os.makedirs(path,exist_ok=True)
                ref_str = ''.join(map(str,reflection))
                outputfile=f"{path}/{self.samplename}_B={self.B}mT_Img{self.file_number}_reflection{ref_str}.csv"
                np.savetxt(outputfile, np.column_stack((beta, data)),delimiter=',')
            if plot:
                plt.figure(figsize=(6,4),dpi=200)
                plt.plot(beta,data,marker='o',linestyle='-',markersize=3)
                plt.xlabel('Beta (degrees)')
                plt.ylabel('Intensity (a.u.)')
                plt.title(f'Azimuthal profile for reflection {reflection}')
                plt.grid()
                plt.tight_layout()
                plt.show()
            return beta, data
        else:
            print('Use mapping=True to use this method')