import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py
import fabio
import os
import re
import math
import glob
from pathlib import Path
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import pandas as pd

# PyFAI imports
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
import pyFAI.detectors

# SasModels imports (optional)
try:
    from sasmodels.core import load_model 
    from sasmodels.data import Data2D
    from sasmodels.direct_model import DirectModel
    SASMODELS_AVAILABLE = True
except ImportError:
    SASMODELS_AVAILABLE = False
    print("Warning: sasmodels not available. Form factor calculations will be disabled.")
try:
    from waxsprocessor import WAXSProcessor
    WAXS=True
except ImportError:
    WAXS=False
try:
    from saxsprocessor import SAXSProcessor
    SAXS=True
except ImportError:
    SAXS= False


class NematicOrderCalculator:
    """Calculate nematic order parameter S using Maier-Saupe distribution."""
    
    def __init__(self, form_factor=None):
        """Initialize with optional form factor object."""
        self.form_factor = form_factor
        
        if form_factor is not None:
            print(f" NematicOrderCalculator initialized with CylinderFormFactor")
            print(f"  Form factor profiles will be extracted on-the-fly for each q")
        else:
            print(f" NematicOrderCalculator initialized without form factor")
            print(f"  Pure Maier-Saupe fitting will be used")
    
    @staticmethod
    def Z_MS(m, x0):
        """Normalization constant for Maier-Saupe distribution."""
        beta0 = np.radians(x0)
        integrand = lambda beta: np.exp(m * np.cos(beta - beta0)**2) * np.sin(beta)
        Z, _ = quad(integrand, 0, np.pi, epsabs=1e-12, epsrel=1e-12)
        return Z
    
    @staticmethod
    def ms_distribution(beta, m, x0):
        """Normalized Maier-Saupe distribution."""
        beta0 = np.radians(x0)
        D = np.exp(m * np.cos(beta - beta0)**2)
        return D / NematicOrderCalculator.Z_MS(m, x0)
    
    @staticmethod
    def compute_S(m, x0):
        """Compute nematic order parameter S from Maier-Saupe parameters."""
        num_integrand = lambda theta: (0.5 * (3*np.cos(theta - np.radians(x0))**2 - 1)) \
                                    * np.exp(m*np.cos(theta - np.radians(x0))**2) * np.sin(theta)
        den_integrand = lambda theta: np.exp(m*np.cos(theta - np.radians(x0))**2) * np.sin(theta)

        num, _ = quad(num_integrand, 0, np.pi, epsabs=1e-10, epsrel=1e-10)
        den, _ = quad(den_integrand, 0, np.pi, epsabs=1e-10, epsrel=1e-10)

        return num / den
    
    def convolve_with_form_factor(self, chi_array, m, x0, n_phi=360):
        """
        Compute azimuthal profile by convolving form factor with Maier-Saupe distribution.
        
        Parameters:
        -----------
        chi_array : array
            Azimuthal angles for calculation (°)
        m, x0 : float
            Maier-Saupe parameters
        n_phi : int
            Number of integration points
            
        Returns:
        --------
        intensities : array
            Computed intensities
        """
        if self.chi_ff is None or self.I_ff is None:
            raise ValueError("Form factor profile not provided")
            
        phi_ref = self.chi_ff[np.argmax(self.I_ff)]
        
        chi_extended = np.concatenate([self.chi_ff - 360, self.chi_ff, self.chi_ff + 360])
        I_extended = np.concatenate([self.I_ff, self.I_ff, self.I_ff])
        ff_interp = interp1d(chi_extended, I_extended, kind='cubic', 
                            bounds_error=False, fill_value='extrapolate')
        
        phi = np.linspace(0, 360, n_phi, endpoint=False)
        d_phi = phi[1] - phi[0]
        
        phi_rad = np.radians(phi)
        x0_rad = np.radians(x0)
        f_ms_unnormalized = np.exp(m * np.cos(phi_rad - x0_rad)**2)
        integral_f_ms = np.sum(f_ms_unnormalized) * d_phi
        f_ms = f_ms_unnormalized / integral_f_ms
        
        intensities = []
        for chi in chi_array:
            angles_for_ff = chi - phi + phi_ref
            F2 = ff_interp(angles_for_ff)
            I = np.sum(f_ms * F2) * d_phi
            intensities.append(I)
        
        return np.array(intensities)
    
    def fit_azimuthal_profile(self, theta_exp, I_exp, 
                             qvalue_ff = 0.034, 
                             threshold_ff = 0.001,
                             target=90.0, 
                             smooth=True, 
                             window_length=9, 
                             polyorder=2, 
                             plot=False,
                             apply_mirror=False,
                             processor=None,
                             verbose=True):
        """
        Fit experimental azimuthal profile with Maier-Saupe model.
        
        Parameters:
        -----------
        theta_exp, I_exp : arrays
            Experimental data
        qvalue_ff : float
            Q value for form factor extraction (if applicable)
        threshold_ff : float
            Threshold for form factor extraction
        target : float
            Initial value for x0 (°)
        smooth : bool
            Apply Savitzky-Golay smoothing
        window_length, polyorder : int
            Smoothing parameters
        plot : bool
            Display result
        apply_mirror : bool
            Apply mirror symmetry to the data
            
        Returns:
        --------
        results : dict
            Dictionary with fit results including S parameter
        """
        if apply_mirror:
            theta_exp, I_exp = self.mirror_profile(theta_exp, I_exp)
        if smooth:
            I_fit_data = savgol_filter(I_exp, window_length, polyorder)
        else:
            I_fit_data = I_exp.copy()
        
        # Filtrer les points à intensité nulle
        mask_nonzero = I_exp > 0
        theta_exp = theta_exp[mask_nonzero]
        I_fit_data = I_exp[mask_nonzero]

        
        #  pas de données -> fit impossible
        if I_fit_data.size == 0:
            if verbose:
                print("⚠️ Aucun point non nul : fit annulé, S = 0")
            return {
                'I0': 0,
                'm': 0,
                'x0': target,
                'a': 0,
                'b': 0,
                'S': 0.0,
                'R2': 0.0,
                'I_model': np.zeros_like(theta_exp),
                'popt': None,
                'pcov': None,}
        
        # Normalize
        I_fit_data = I_fit_data / np.max(I_fit_data)

        
        
        if self.form_factor is not None:
            self.chi_ff, self.I_ff = self.form_factor.extract_azim_profile_formfactor(
                self.form_factor.I_ff2D, q0=qvalue_ff, threshold=threshold_ff, plot=False)
        else:
            self.chi_ff, self.I_ff = None, None
        if self.chi_ff is not None and self.I_ff is not None:
            # Fit with form factor convolution
            def model_func(theta, I0, m, x0, a, b):
                I_ms = self.convolve_with_form_factor(theta, m, x0)
                return I0 * I_ms + a * theta + b
        else:
            # Fit with pure Maier-Saupe
            def model_func(theta, I0, m, x0, a, b):
                theta_rad = np.radians(theta)
                I_ms = self.ms_distribution(theta_rad, m, x0)
                return I0 * I_ms + a * theta + b
        
        p0 = (1.0, 5.0, target, 0.0, np.median(I_fit_data))
        bounds = ([0, 0, target-45, -np.inf, -np.inf],
                 [np.inf, np.inf, target+45, np.inf, np.inf])
        
        try:
            popt, pcov = curve_fit(model_func, theta_exp, I_fit_data, 
                                  p0=p0, bounds=bounds, maxfev=20000)
        except RuntimeError as e:
            print(f"Fitting error: {e}")
            return None
        
        I0_opt, m_opt, x0_opt, a_opt, b_opt = popt
        S_opt = self.compute_S(m_opt, x0_opt)
        I_model = model_func(theta_exp, *popt)
        
        residuals = I_fit_data - I_model
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((I_fit_data - np.mean(I_fit_data))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(theta_exp, I_fit_data / np.max(I_fit_data), 'o', 
                    label='Experimental', alpha=0.6)
            plt.plot(theta_exp, I_model / np.max(I_model), '-', 
                    label=f'Fit (m={m_opt:.2f}, S={S_opt:.3f}, R²={r_squared:.3f})', linewidth=2)
            if self.chi_ff is not None:
                plt.plot(self.chi_ff, self.I_ff / np.max(self.I_ff), '--', 
                        label='Form factor', alpha=0.5)
            plt.xlabel('θ (°)')
            plt.ylabel('Normalized intensity')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.title(f'Maier-Saupe fit (x0={x0_opt:.1f}°)')
            plt.tight_layout()
            if processor is not None:
            
                try: #position x
                    figname=os.path.join(os.path.join(processor.path,'azimuthal_profiles'),f'{processor.samplename}_B={processor.B}_x={float(processor.x):.2f}_z={float(processor.z):.2f}_Img{processor.file_number}_nematic_determination.png')
                except:
                    figname=os.path.join(os.path.join(processor.path,'azimuthal_profiles'),f'{processor.samplename}_B={processor.B}_Img{processor.file_number}_nematic_determination.png')
                plt.savefig(figname)
            if verbose:
                print(f'Fit image saved in {figname}')
            #    except:
            #        pass
        
        x0_opt = (x0_opt + 180) % 360 - 180
        
        results = {
            'I0': I0_opt,
            'm': m_opt,
            'x0': x0_opt,
            'a': a_opt,
            'b': b_opt,
            'S': S_opt,
            'R2': r_squared,
            'I_model': I_model,
            'popt': popt,
            'pcov': pcov 
        }
        relevant_items = ['x0','S','R2']
        if verbose:
            print("\nFitting results:")
            for key, value in results.items():
                if key in relevant_items:
                    print(f"{key}: {value}")
        return results
    
    @staticmethod
    def mirror_profile(theta_exp, I_exp, center=180):
        """
        Remplit les trous dans le profil azimuthal avec la symétrie miroir.
        Ne duplique que les données manquantes, pas l'intégralité du profil.
        """
        theta_exp = np.array(theta_exp)
        I_exp = np.array(I_exp)
        
        # Mapper les angles dans [-180, 180]
        theta_norm = ((theta_exp + 180) % 360) - 180
        
        # Créer une grille dense pour déterminer les trous
        theta_grid = np.linspace(-180, 180, 361)
        I_grid = np.full_like(theta_grid, np.nan)
        
        # Placer les données expérimentales
        for t, I in zip(theta_norm, I_exp):
            idx = np.argmin(np.abs(theta_grid - t))
            if np.isnan(I_grid[idx]) or I > I_grid[idx]:  # Prendre la valeur la plus forte
                I_grid[idx] = I
        
        # Identifier les trous (NaN)
        holes = np.isnan(I_grid)
        
        # Pour chaque trou, essayer de le remplir avec le symétrique
        for i in np.where(holes)[0]:
            theta_hole = theta_grid[i]
            theta_mirror = (2*center - theta_hole) % 360
            theta_mirror = ((theta_mirror + 180) % 360) - 180
            
            # Trouver l'indice le plus proche du symétrique
            idx_mirror = np.argmin(np.abs(theta_grid - theta_mirror))
            
            if not np.isnan(I_grid[idx_mirror]):
                I_grid[i] = I_grid[idx_mirror]
        
        # Retourner les données interpolées (ignorer les NaN restants)
        mask_valid = ~np.isnan(I_grid)
        return theta_grid[mask_valid], I_grid[mask_valid]






# ============================================================================
# CYLINDER FORM FACTOR CALCULATION USING SASMODELS
# ============================================================================

if SASMODELS_AVAILABLE:
    class CylinderFormFactor:
        """
        Calculate 2D cylinder form factor.
        Azimuthal profiles are extracted separately for each q by BatchNematic.
        """
        
        def __init__(self,
                     processor=None,
                     npix1=None, 
                     npix2=None, 
                     pix_size=None, 
                     distance=None, 
                     wl=None,  
                     # Model parameters
                     radius=78,
                     L=840,
                     theta=90,
                     phi=0.4,
                     radius_pd=0.3,
                     L_pd=0.75,
                     phi_pd=0,
                     theta_pd=0,
                     background=0.00001,
                     scale=1,
                     plot=False,
                     verbose=False):
            """
            Initialize cylinder form factor calculation.
            
            Parameters:
            -----------
            processor : SAXSProcessor
                SAXS data processor (detector geometry inherited if provided)
            npix1, npix2 : int
                Detector pixels (required if processor not provided)
            pix_size : float
                Pixel size in m (required if processor not provided)
            distance : float
                Sample-detector distance in m (required if processor not provided)
            wl : float
                Wavelength in Å (required if processor not provided)
            radius : float
                Cylinder radius (Å)
            L : float
                Cylinder length (Å)
            theta, phi : float
                Orientation angles (°)
            radius_pd, L_pd : float
                Polydispersity
            background : float
                Background intensity
            scale : float
                Scaling factor
            plot : bool
                Display 2D form factor pattern
            verbose : bool
                Print initialization details 
            """
            # Inherit detector geometry from processor if provided
            if processor is not None:
                self.npix1 = processor.num_pixel_x
                self.npix2 = processor.num_pixel_z
                self.wl = processor.wl * 1e10  # Convert to Å
                self.D = processor.D
                self.distance = self.D
                self.pixel_size = processor.pixel_size_x
                if verbose:
                    print(f"✓ Detector geometry inherited from {processor.samplename}")
                    print(f"  - Detector: {self.npix1} x {self.npix2} pixels")
                    print(f"  - Pixel size: {self.pixel_size*1e3:.3f} mm")
                    print(f"  - Distance: {self.D:.3f} m")
                    print(f"  - Wavelength: {self.wl:.5f} Å")
            else:
                # Use manual parameters
                if None in [npix1, npix2, pix_size, distance, wl]:
                    raise ValueError("Either provide a processor or all detector parameters")
                self.npix1 = npix1
                self.npix2 = npix2
                self.wl = wl
                self.D = distance
                self.distance = distance
                self.pixel_size = pix_size
                if verbose:
                    print("✓ Using manual detector geometry")
            
            # Store model parameters
            self.radius = radius
            self.L = L
            self.theta = theta
            self.phi = phi
            self.radius_pd = radius_pd
            self.L_pd = L_pd
            self.phi_pd = phi_pd
            self.theta_pd = theta_pd
            self.background = background
            self.scale = scale

            # Compute Q grids
            if verbose:
                print("\nComputing Q grids...")
            i_vals = np.arange(-self.npix2 // 2, self.npix2 // 2)
            j_vals = np.arange(-self.npix1 // 2, self.npix1 // 2)
            I, J = np.meshgrid(i_vals, j_vals, indexing="xy")
            delta_i = I * self.pixel_size
            delta_j = J * self.pixel_size
            denom = np.sqrt(self.D**2 + delta_i**2 + delta_j**2)
            a = 2.0 * np.pi / self.wl
            eps = 1e-10
            self.Qx = (a / denom) * delta_i + eps
            self.Qz = (a / denom) * delta_j + eps
            self.Qy = (a / denom) * (self.D - denom) + eps

            # Compute 2D form factor (stored for later azimuthal extraction)
            if verbose:
                print("Computing 2D cylinder form factor...")
            self.I_ff2D = self.compute_cylinder_form_factor(plot=plot)
            if verbose:
                print(f"✓ Form factor computed (shape: {self.I_ff2D.shape})")
                print("\nNote: Azimuthal profiles will be extracted by BatchNematic for each q value")
        
        def compute_cylinder_form_factor(self, plot=False, vmin=-6):
            """
            Compute 2D form factor for polydisperse cylinders using sasmodels.
            
            Parameters:
            -----------
            plot : bool
                Display 2D pattern
            vmin : float
                Minimum value for log scale (10^vmin)
                
            Returns:
            --------
            Iq : array
                Flattened 2D intensity array
            """
            params = {
                "phi": self.phi, 
                "theta": self.theta,
                "radius": self.radius,
                "radius_pd_type": 'gaussian',
                "radius_pd": self.radius_pd,
                "radius_pd_n": 50,
                "radius_pd_nsigma": 6,
                "length": self.L,
                "length_pd": self.L_pd,
                "length_pd_n": 50,
                "length_pd_nsigma": 6,
                "background": self.background,
                "scale": self.scale
            }
            
            # Load sasmodels model
            kernel = load_model("cylinder")
            z = np.zeros_like(self.Qx)
            data2d = Data2D(x=self.Qx, y=self.Qz, z=z, dx=None, dy=None, dz=z) 
            calculator = DirectModel(data2d, kernel)
            Iq = calculator(**params)
            
            if plot:
                Iq_2D = Iq.reshape(self.npix1, self.npix2)
                x_1d = self.Qx[0, :]
                y_1d = self.Qz[:, 0]
                extent = [x_1d[0], x_1d[-1], y_1d[0], y_1d[-1]]
                
                plt.figure(figsize=(8, 6))
                plt.imshow(
                    Iq_2D / Iq_2D.max(),
                    origin='lower',
                    extent=extent,
                    norm=LogNorm(vmin=10.0 ** vmin, vmax=1.0),
                    cmap='jet'
                )
                plt.colorbar(label='Normalized Intensity')
                plt.xlabel('Qx (Å⁻¹)')
                plt.ylabel('Qz (Å⁻¹)')
                plt.title(f'Cylinder Form Factor\nR={self.radius}Å, L={self.L}Å')
                plt.tight_layout()
                plt.show()

            return Iq

        def extract_azim_profile_formfactor(self, Iflat, q0=0.034, threshold=0.001, plot=False):
            """
            Extract azimuthal profile at constant q using pyFAI.
            
            Parameters:
            -----------
            Iflat : array
                Flattened 2D intensity
            q0 : float
                Q-value for extraction (Å⁻¹)
            threshold : float
                Relative Q-range for integration
            plot : bool
                Display azimuthal profile
                
            Returns:
            --------
            chi : array
                Azimuthal angles (°)
            I : array
                Intensity vs chi
            """
            I_2D = Iflat.reshape((self.npix1, self.npix2))
            detector = pyFAI.detectors.Detector(pixel1=self.pixel_size, pixel2=self.pixel_size)
            ai = AzimuthalIntegrator(dist=self.distance, detector=detector)
            ai.setFit2D(self.distance * 1000, self.npix2/2, self.npix1/2, wavelength=self.wl)
            
            chi, I = ai.integrate_radial(
                I_2D, 
                540, 
                radial_range=(q0 * (1 - threshold), q0 * (1 + threshold)), 
                radial_unit="q_A^-1"
            )
            
            if plot:
                plt.figure(figsize=(8, 5))
                plt.plot(chi, I, linewidth=2)
                plt.xlabel('χ (°)', fontsize=12)
                plt.ylabel('Intensity', fontsize=12)
                plt.title(f'Form Factor Azimuthal Profile at q = {q0:.4f} Å⁻¹', fontsize=14)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
            
            return -chi, I
            #return chi, I  # Invert chi to match experimental convention

class BatchNematic:
    """
    Process multiple SAXS files and extract nematic order parameters.
    Simplified: uses a single NematicOrderCalculator with CylinderFormFactor.
    """
    
    def __init__(self, 
                 path, 
                 instrument='ID02',
                 qvalues=None,
                 file_filter='*_eiger2*_raw.h5',
                 threshold=0.0001,
                 binning=1,
                 mask=None,
                 reference_file: str = None,
                 k=1,
                 autosubstract: bool = True,
                 form_factor=None,
                 output_dir=None,
                 structure = None
                 ):
        """
        Initialize batch processor.
        
        Parameters:
        -----------
        path : str
            Directory containing data files
        instrument : str
            Beamline name
        qvalues : list
            Q values for azimuthal profiles (Å⁻¹)
        file_filter : str
            Wildcard pattern for files
        threshold : float
            Q tolerance for azimuthal profile extraction
        binning : int
            Binning factor
        mask : str
            Path to mask file
        reference_file : str
            Path to reference file for background subtraction
        k : float
            Scaling factor for background subtraction
        autosubstract : bool
            Enable automatic background subtraction
        form_factor : CylinderFormFactor or None
            Form factor object (profiles extracted on-the-fly for each q)
        output_dir : str
            Directory to save output files
        structure:
            CrystalStructure instance for WAXS data analysis
          
        """
        if structure is not None:
            self.structure = structure
        self.path = path
        self.instrument = instrument
        self.qvalues = qvalues if qvalues is not None else None
        self.reflections = structure.reflections if structure is not None else None
        self.threshold = threshold
        self.binning = binning
        self.mask = mask
        self.reference_file = reference_file
        self.k = k
        self.autosubstract = autosubstract
        self.h5_filelist = sorted(glob.glob(os.path.join(path, file_filter)), 
                                 key=self._extract_number)
        self.output_dir = output_dir if output_dir is not None else self.path
        
        # Create a single NematicOrderCalculator with the form factor
        self.nematic_calc = NematicOrderCalculator(form_factor=form_factor)
        
        print(f"\n{'=' * 60}")
        print(f"BatchNematic initialized")
        print(f"{'=' * 60}")
        print(f"Path: {self.path}")
        if SAXS:
            print(f"Q values: {self.qvalues}")
        if WAXS:
            print(f'Reflections: {self.structure.reflections}')
        print(f"Found {len(self.h5_filelist)} files to process")
        print(f"Output directory: {self.output_dir}")
        print(f"{'=' * 60}\n")

    def _extract_number(self, file_path):
        """Extract file number for sorting."""
        filename = os.path.basename(file_path)
        extension = filename.split('.')[-1]
        
        if self.instrument == 'ID02':
            number = filename.split('_')[2]
        elif self.instrument == 'SWING':
            number = filename.split('_')[1]
        elif extension == 'edf':
            basename = filename.split('.')[0]
            number = basename.split('_')[-1].split('-')[0]
        else:
            number = '0'
            
        return int(number)
    
    def process_all(self, save_profiles=True, plot=False, apply_mirror=False):
        """
        Process all files and compute nematic order parameters.
        
        Parameters:
        -----------
        save_profiles : bool
            Save azimuthal profiles
        plot : bool
            Display fitting results
        apply_mirror : bool
            Apply mirror symmetry to experimental profiles
            
        Returns:
        --------
        df : DataFrame
            Results table
        """
        results = []
        logfile = os.path.join(self.output_dir, 'batch_processing.log')
        log_lines = []
        
        for file in self.h5_filelist:
            try:
                print(f"\n{'=' * 60}")
                print(f"Processing: {os.path.basename(file)}")
                print(f"{'=' * 60}")
                if SAXS:
                    processor = SAXSProcessor(
                        file,
                        instrument=self.instrument,
                        binning=self.binning,
                        mask=self.mask,
                        reference_file=self.reference_file,
                        k=self.k,
                        autosubstract=self.autosubstract
                    )
                elif WAXS:
                    processor = WAXSProcessor(
                        file,
                        structure=self.structure,
                        mask=self.mask,
                        mapping = True,
                        instrument = self.instrument,
                        output_dir=self.output_dir
                    )
                if SAXS:
                    for qvalue in self.qvalues:
                        print(f"\nProcessing q = {qvalue:.4f} Å⁻¹...")
                        
                        # Extract experimental azimuthal profile
                        chi_exp, I_exp = processor.extract_azimuthal_profile(
                            qvalue, 
                            threshold=self.threshold,
                            save=save_profiles,
                            output_dir=self.output_dir
                        )
                        
                        # Fit with NematicOrderCalculator (form factor extracted on-the-fly)
                        fit_results = self.nematic_calc.fit_azimuthal_profile(
                            chi_exp, I_exp,
                            qvalue_ff=qvalue,  # ← Extract FF profile at this q
                            threshold_ff=self.threshold,
                            target=90.0,
                            smooth=False,
                            plot=plot,
                            apply_mirror=apply_mirror
                        )
                        
                        if fit_results is not None:
                            results.append({
                                'File': os.path.basename(file),
                                'Sample': processor.samplename,
                                'B (mT)': processor.B,
                                'File_Number': processor.file_number,
                                'q (Å⁻¹)': qvalue,
                                'S': fit_results['S'],
                                'm': fit_results['m'],
                                'x0 (°)': fit_results['x0'],
                                'I0': fit_results['I0'],
                                'a (slope)': fit_results['a'],
                                'b (offset)': fit_results['b'],
                                'R²': fit_results['R2']                           
                            })
                            
                            success_msg = (f"  ✓ Success | q={qvalue:.4f} | S={fit_results['S']:.4f} | "
                                        f"m={fit_results['m']:.2f} | R²={fit_results['R2']:.4f}")
                            print(success_msg)
                            log_lines.append(success_msg)
                        else:
                            fail_msg = f"  ✗ Fit failed for q = {qvalue:.4f} Å⁻¹"
                            print(fail_msg)
                            log_lines.append(fail_msg)
                if WAXS:
                    for reflection in self.reflections:
                        ref_str = "".join(str(a) for a in reflection)
                        print(f"\nProcessing reflection = {ref_str}...")
                        
                        # Extract experimental azimuthal profile
                        chi_exp, I_exp = processor.extract_azimuthal_profile(
                            reflection, 
                            save=save_profiles)
                        
                        # Fit with NematicOrderCalculator (form factor extracted on-the-fly)
                        fit_results = self.nematic_calc.fit_azimuthal_profile(
                            chi_exp, I_exp,
                            target=90.0,
                            smooth=False,
                            plot=plot,
                            apply_mirror=apply_mirror
                        )
                        
                        if fit_results is not None:
                            results.append({
                                'File': os.path.basename(file),
                                'Sample': processor.samplename,
                                'B (mT)': processor.B,
                                'File_Number': processor.file_number,
                                'reflection': ref_str,
                                'S': fit_results['S'],
                                'm': fit_results['m'],
                                'x0 (°)': fit_results['x0'],
                                'I0': fit_results['I0'],
                                'a (slope)': fit_results['a'],
                                'b (offset)': fit_results['b'],
                                'R²': fit_results['R2']                           
                            })
                            
                            success_msg = (f"  ✓ Success | reflection={ref_str} | S={fit_results['S']:.4f} | "
                                        f"m={fit_results['m']:.2f} | R²={fit_results['R2']:.4f}")
                            print(success_msg)
                            log_lines.append(success_msg)
                        else:
                            fail_msg = f"  ✗ Fit failed for reflection = {ref_str}"
                            print(fail_msg)
                            log_lines.append(fail_msg)
                
                        
            except Exception as e:
                error_msg = f"ERROR processing {os.path.basename(file)}: {str(e)}"
                print(f"\n{error_msg}")
                log_lines.append(error_msg)
                import traceback
                traceback.print_exc()
        
        # Save log
        print(f"\n{'=' * 60}")
        print("Saving results...")
        print(f"{'=' * 60}")
        
        with open(logfile, 'w') as f:
            f.write('\n'.join(log_lines))
        print(f"✓ Log file saved: {logfile}")
        
        # Create DataFrame
        if len(results) > 0:
            df = pd.DataFrame(results)
            
            # Sort by file number and q value
            if SAXS:
                df = df.sort_values(['File_Number', 'q (Å⁻¹)'])
            if WAXS:
                df = df.sort_values(['File_Number', 'reflection'])

            
            output_csv = os.path.join(self.output_dir, 'nematic_order_results.csv')
            df.to_csv(output_csv, index=False)
            print(f"✓ Results CSV saved: {output_csv}")
            
                      
            return df
        else:
            print("⚠ No successful fits to save")
            return pd.DataFrame()

