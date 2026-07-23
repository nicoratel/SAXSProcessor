from .filereaders import h5File_ID02, h5File_SWING, EdfFile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.signal import savgol_filter
import fabio
import os

import math

from pathlib import Path

from scipy.interpolate import interp1d, griddata

import pandas as pd

# PyFAI imports
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
import pyFAI.detectors


class SAXSProcessor:
    """
    Unified SAXS data processor compatible with multiple beamlines.
    Handles azimuthal and radial profile extraction, caving, and visualization.
    """
    
    def __init__(self,
                 file: str,
                 reference_file: str = None,
                 k =1,
                 autosubstract: bool = True,
                 instrument='ID02',
                 frame = 'mean',
                 binning: int = 1,
                 mask=None,
                 average = True,
                 mapping = False):
        """
        Initialize SAXS data processor.
        
        Parameters:
        -----------
        file : str
            Path to data file
        reference_file : str
            Path to reference data file for background subtraction
        k : float = 1
            Multiplier factor for reference substraction
        autosubstract: bool = True
            Automatic determination of k factor for reference subtraction
        instrument : str
            'ID02', 'SWING', or 'LGC'
        frame : int or 'mean'
            index of frame to consider or data average if 'mean'
        binning : int
            Downsampling factor
        mask : str
            Path to mask file (pyFAI - EDF format)
        
        mapping : bool
            If True, compute Q-space mapping (Q_parallel, Q_perpendicular)
        """
        self.filepath = file
        self.path = os.path.dirname(file)
        
        # Load data using appropriate reader
        if instrument == 'ID02':
            self.file = h5File_ID02(file)
            self.reffile = h5File_ID02(reference_file) if reference_file else None
        elif instrument == 'LGC':
            self.file = EdfFile(file)
            self.reffile = EdfFile(reference_file) if reference_file else None
        elif instrument == 'SWING':
            self.file = h5File_SWING(file, frame = frame)
            self.reffile = h5File_SWING(reference_file) if reference_file else None
                        
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
        try:
            self.x=self.file.x
            self.z=self.file.z
        except:
            print('No motor positions provided. Update filereaders.py if required')
            pass
       
        # Apply binning
        if self.bin_x!=1 or self.bin_y!=1:
            #print(f'Initial pixel size: {self.pixel_size_x}x{self.pixel_size_z}')
            self.pixel_size_x *= self.bin_x
            self.pixel_size_z *= self.bin_y
            #print(f'Due to binning, pixel size was modified to {self.pixel_size_x}x{self.pixel_size_z}')
        # Load and apply mask
        print(f"Data shape après readout par filereader: {self.data.shape}")
        print(f"Number of frames: {self.nb_frames}, Number of pixels (x, z): ({self.num_pixel_x}, {self.num_pixel_z})")
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
            if frame == 'mean':
                print('SAXSProcessor uses data average')
                self.data = np.mean(self.data, axis=0)
            else:
                print(f'Extraction of frame #{frame}/{self.data.shape[0]}')
                self.data = self.data[frame] 

        # build pyFAI azimuthal integrator
        detector = pyFAI.detectors.Detector(pixel1=self.pixel_size_x, pixel2=self.pixel_size_z)
        poni1 = self.pixel_size_z*self.z_center
        poni2 = self.pixel_size_x*self.x_center
        self.ai = AzimuthalIntegrator(poni1 = poni1, poni2= poni2, dist=self.D, detector=detector, wavelength=self.wl)
        
        # Substract reference if provided
        if self.reffile is not None:
            ref_data = self.reffile.data
            if len(ref_data.shape) == 3:
                ref_data = np.mean(ref_data, axis=0)
            if autosubstract:
                k = self.determine_k(self.data, ref_data)
            self.data = self.data - k * ref_data
        
        # Store mapping parameter
        self.mapping = mapping
        
        # Compute Q grids if mapping is enabled
        if self.mapping:
            print('Computing Q components...')
            # Compute Q_parallel and Q_perpendicular
            self.q_parr = np.zeros(self.data.shape, dtype='float')
            self.q_perp = np.zeros(self.data.shape, dtype='float')
            self.qx = np.zeros(self.data.shape, dtype='float')
            self.qy = np.zeros(self.data.shape, dtype='float')
            self.qz = np.zeros(self.data.shape, dtype='float')
            self.norm_Q = np.zeros(self.data.shape, dtype='float')
            self.thetaB = np.zeros(self.data.shape, dtype='float')
            self.beta = np.zeros(self.data.shape, dtype='float')
            self.phi = np.zeros(self.data.shape, dtype='float')
            
            phi_error = False
            for i in range(self.num_pixel_z):
                delta_i = (i - self.x_center) * self.pixel_size_x
                for j in range(self.num_pixel_x):
                    delta_j = (j - self.z_center) * self.pixel_size_z
                    denom = (self.D ** 2 + delta_i ** 2 + delta_j ** 2) ** (1/2)
                    a = 2 * np.pi / self.wl
                    self.q_parr[j, i] = a * delta_i / denom  # qx
                    self.q_perp[j, i] = (a / denom) * (delta_j ** 2 + (self.D - denom) ** 2) ** (1/2)
                    self.qy[j, i] = (a / denom) * (self.D - denom)
                    self.qz[j, i] = (a / denom) * delta_j
                    self.norm_Q[j, i] = np.sqrt(self.q_parr[j, i] ** 2 + self.q_perp[j, i] ** 2)
                    if self.norm_Q[j, i] != 0:
                        self.thetaB[j, i] = np.arcsin(self.q_parr[j, i] / self.norm_Q[j, i]) * 180 / np.pi  # in degrees
                        self.beta[j, i] = np.arccos(self.q_parr[j, i] / self.norm_Q[j, i]) * 180 / np.pi  # in degrees
                    # build phi_array
                    try:
                        if j != self.x_center:
                            self.phi[j, i] = 90 - (np.arctan((i - self.z_center) / (j - self.x_center)) * (180 / np.pi))
                    except:
                        phi_error = True
                        pass
            
            if phi_error:
                print("Some phi values could not be computed (divide by zero)")
                print("Azimuthal profiles should be plotted against beta only")
            
            self.qx = self.q_parr
            print('Q-space mapping completed.')
        
        

          
    def update_ai(self, x_center = None, z_center = None, D = None):
        """
        Update the pyFAI azimuthal integrator with new parameters.
        
        Parameters:
        -----------
        x_center : float
            New x center (in pixels)
        z_center : float
            New z center (in pixels)
        D : float
            New sample-detector distance (in meters)
        """
        if x_center is not None:
            self.x_center = x_center
        if z_center is not None:
            self.z_center = z_center
        if D is not None:
            self.D = D
        
        poni1 = self.pixel_size_z * self.z_center
        poni2 = self.pixel_size_x * self.x_center
        
        detector = pyFAI.detectors.Detector(pixel1=self.pixel_size_x, pixel2=self.pixel_size_z)
        
        self.ai = AzimuthalIntegrator(poni1=poni1, poni2=poni2, dist=self.D, detector=detector, wavelength=self.wl)
    
    def determine_k(self, data, ref_data):
        """
        Determine optimal k factor for reference subtraction.
        
        Parameters:
        -----------
        data : array
            Sample data
        ref_data : array
            Reference data
            
        Returns:
        --------
        k_opt : float
            Optimal k factor
        """
                
        q, I = self.ai.integrate1d(
            data, 
            1000,
            unit="q_A^-1",
            mask=self.maskdata, 
            normalization_factor=1,
            method="csr"
        )
        q_ref, I_ref = self.ai.integrate1d(
            ref_data, 
            1000,
            unit="q_A^-1",
            mask=self.maskdata, 
            normalization_factor=1,
            method="csr"
        )
        qmax = 0.75 * q.max()
        masq = (q >= qmax)
        k_opt = np.mean(I[masq] / I_ref[masq])
        print('Optimized k factor for reference subtraction:', k_opt)
        return k_opt
    
    def apply_caving(self, max_iter=10):
        """
        Replace masked pixels with symmetric values relative to beam center.
        
        Parameters:
        -----------
        max_iter : int
            Maximum number of iterations
        """
        if not hasattr(self, '_original_maskdata'):
            self._original_maskdata = self.maskdata.copy()
        self.maskdata = self._original_maskdata.copy()
        self.data = np.where(self.maskdata == 1.0, np.nan, self.data)

        for it in range(max_iter):
            modified = False
            for x in range(int(self.num_pixel_x)):
                for z in range(int(self.num_pixel_z)):
                    if np.isnan(self.data[x, z]):
                        xsym = int(2 * self.z_center - x)
                        zsym = int(2 * self.x_center - z)

                        if 0 <= xsym < int(self.num_pixel_x) and 0 <= zsym < int(self.num_pixel_z):
                            if not np.isnan(self.data[xsym, zsym]):
                                self.data[x, z] = self.data[xsym, zsym]
                                self.maskdata[x, z] = 0
                                modified = True

            if not modified:
                break

    def compute_q_grids(self):
        """Compute Qx, Qy, Qz grids for the detector."""
        qx = np.zeros((self.num_pixel_x, self.num_pixel_z), dtype=float)
        qy = np.zeros((self.num_pixel_x, self.num_pixel_z), dtype=float)
        qz = np.zeros((self.num_pixel_x, self.num_pixel_z), dtype=float)
        
        for i in range(self.num_pixel_z):
            delta_i = (i - self.x_center) * self.pixel_size_x
            for j in range(self.num_pixel_x):
                delta_j = (j - self.z_center) * self.pixel_size_z
                denom = np.sqrt(self.D**2 + delta_i**2 + delta_j**2)
                a = 2 * np.pi / self.wl
                qx[j, i] = (a * delta_i / denom) * 1e-10
                qy[j, i] = (a / denom) * (self.D - denom) * 1e-10
                qz[j, i] = (a * delta_j / denom) * 1e-10
                
        return qx, qy, qz

    
    def export_sasview(self, output_dir=None, q_range=None):
        """
        Export the 2D scattering intensity to an ASCII file compatible with SASView.

        This method computes the reciprocal-space grids (qx, qy, qz), applies the
        data mask by setting masked pixels to NaN, optionally restricts the exported
        data to a given q-range, and writes the flattened data to a `.dat` file
        readable by SASView as 2D data.

        The exported file contains three columns:
            - Qx : in-plane scattering vector component
            - Qz : out-of-plane scattering vector component
            - I(Qx, Qz) : scattering intensity

        Parameters
        ----------
        output_dir : str or None, optional
            Directory where the SASView file will be written.
            If None, a default directory `<self.path>/sasview_exports/` is used.
            The directory is created if it does not exist.

        q_range : tuple(float, float) or None, optional
            Tuple `(qmin, qmax)` defining the range of q values to export.
            Only values in the range `[0, qmax]` are kept for both Qx and Qz.
            If `qmin` is not zero, it is automatically reset to zero and a warning
            is printed. If None, the full q-range is exported.

        Notes
        -----
        - Masked pixels (`self.maskdata == 1`) are written as NaN values.
        - The data are flattened before export, as required by SASView for 2D data.
        - The output filename follows the pattern:
        `<output_dir>/<samplename>_<file_number>.dat`.

        Output
        ------
        A text file in ASCII format compatible with SASView 2D data import.
        """


        # ---------- dossier de sortie ----------
        output_dir = output_dir if output_dir is not None else self.path + '/sasview_exports/'
        os.makedirs(output_dir, exist_ok=True)

        outputfilename = f"{output_dir}/{self.samplename}_{self.file_number}.dat"

        # ---------- grilles q ----------
        qx, qy, qz = self.compute_q_grids()

        # ---------- intensité avec masque → NaN ----------
        intensity = self.data.astype(float).copy()
        intensity[self.maskdata == 1] = np.nan

        # ---------- masque q_range (optionnel) ----------
        if q_range is not None:
            qmin, qmax = q_range
            if qmin!=0:
                print('Sasview export is performed in [0,qmax] range. qmin is set equal to 0.')
                qmin=0
            qmask = (
                (qx >= 0) & (qx <= qmax) &
                (qz >= 0) & (qz <= qmax)
            )
        else:
            qmask = np.ones_like(intensity, dtype=bool)

        # ---------- flatten ----------
        qx_flat = qx[qmask]
        qz_flat = qz[qmask]
        intensity_flat = intensity[qmask]

        # ---------- écriture fichier ----------
        header = "Data columns Qx - Qy - I(Qx,Qy)\n"
        header += "ASCII data"

        data2write = np.column_stack((qx_flat, qz_flat, intensity_flat))

        np.savetxt(
            outputfilename,
            data2write,
            header=header,
            comments=""
        )

        print(f"Export to SASView 2D format in : {outputfilename}")
        

    def plot2d_vsq(self, 
            q_range=[0, 0.2],
            cmap='jet',
            log=True,
            grid_size=1000,
            vmin=-4, vmax=0,
            normalize=True,
            q_circles=None,
            q_circles_labels=True,
            output_dir=None,
            rotate90=False):
        """
        Plot 2D SAXS pattern in reciprocal space.
        If mapping=True, plots in (Q//, Q⊥) space, otherwise in (Qx, Qz) space.
        
        Parameters:
        -----------
        q_range : list
            [qmin, qmax] range to display (Å⁻¹)
        cmap : str
            Colormap name
        log : bool
            Use logarithmic scale
        grid_size : int
            Interpolation grid size (only used if mapping=False)
        vmin, vmax : float
            Color scale limits (log scale exponents if log=True)
        normalize : bool
            Normalize intensity
        q_circles : list
            Q values for reference circles
        output_dir : str
            Output directory for figure
        rotate90 : bool
            Rotate image 90° (only used if mapping=False)
        """
        
        # Apply caving if mask provided
        if self.mask is not None:
            if self.mask.split('.')[-1] == 'edf':  
                self.apply_caving()
            else:
                print('Please provide a mask file edited with pyFAI and saved with ".edf" file extension')
        
        # Use mapping if available
        if self.mapping:
            # Plot using q_parr and q_perp (same as WAXSProcessor.plot2D)
            # Apply mask (set masked pixels to NaN for visualization)
            data = self.data.astype(float).copy()
            if hasattr(self, 'maskdata'):
                data[self.maskdata == 1] = np.nan
            
            # Normalize data (0–1) then clip for LogNorm
            data = data / np.nanmax(data)
            data = np.clip(data, 1e-12, None)
            
            # Handle vmin/vmax given as log10 values
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
            
            norm = LogNorm(vmin=vmin_val, vmax=vmax_val) if log else None
            
            # Prepare figure
            fig, ax = plt.subplots(figsize=(7, 6), dpi=200)
            
            # q-space mesh (convert from m⁻¹ to Å⁻¹)
            q_parr_scaled = 1e-10 * self.q_parr
            q_perp_scaled = 1e-10 * self.q_perp
            
            # Create edges for pcolormesh
            q_parr_edges = np.zeros((q_parr_scaled.shape[0] + 1, q_parr_scaled.shape[1] + 1))
            q_perp_edges = np.zeros((q_perp_scaled.shape[0] + 1, q_perp_scaled.shape[1] + 1))
            
            q_parr_edges[:-1, :-1] = q_parr_scaled
            q_parr_edges[:-1, -1] = q_parr_scaled[:, -1]
            q_parr_edges[-1, :-1] = q_parr_scaled[-1, :]
            q_parr_edges[-1, -1] = q_parr_scaled[-1, -1]
            
            q_perp_edges[:-1, :-1] = q_perp_scaled
            q_perp_edges[:-1, -1] = q_perp_scaled[:, -1]
            q_perp_edges[-1, :-1] = q_perp_scaled[-1, :]
            q_perp_edges[-1, -1] = q_perp_scaled[-1, -1]
            
            # Plot
            mesh = ax.pcolormesh(q_parr_edges, q_perp_edges, data, 
                                shading='flat', cmap=cmap, norm=norm)
            
            cbar = fig.colorbar(mesh, ax=ax)
            cbar.set_label('Intensity (log scale)' if log else 'Intensity', fontsize=14)
            
            ax.set_xlabel(r'$q_{\parallel}$ (Å$^{-1}$)', fontsize=14)
            ax.set_ylabel(r'$q_{\perp}$ (Å$^{-1}$)', fontsize=14)
            ax.set_title(f'Img{self.file_number:05d}_{self.samplename}_{self.B}mT')
            ax.set_aspect('equal')
            
            # Set axis limits to q_range if specified
            if q_range is not None:
                qmax = q_range[1]
                ax.set_xlim(-qmax, qmax)
                ax.set_ylim(q_range[0], qmax)
            
            # Add q-circles if requested
            if q_circles is not None:
                theta = np.linspace(0, np.pi, 200)
                colors = ['white', 'red', 'blue', 'green', 'yellow']
                for n, q in enumerate(q_circles):
                    qx_circle = q * np.cos(theta)
                    qy_circle = q * np.sin(theta)
                    ax.plot(qx_circle, qy_circle, color=colors[n % len(colors)],
                           linestyle='dashed', linewidth=1.5, label=f'q={q:.3f}')
                ax.legend()
            
            plt.tight_layout()
            
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                figname = os.path.join(output_dir, 
                                      f'Img{self.file_number:05d}_{self.samplename}_{self.B}mT_qspace.png')
                plt.savefig(figname, dpi=200)
            
            plt.show()
            return
        
        
        else:
            # Original method for non-mapping case
            qx, qy, qz = self.compute_q_grids()
            qnorm = np.sqrt(qx**2 + qy**2 + qz**2)
            
            if q_range is not None:
                qmin, qmax = q_range
                mask = (qx >= -qmax) & (qx <= qmax) & (qz >= -qmax) & (qz <= qmax)
                qx_masked, qz_masked, intensity = qx[mask], qz[mask], self.data[mask]
            else:
                qx_masked, qz_masked, intensity = qx.flatten(), qz.flatten(), self.data.flatten()

            if normalize:
                intensity = intensity / np.nanmax(intensity)

            if len(qx_masked) < 4:
                plt.figure(figsize=(6, 6))
                sc = plt.scatter(qx_masked, qz_masked, c=intensity, cmap=cmap, vmin=vmin, vmax=vmax)
                plt.xlabel("$Q_x$ (Å⁻¹)")
                plt.ylabel("$Q_z$ (Å⁻¹)")
                plt.colorbar(sc, label="Normalized Intensity" if normalize else "Intensity")
                plt.gca().set_aspect('equal')
                plt.show()
                return

            qx_lin = np.linspace(qx_masked.min(), qx_masked.max(), grid_size)
            qz_lin = np.linspace(qz_masked.min(), qz_masked.max(), grid_size)
            QX, QZ = np.meshgrid(qx_lin, qz_lin)
            Z = griddata((qx_masked, qz_masked), intensity, (QX, QZ), method='linear')
            
            if rotate90:
                Z = np.rot90(Z, k=-1)
                QZ, QX = np.rot90(QX, k=-1), np.rot90(QZ, k=-1)

            fig = plt.figure(dpi=150)
            ax = plt.gca()
            
            norm = LogNorm(vmin=10**vmin, vmax=10**vmax) if log else None
            mesh = plt.pcolormesh(QX, -QZ, Z, shading='auto', cmap=cmap, norm=norm)
            plt.xlabel(r"$Q_x$ (Å$^{-1}$)", fontsize=12)
            plt.ylabel(r"$Q_z$ (Å$^{-1}$)", fontsize=12)
            cbar = plt.colorbar(mesh, shrink=0.5, aspect=20)
            cbar.set_label("Normalized Intensity" if normalize else "Intensity", fontsize=12)
            cbar.ax.tick_params(labelsize=12)
            plt.gca().set_aspect('equal')
        
            # Ajout du format_coord pour afficher q, d et θ dans la barre de statut
            def format_coord(x, y):
                r = np.sqrt(x**2 + y**2)
                theta = np.degrees(np.arctan2(y, x))
                if r > 0:
                    d = 2.0 * np.pi / (10.0 * r)
                    return f"q={r:.4f} Å⁻¹, d={d:.4f} nm, chi={theta:.1f}°"
                else:
                    return f"q={r:.4f} Å⁻¹, chi={theta:.1f}°"
            
            ax.format_coord = format_coord
            
            if q_circles is not None:
                colors = ['black', 'purple', 'pink', 'palegreen']
                xmin, xmax = sorted(ax.get_xlim())
                ymin, ymax = sorted(ax.get_ylim())

                for i, q_val in enumerate(q_circles):
                    theta = np.linspace(0, 2*np.pi, 2000)
                    x_circle = q_val * np.cos(theta)
                    y_circle = -q_val * np.sin(theta)

                    mask = (x_circle >= xmin) & (x_circle <= xmax) & (y_circle >= ymin) & (y_circle <= ymax)

                    if not np.any(mask):
                        continue

                    ax.plot(x_circle[mask], y_circle[mask],
                        linestyle='dashed', color=colors[i % len(colors)], linewidth=2)

                    x_text = -q_val
                    y_text = q_val * (-0.8)**i
                    if x_text < xmin or x_text > xmax:
                        x_text = -x_text
                    if y_text < ymin or y_text > ymax:
                        y_text = -y_text

                    y_offset = 0.002
                    if q_circles_labels:
                        ax.text(x_text, y_text + y_offset, f"{q_val:.3f}",
                            color=colors[i % len(colors)], fontsize=10,
                            ha='center', va='bottom',
                            bbox=dict(facecolor='white', alpha=1, edgecolor='none', pad=1),
                            clip_on=True)
                    
            plt.tight_layout()
        
        

        # Sauvegarde de l'image
        if output_dir is None:
            output_dir = self.path + '/saxs_images/'
        else:
            output_dir += '/saxs_images/'
        os.makedirs(output_dir, exist_ok=True)

        figname = os.path.join(output_dir, f'{self.samplename}_Img_{self.file_number}')
        if q_circles is not None:
            figname += '_with_q-circles'
        figname += '.png'

        fig.savefig(figname, dpi=200, bbox_inches='tight')
        print(f'✅ SAXS image saved: {figname}')
        # Affichage dans le notebook
        plt.show()
        
        

        # Sauvegarde de l'image
        if output_dir is None:
            output_dir = self.path + '/saxs_images/'
        else:
            output_dir += '/saxs_images/'
        os.makedirs(output_dir, exist_ok=True)

        figname = os.path.join(output_dir, f'{self.samplename}_Img_{self.file_number}')
        if q_circles is not None:
            figname += '_with_q-circles'
        figname += '.png'

        fig.savefig(figname, dpi=200, bbox_inches='tight')
        print(f'✅ SAXS image saved: {figname}')
        # Affichage dans le notebook
        plt.show()
        
    
    
    

    def extract_azimuthal_profile(self, qvalue, threshold = 0.0001, save=True,output_dir=None,apply_mirror=False):
        """
        Extract the azimuthal intensity profile at a given Q value.
        
        If mapping=True, extracts intensity as a function of beta angle.
        If mapping=False, uses pyFAI radial integration.

        Parameters
        ----------
        qvalue : float
            Target Q value for the azimuthal integration (Å⁻¹).
        threshold : float, optional
            Relative half-width of the radial integration window around `qvalue`
            (default: 1e-4).
        save : bool, optional
            If True, save the azimuthal profile to a text file (default: True).
        output_dir : str or None, optional
            Base directory where the profile will be saved. If None, the profile
            is saved in `<self.path>/azimuthal_profiles/`.
        apply_mirror : bool, optional
            Apply mirroring to the profile (only for mapping=False)

        Returns
        -------
        angle : ndarray
            Azimuthal angles (beta if mapping=True, chi if mapping=False) in degrees.
        I : ndarray
            Intensity values.
        """
        # Apply caving if mask provided
        if self.mask is not None:
            if self.mask.split('.')[-1] == 'edf':  
                self.apply_caving()
            else:
                print('Please provide a mask file edited with pyFAI and saved with ".edf" file extension')
        
        # Use mapping-based extraction if available
        if self.mapping:
            # Extract pixels at constant Q using beta angle (similar to WAXSProcessor)
            # Convert qvalue from Å⁻¹ to m⁻¹
            qvalue_m = qvalue * 1e10
            
            # Find pixels within threshold of qvalue
            q_min = qvalue_m * (1 - threshold)
            q_max = qvalue_m * (1 + threshold)
            
            # Use norm_Q to find pixels at constant Q
            mask = (self.norm_Q >= q_min) & (self.norm_Q <= q_max)
            
            # Exclude masked pixels if mask is available
            if hasattr(self, 'maskdata'):
                mask = mask & (self.maskdata != 1)
            
            beta = []
            data = []
            
            # Extract beta and intensity for pixels at constant Q
            for i in range(self.data.shape[0]):
                for j in range(self.data.shape[1]):
                    if mask[i, j]:
                        beta.append(self.beta[i, j])
                        data.append(self.data[i, j])
            
            # Sort by beta angle
            if len(beta) > 0:
                results = list(zip(beta, data))
                results = sorted(results)
                # Remove zero intensity (detector gaps)
                results = [(b, d) for b, d in results if d != 0]
                
                if results:
                    beta, data = zip(*results)
                    beta = np.array(beta)
                    data = np.array(data)
                else:
                    beta = np.array([])
                    data = np.array([])
            else:
                beta = np.array([])
                data = np.array([])
            
            # Save profile
            if save and len(beta) > 0:
                if output_dir is None:
                    output_dir = os.path.join(self.path, 'azimuthal_profiles')
                else:
                    output_dir = os.path.join(output_dir, 'azimuthal_profiles')
                
                os.makedirs(output_dir, exist_ok=True)
                
                try:
                    output = os.path.join(output_dir, 
                                f'{self.samplename}_{self.B}mT_q={qvalue:.3f}_beta_Img{self.file_number:05d}_x={float(self.x):.2f}_z={float(self.z):.2f}.dat')
                except:
                    output = os.path.join(output_dir, 
                                f'{self.samplename}_{self.B}mT_q={qvalue:.3f}_beta_Img{self.file_number:05d}.dat')
                
                np.savetxt(output, np.column_stack([beta, data]), 
                          header='beta(degrees) Intensity', comments='')
            
            return beta, data
        
        # Original pyFAI-based method
        chi, I = self.ai.integrate_radial(
            self.data, 
            540,
            mask=self.maskdata, 
            radial_range=(qvalue * (1 - threshold), qvalue * (1 + threshold)), 
            radial_unit="q_A^-1",
            method=("no", "histogram", "cython")
        )
        if apply_mirror:
            chi, I = self.mirror_profile(chi, I, center=180)
        
        if save:
            if output_dir is None:
                output_dir = os.path.join(self.path, 'azimuthal_profiles')
            else:
                output_dir += '/azimuthal_profiles/'
                           
            os.makedirs(output_dir, exist_ok=True) 
            try:
                output = os.path.join(output_dir, 
                                 f'{self.samplename}_{self.B}mT_q={float(qvalue):.3f}_Img{self.file_number:05d}_x={float(self.x):.2f}_z={float(self.z):.2f}.dat')
            except:
                output = os.path.join(output_dir, 
                                 f'{self.samplename}_{self.B}mT_q={float(qvalue):.3f}_Img{self.file_number:05d}.dat')
            np.savetxt(output, np.column_stack([chi, I]))
            
        return chi, I
    
    @staticmethod
    def mirror_profile(theta_exp, I_exp, center=180):
        """
        Génère un profil azimutal symétrique sans interpolation,
        angles dans [-180°, 180°].
        """
        theta_exp = np.array(theta_exp)
        I_exp = np.array(I_exp)
        """
        # --- Détecter le pic principal ---
        peaks, _ = find_peaks(I_exp, height=0)
        if len(peaks) == 0:
            raise ValueError("Aucun pic détecté.")
        
        main_peak = peaks[np.argmax(I_exp[peaks])]
        x0 = theta_exp[main_peak]
        """
        # --- Calculer les angles miroir ---
        theta_mirror_array = (2*center - theta_exp) % 360
        I_mirror_array = I_exp.copy()  # intensité identique
        
        # --- Fusionner ---
        theta_aug = np.concatenate([theta_exp, theta_mirror_array])
        I_aug = np.concatenate([I_exp, I_mirror_array])
        
        # --- Mapper dans [-180, 180] et trier ---
        theta_aug = ((theta_aug + 180) % 360) - 180
        sort_idx = np.argsort(theta_aug)
        
        return theta_aug[sort_idx], I_aug[sort_idx]

    def extract_radial_profile(self, azimuth: float = 90, width: float = 10, save=True, output_dir=None):
        """
        Extract radial profile in angular sector.
        
        Parameters:
        -----------
        azimuth : float
            Central azimuthal angle (°)
        width : float
            Angular sector full width (°)
        save : bool
            Save profile to file
        outputdir : str or None
            Directory to save profile
            
        Returns:
        --------
        q : array
            Q values (Å⁻¹)
        I : array
            Intensity values
        """
        azimuth_rad = math.radians(azimuth)
        azimuth_rad = azimuth_rad % (2 * math.pi)
        if azimuth_rad > math.pi:
            azimuth_rad -= 2 * math.pi
        if azimuth_rad < -math.pi:
            azimuth_rad += 2 * math.pi
        azimuth = math.degrees(azimuth_rad)
                
        min_az = max(azimuth - width/2, -180)
        max_az = min(azimuth + width/2, 180)
        
        q, I = self.ai.integrate1d(
            self.data, 
            1000,
            azimuth_range=(min_az, max_az),
            mask=self.maskdata, 
            unit="q_A^-1", 
            normalization_factor=1,
            method="csr"
        )
        
        if save:
            if output_dir is None:
                output_dir = os.path.join(self.path, 'radial_profiles')
            else:
                output_dir += '/radial_profiles/'
            os.makedirs(output_dir, exist_ok=True)
            try:
                output = os.path.join(output_dir,
                                 f'{self.samplename}_{self.B}mT_azimuth={int(azimuth)}_width={width}_Img{self.file_number:05d}_x={self.x:.2f}_z={self.z:.2f}.dat')
            except:
                output = os.path.join(output_dir,
                                 f'{self.samplename}_{self.B}mT_azimuth={int(azimuth)}_width={width}_Img{self.file_number:05d}.dat')
            np.savetxt(output, np.column_stack([q, I]))
            
        return q, I
    

    def find_main_orientation(self, qvalue=0.01, threshold=0.05):
        """
        Determine the main orientation angle from the azimuthal intensity profile.

        The azimuthal profile at a given Q value is smoothed and the angle
        corresponding to the maximum intensity is returned.

        Parameters
        ----------
        qvalue : float, optional
            Q value used for the azimuthal profile extraction.
        threshold : float, optional
            Relative width of the radial integration window around `qvalue`.

        Returns
        -------
        angle : float
            Azimuthal angle (degrees) corresponding to the maximum intensity.
        """
        chi, I = self.extract_azimuthal_profile(qvalue=qvalue, threshold=threshold)
        # apply savgol filter to I and remove zeros
        I = savgol_filter(I,9,2)
        # filtrer les points à intensité nulle
        mask = (I > 0) & (chi > -90) & (chi < 90) 
        chi = chi[mask]; I = I[mask]
        angle = chi[np.argmax(I)]
        return angle
        #return min (angle, 180 + angle)
        