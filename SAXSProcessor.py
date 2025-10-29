
import h5py
import hdf5plugin
import numpy as np
import os
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree
from ase.io import read
from skimage.measure import block_reduce
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
from scipy.integrate import quad
import glob
import pandas as pd
import ast
import re
import fabio
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
import pyFAI, pyFAI.detectors
import math
from pathlib import Path
from matplotlib.colors import LogNorm

class EdfFile:
    def __init__(self, file):
        """
        This class is designed to handle files in edf formats.
        It parses file metadata into comprehensible variables for further use in SAXSDataProcessor class.
        file: path to file
        """
        self.file = file
        
        # check if data corresponds to lineEraser case
        if 'vd' in self.file.split('_'):
            self.lineEraser=True
            # Redefine x and y centers from individual images
            file1,file2=self.getindividualfiles_lineEraser(self.file)
            im1=fabio.open(file1)
            header1=im1.header
            im2=fabio.open(file2)
            header2=im2.header
            
            self.x_center=float(header1['Center_1'])
            date = header1['Date']
            annee = int(date[:4])
            if annee<2025: # Upgrade soft
                print('self.z_center=moyenne')
                self.z_center=(float(header2['Center_2'])+float(header1['Center_2']))/2
            else:
                self.z_center=float(header2['Center_2'])
                print('self.z_center=header2')
            #self.z_center=(float(header2['Center_2'])+float(header1['Center_2']))/2
        else:
            self.lineEraser=False
              
        image = fabio.open(self.file)
        header = image.header
        
        self.data = image.data
        
        shape=self.data.shape
        self.num_pixel_x = shape[0]
        self.num_pixel_z = shape[1]
               
        # Experimental details
        self.wl = float(header['WaveLength'])
        
        if not self.lineEraser:
            self.x_center = float(header['Center_1'])
            self.z_center = float(header['Center_2'])
        self.pixel_size_x = float(header['PSize_1'])
        self.pixel_size_z = float(header['PSize_2'])
        self.D = float(header['SampleDistance'])
        self.samplename = header['Comment']
        self.nb_frames=1
        self.file_number=int(file.split('/')[-1].split('.')[0].split('_')[-1].split('-')[0])
        

        self.bin_x=1
        self.bin_y=1
        self.B=self.extract_B_value()
        
        
    
    def getindividualfiles_lineEraser(self,file):
        directory=os.path.dirname(file)
        filename=file.split('/')[-1].split('.')[0]
        prefix=filename.split('_')[0]+'_0'
        filenumbers=filename.split('_')[3]
        file1=f'{directory}/{prefix}_{int(filenumbers.split("-")[0]):05d}.edf'
        file2=f'{directory}/{prefix}_{int(filenumbers.split("-")[1]):05d}.edf'
        return file1,file2
    
    def extract_B_value(self):
        match = re.search(r'(\d+)\s*mT', self.file)
        if match:
            return int(match.group(1))  
        return 0


class h5File_ID02:
    def __init__(self,file):
        """
        This class is designed to handle files from ESRF-ID02 beamline.
        It parses file metadata into comprehensible variables for further use in SAXSDataProcessor class."""
        self.file=file
        self.file_number=self.extract_number()
               
        if file is None:
            print("Please specify a data file path")
        if "_eiger2_" in file:
            with h5py.File(file, "r") as f:
                group = list(f.keys())[0]  # Retrieve first group key
                self.title=str(f[group+'/instrument/id02-eiger2-saxs/header/Title'][()].decode('utf-8'))
                self.nb_frames = int(f[group + '/instrument/id02-eiger2-saxs/acquisition/nb_frames'][()])
                self.acq_time = float(f[group + '/instrument/id02-eiger2-saxs/acquisition/exposure_time'][()])
                
                # Retrieve image data
                target = group + '/measurement/data'
                self.data = np.array(f[target])
                shape = np.shape(self.data)
                if len(shape)==2:
                    self.num_pixel_x = shape[0]
                    self.num_pixel_z = shape[1]
                elif len(shape)==3:
                    self.num_pixel_x=shape[1]
                    self.num_pixel_z=shape[2]
                else:
                    print(f"Data in file {self.file} should have 2 or 3 dimensions")
                
                # Retrieve header information
                header = '/entry_0000/instrument/id02-eiger2-saxs/header'
                self.pixel_size_x = float(f[header + '/PSize_1'][()].decode('utf-8'))
                self.pixel_size_z = float(f[header + '/PSize_2'][()].decode('utf-8'))
                self.wl = float(f[header + '/WaveLength'][()])
                self.x_center = float(f[header + '/Center_1'][()].decode('utf-8'))
                self.z_center = float(f[header + '/Center_2'][()].decode('utf-8'))
                self.D = float(f[header + '/SampleDistance'][()])

                # Retrieve binning info
                header = '/entry_0000/instrument/id02-eiger2-saxs/image_operation/binning'
                self.bin_x = f[header + '/x'][()]
                self.bin_y = f[header + '/y'][()]
        self.samplename=self.extract_sample_name()
        self.B=self.extract_magnetic_field()

    def extract_magnetic_field(self):
        # Match patterns like '100mT', '1T', '1.4T', etc.
        match = re.search(r'(\d+(\.\d+)?)(mT|T)', self.title)
        
        if match:
            value, _, unit = match.groups()
            value = float(value)
            
            # Convert T to mT
            if unit == 'T':
                value *= 1000

            return int(value)
        else:
            # If no magnetic field is found, return '0mT'
            return 0
    

    def extract_sample_name(self):
        """Extracts the sample name from a string before the magnetic field value."""
        pattern = re.compile(r"^(.*?)(?:_\d+(?:\.\d+)?(?:mT|T).*)$")
        match = pattern.match(self.title)
        if match:
            return match.group(1)
        else:
            return self.title
        
    def extract_number(self):
        """
        Extract the number from the filename.
        
        Args:
            file_path (str): Path to the file.
        
        Returns:
            int: Extracted number from the filename.
        """
        # Split the filename to get the number
        filename = self.file.split('/')[-1]  
        number = filename.split('_')[2]       
        return int(number)



class h5File_SWING:
    def __init__(self,file: str, mean=True):
        """ 
        This class is designed to handle files from SOLEIL-SWING beamline.
        It parses file metadata into comprehensible variables for further use in SAXSDataProcessor class.
        filename: str path to data faile (*.h5 or *.nxs)
        mean: bool 
            Average frames if True
        
        """
        self.file=file
        self.file_number=self.extract_number()
               
        if not os.path.exists(file):
            print(f"File {file} not found.")
        # Extract parameters and metadata from the HDF5 file; results stored in self.params dictionary.
        self.extract_from_h5()
        # compute intensity taking into accoun average and anisotropic tags
        self.mean = mean
        self.eiger=self.extract_scatteringdata()
        
        self.B=1000
        
     
            
    def extract_number(self):
        number=self.file.split('/')[-1].split('_')[1]
        return int(number)

    def extract_from_h5(self):
        """Extract data from the HDF5 file and return a dictionary of parameters."""
        with h5py.File(self.file, "r") as f:
            # Retrieve sample name from the first group in the file.
            group = list(f.keys())[0]
            self.samplename = f[group+'/sample_info/ChemSAXS/sample_name'][()].decode('utf-8')
                       
            # Retrieve experimental parameters from the Eiger-4M group
            target = group + '/SWING/EIGER-4M'
            self.D = f[target + '/distance'][0] / 1000  # Convert distance to meters
            self.pixel_size_x = f[target + '/pixel_size_x'][0] * 1e-6  # Convert pixel size to meters
            self.pixel_size_z = f[target + '/pixel_size_z'][0] * 1e-6
            self.x_center = f[target + '/dir_beam_x'][0]
            self.z_center = f[target + '/dir_beam_z'][0]
            self.nb_frames = f[target + '/nb_frames'][0]
            self.bin_x = f[target + '/binning_x'][0]
            self.bin_y = f[target + '/binning_y'][0]
            self.acq_time = f[target + '/exposure_time'][0]
                        
            # Retrieve monochromator (wavelength) information
            target = group + '/SWING/i11-c-c03__op__mono'
            self.wl = f[target + '/wavelength'][0]*1e-10
                        
            self.folder = os.path.dirname(self.file)

            # Retrieve the Eiger SAXS data (2D detector images)
            target = group + '/scan_data/eiger_image'
            self.data = np.array(f[target])
        return 

    def extract_scatteringdata(self):

        with h5py.File(self.file, "r") as f:
            # Retrieve sample name from the first group in the file.
            group = list(f.keys())[0]
            # Retrieve the Eiger SAXS data (2D detector images)
            target = group + '/scan_data/eiger_image'
            eiger_raw = np.array(f[target])
            
            self.Dim_1=eiger_raw.shape[1];self.Dim_2=eiger_raw.shape[2]
            # Sometimes the shape is (n_images, 1, height, width); if so, squeeze out the extra dimension.
            if eiger_raw.shape[1] == 1:
                eiger_raw = eiger_raw.squeeze(axis=1)
        if self.mean:
            # Replace data with average data when averageframes=True
            eiger = np.expand_dims(np.mean(eiger_raw, axis=0),axis=0)
            
            
        # Case of no averaging
        else:            
            eiger=eiger_raw
        self.num_pixel_x=eiger.shape[2]
        self.num_pixel_z=eiger.shape[1]
        return eiger

    def convert2edf(self,outputdir=None):
        filelist =[]
        for i in range(self.nb_frames):
            data2save = self.eiger[i]
            header = {
                "WaveLength": str(self.wl),
                "Center_1": str(self.x_center),
                "Center_2": str(self.z_center),
                "PSize_1": str(self.pixel_size_x),
                "PSize_2": str(self.pixel_size_z),
                "SampleDistance": str(self.D),
                "Comment": str(self.samplename) } 
            img = fabio.edfimage.edfimage(data=data2save,header=header)
            if outputdir is None:
                outputdir=self.folder
            else:
                outputdir=outputdir
                os.makedirs(outputdir,exist_ok=True)
            filename = os.path.join(outputdir,f'{self.samplename}_File_{self.file_number}_Img_{i}.edf')
            img.write(filename)
            filelist.append(filename)
        return filelist





class SAXSDataProcessor:
    def __init__(self,
                 file: str = None,
                 instrument='ID02',
                 qvalues:np.ndarray = [0.034,0.068],
                 threshold: float = 0.0001,
                 binning: int = 1,
                 mask=None):
        """
        Initialize the Mapping class.
        
        Args:
            file (str): Path to the .h5 file.
            instrument: 'ID02', 'SWING' or 'LGC'
            qvalues (np.ndarray): array containing q values at which azimuthal profiles should be extracted (saxs data)
            qpeakrange: [qmin,qmax] for peak on radial profiles
            threshold (float): Relative tolerance for q values.
            binning (int): Factor for downsampling the image data.
            mask: path to mask file for pyFAI method (optional)
        """
        self.filepath=file
        self.path=file.split('/')[:-1]
        self.path='/' + '/'.join(self.path[1:])
        #print('self.path=', self.path)
        if instrument=='ID02':
            self.file=h5File_ID02(file)
        elif instrument=='LGC':
            self.file=EdfFile(file)
        elif instrument == 'SWING':
            self.file= h5File_SWING(file)

        self.file_number=self.file.file_number
        self.threshold = threshold
        self.binning = binning
        self.data=self.file.data
        
        if file is None:
            print("Please specify a data file path")
        
        self.qvalues=qvalues
        
                
        # Load data from file (either LGC, WAXS or EIGER2 detector)
        self.nb_frames=self.file.nb_frames
        #self.acq_time=self.file.acq_time
        
        self.num_pixel_x=self.file.num_pixel_x
        self.num_pixel_z=self.file.num_pixel_z
        self.pixel_size_x=self.file.pixel_size_x
        self.pixel_size_z=self.file.pixel_size_z
        self.wl=self.file.wl
        self.x_center=self.file.x_center
        self.z_center=self.file.z_center
        self.D=self.file.D
        self.bin_x=self.file.bin_x
        self.bin_y=self.file.bin_y
        self.samplename=self.file.samplename.replace(' ','_')
        self.B=self.file.B 
       
             
        # Apply binning if needed
        if self.binning != 1:
            print('Binning applied')
            """
            self.x_center /= self.binning
            self.z_center /= self.binning
            self.pixel_size_x *= self.binning
            self.pixel_size_z *= self.binning
            self.num_pixel_x //= self.binning
            self.num_pixel_z //= self.binning
            """
            self.pixel_size_x *= self.binning
            self.pixel_size_z *= self.binning
            
            # Downsample the image
            #self.data = self.downsample_image()
            
        # Average data if needed
        if len(self.data.shape)==3:
            self.data=np.mean(self.data,axis=0)
                      
        # Apply median filter for noise reduction
        #self.data = median_filter(self.data, size=self.binning)
        # assing maskdata
        if mask is not None:
            if mask.split('.')[-1]=='edf':
                maskimage=fabio.open(mask)
                self.mask=mask
                self.maskdata=maskimage.data
                #if binning !=1:
                #    self.maskdata=self.bin_mask()
            else:
                print('Mask Format Error: Mask files should be provided in edf format')
        else:#no mask
            self.maskdata=np.zeros_like(self.data)

        
    def bin_mask(self):
        h, w = self.maskdata.shape
        bin_factor=self.binning
        # Ensure mask shape is divisible by bin_factor
        h_binned = h // bin_factor
        w_binned = w // bin_factor
        mask = self.maskdata[:h_binned * bin_factor, :w_binned * bin_factor]  # crop to divisible size

        # Reshape and apply max to propagate masked pixels
        reshaped = mask.reshape(h_binned, bin_factor, w_binned, bin_factor)
        binned_mask = reshaped.max(axis=(1, 3))  
        return binned_mask
      

       
    def plot2d_vsq(
        self, 
        q_range=[0,0.2],
        cmap='jet',
        log=True,
        grid_size=1000,
        vmin=-4, vmax=0,
        normalize=True,
        caving=True,
        q_circles = None,
        outputdir = None,
        rotate90=False):
        """
        Plot a 2D SAXS intensity map (qx, qz).

        Parameters
        ----------
        q_range : tuple (qmin, qmax), optional
            |q| range to display. If None, all data is used.
        cmap : str
            Colormap.
        log : bool
            If True, logarithmic color scale.
        grid_size : int
            Number of points in the interpolated grid along each axis.
        vmin, vmax : float, optional
            Colorbar bounds.
        normalize : bool
            If True, normalize intensity to [0,1].
        caving : bool
            If True, apply caving correction.
        q_circles: tuple
            q values to plot as circle on 
        """

        # Apply caving if requested
        if caving:
            self.caving2()
        data = self.data

        # Compute qx, qy, qz
        qx = np.zeros(data.shape, dtype=float)
        qy = np.zeros(data.shape, dtype=float)
        qz = np.zeros(data.shape, dtype=float)
        for i in range(self.num_pixel_z):
            delta_i = (i - self.x_center) * self.pixel_size_x
            for j in range(self.num_pixel_x):
                delta_j = (j - self.z_center) * self.pixel_size_z
                denom = np.sqrt(self.D**2 + delta_i**2 + delta_j**2)
                a = 2 * np.pi / self.wl
                qx[j, i] = (a * delta_i / denom) * 1e-10
                qy[j, i] = (a / denom) * (self.D - denom) * 1e-10
                qz[j, i] = (a * delta_j / denom) * 1e-10

        # Compute |q| and mask
        qnorm = np.sqrt(qx**2 + qy**2 + qz**2)
        if q_range is not None:
            qmin, qmax = q_range
            #mask = (qnorm >= qmin) & (qnorm <= qmax)
            mask=(qx >= -qmax) & (qx <= qmax) & (qz >= -qmax) & (qz <= qmax)
            qx_masked, qz_masked, intensity = qx[mask], qz[mask], data[mask]
        else:
            qx_masked, qz_masked, intensity = qx.flatten(), qz.flatten(), data.flatten()

        # Normalize intensity
        if normalize:
            intensity = intensity / np.nanmax(intensity)

        # Interpolate onto regular grid
        if len(qx_masked) < 4:
            # Trop peu de points pour interpolation linéaire => scatter plot
            plt.figure(figsize=(6,6))
            sc = plt.scatter(qx_masked, qz_masked, c=intensity, cmap=cmap,
                            vmin=vmin, vmax=vmax)
            plt.xlabel("qx (Å⁻¹)")
            plt.ylabel("qz (Å⁻¹)")
            plt.colorbar(sc, label="Normalized Intensity" if normalize else "Intensity (a.u.)")
            plt.gca().set_aspect('equal')
            plt.show()
            return

        qx_lin = np.linspace(qx_masked.min(), qx_masked.max(), grid_size)
        qz_lin = np.linspace(qz_masked.min(), qz_masked.max(), grid_size)
        QX, QZ = np.meshgrid(qx_lin, qz_lin)
        Q = np.sqrt(QX**2+QZ**2)
        Z = griddata((qx_masked, qz_masked), intensity, (QX, QZ), method='linear')
        # ---- rotation si demandé ----
        if rotate90:
            print('Rotation applied')
            Z = np.rot90(Z, k=-1)  # -1 pour sens horaire
            QZ, QX = np.rot90(QX, k=-1), np.rot90(QZ, k=-1)

        # Plot
        plt.figure(dpi=200)
        norm = LogNorm(vmin=10**vmin, vmax=10**vmax) if log else None
        mesh = plt.pcolormesh(QX, -QZ, Z, shading='auto', cmap=cmap, norm=norm)
        plt.xlabel(r"$Q_x (\AA^{-1})$",fontsize = 12)
        plt.ylabel(r"$Q_z (\AA^{-1})$",fontsize = 12)
        cbar = plt.colorbar(mesh,shrink = 0.5, aspect= 20)
        cbar.set_label("Normalized Intensity" if normalize else "Intensity (a.u.)", fontsize=12)
        cbar.ax.tick_params(labelsize=12)  # taille des chiffres sur la barre
        plt.gca().set_aspect('equal')
        
        # ---- Ajouter les cercles de q constants (partie visible seulement) ----
        if q_circles is not None:
            colors = ['black','purple','pink','palegreen']
            ax = plt.gca()

            # prendre les limites effectives de l'axe (après pcolormesh)
            xmin, xmax = sorted(ax.get_xlim())
            ymin, ymax = sorted(ax.get_ylim())

            for i, q_val in enumerate(q_circles):
                theta = np.linspace(0, 2*np.pi, 2000)      # haute résolution pour éviter les trous
                x_circle = q_val * np.cos(theta)
                y_circle = q_val * np.sin(theta)

                # adapter l'ordonnée au système de l'affichage (on a tracé -QZ)
                y_plot = -y_circle

                # masque 1D : garder uniquement les points du cercle qui tombent dans la fenêtre affichée
                mask = (x_circle >= xmin) & (x_circle <= xmax) & (y_plot >= ymin) & (y_plot <= ymax)

                # si aucune portion visible, on saute
                if not np.any(mask):
                    continue

                # tracer la portion visible
                ax.plot(x_circle[mask], y_plot[mask],
                        linestyle='dashed', color=colors[i % len(colors)], linewidth=2)

                
                x_text = -q_val
                y_text = q_val*(-1)**i
                if x_text<xmin or x_text>xmax:
                    x_text = -x_text
                if y_text<ymin or y_text>ymax:
                    y_text=-y_text

                # petit décalage pour que le texte ne chevauche pas le trait (décalage relatif à la hauteur de l'axe)
                y_offset = 0.002
                
                ax.text(x_text, y_text + y_offset, f"{q_val:.3f}",
                        color=colors[i % len(colors)], fontsize=10,
                        ha='center', va='bottom',
                        bbox=dict(facecolor='white', alpha=1, edgecolor='none', pad=1),
                        clip_on=True)
        plt.tight_layout()

        if outputdir == None:
            outputdir = self.folder
        else:
            os.makedirs(outputdir,exist_ok=True)

        figname = outputdir + f'{self.samplename}_Img_{self.file_number}.png'
        if q_circles is not None:
            figname = outputdir + f'{self.samplename}_Img_{self.file_number}_with_q-circles.png'

        plt.savefig(figname)
        print(f'SAXS image generated in {figname}')
        plt.close()
        
        
    
    def downsample_image(self):
        """Downsample the image using block averaging with crop to match mask."""
        h, w = self.data.shape
        bin_factor = self.binning
        h_binned = h // bin_factor
        w_binned = w // bin_factor
        
        # Crop pour que les dimensions soient divisibles par bin_factor
        img_cropped = self.data[:h_binned*bin_factor, :w_binned*bin_factor]
        
        # Downsample
        return block_reduce(img_cropped, block_size=(bin_factor, bin_factor), func=np.mean)
    
    def pyFAI_extract_azimprofiles(self,qvalue):
        # set pyFAI detector instance
        detector = pyFAI.detectors.Detector(pixel1=self.pixel_size_x, pixel2=self.pixel_size_z)
        ai = AzimuthalIntegrator(dist=self.D, detector=detector)
        # extract azimuthal profile at given q value using integrate_radial method from AzimuthalIntegrator instance
        ai.setFit2D(self.D*1000,self.x_center,self.z_center,wavelength=self.wl*1e10)
        outputdir=self.path+'/azimuthal_profiles'    
        os.makedirs(outputdir,exist_ok=True) 
        output= f'{outputdir}/{self.samplename}_{self.B}mT_q={float(qvalue):.3f}_Img{self.file_number:05d}.dat'
        chi,I=ai.integrate_radial(self.data, 540,mask=self.maskdata, radial_range=(qvalue*(1-self.threshold), qvalue*(1+self.threshold)), radial_unit="q_A^-1",method=("no", "histogram", "cython"))
        np.savetxt(output,np.column_stack([chi,I]))
        return chi,I

       
    def caving(self):
        # Remplacer directement les pixels masqués par NaN
        self.data = np.where(self.maskdata == 1.0, np.nan, self.data)

        # Ici : première dimension = num_pixel_x, deuxième = num_pixel_z
        for x in range(int(self.num_pixel_x)):      # lignes (axe 0)
            for z in range(int(self.num_pixel_z)):  # colonnes (axe 1)
                if np.isnan(self.data[x, z]):       # pixel masqué
                    xsym = int(2 * self.z_center - x)  
                    zsym = int(2 * self.x_center - z)

                    # Vérifier les bornes dans TON système
                    if 0 <= xsym < int(self.num_pixel_x) and 0 <= zsym < int(self.num_pixel_z):
                        if not np.isnan(self.data[xsym, zsym]):  # pixel symétrique valide
                            #print(f"Modifying pixel ({x},{z}) from NaN to {self.data[xsym, zsym]}")
                            self.data[x, z] = self.data[xsym, zsym]
                            self.maskdata[x, z] = 0  # optionnel
    
    def caving2(self, max_iter=10):
        """
        Remplace les pixels masqués par la valeur symétrique
        (par rapport à x_center, z_center) en plusieurs passes.

        max_iter : nombre maximal d'itérations
        """

        # Remplacer directement les pixels masqués par NaN
        self.data = np.where(self.maskdata == 1.0, np.nan, self.data)

        for it in range(max_iter):
            modified = False
            for x in range(int(self.num_pixel_x)):      # lignes
                for z in range(int(self.num_pixel_z)):  # colonnes
                    if np.isnan(self.data[x, z]):       # pixel masqué
                        xsym = int(2 * self.z_center - x)
                        zsym = int(2 * self.x_center - z)

                        # Vérifier les bornes
                        if 0 <= xsym < int(self.num_pixel_x) and 0 <= zsym < int(self.num_pixel_z):
                            if not np.isnan(self.data[xsym, zsym]):  # pixel symétrique valide
                                self.data[x, z] = self.data[xsym, zsym]
                                self.maskdata[x, z] = 0
                                modified = True

            # Si aucune modification -> arrêter
            if not modified:
                break
        #print(f"Caving terminé en {it+1} itérations")

    def pyFAI_extract_radialprofiles(self,azimuth:float=90,width:float=40,caving=False,remove=True):
        """
        azimuth: angular value (in °) of azimuthal direction
        width: full width of angular sector (default=40)
        """
        if caving:
            self.caving()
        azimuth=math.radians(azimuth)
        azimuth = azimuth % (2 * math.pi)  # Normalize to [0, 2π)
        if azimuth > math.pi:
            azimuth -= 2 * math.pi  # Adjust to [-π, π)
        if azimuth < -math.pi:
            azimuth += 2 * math.pi  # Adjust to [-π, π)
        azimuth=math.degrees(azimuth)
                
        min_az=azimuth-width/2; max_az=azimuth+width/2
        if min_az<-180:
            min_az=-180
        if max_az>180:
            max_az=180
        detector = pyFAI.detectors.Detector(pixel1=self.pixel_size_x, pixel2=self.pixel_size_z)
        ai = AzimuthalIntegrator(dist=self.D, 
                                detector=detector)#,
                                #poni1=self.z_center * self.pixel_size_z,
                                #poni2=self.x_center * self.pixel_size_x,
                                #wavelength=self.wl)
        # extract radial profile
        ai.setFit2D(self.D*1000,self.x_center,self.z_center,wavelength=self.wl*1e10)  
        #  specify output name
        outputdir=self.path+'/radial_profiles'    
        os.makedirs(outputdir,exist_ok=True) 
        output= f'{outputdir}/{self.samplename}_{self.B}mT_azimuth={int(azimuth)}_width={width}_Img{self.file_number:05d}.dat'
        q, I = ai.integrate1d(self.data, 1000,filename=output, azimuth_range=(min_az,max_az),mask=self.maskdata, unit="q_A^-1", normalization_factor=1,method="csr")
        return q,I

    def detect_all_peaks_by_second_derivative(self, q, I,
                                          nb_peaks = 1,
                                          window_length=15,
                                          polyorder=3,
                                          prominence=0.5,
                                          distance_pts=20,
                                          q_range=None,
                                          plot=False):
        """
        Détecte plusieurs pics dans I(q) via la dérivée seconde (méthode de la courbure).

        Paramètres
        ----------
        q : ndarray
            Vecteur des q.
        I : ndarray
            Intensité I(q).
        nb_peaks: int
            Number of peaks to detect
        window_length : int
            Fenêtre du filtre Savitzky-Golay (doit être impair).
        polyorder : int
            Ordre du polynôme utilisé pour le filtre.
        prominence : float
            Profondeur minimale des pics dans la dérivée seconde (sensibilité).
        distance_pts : int
            Distance minimale (en points) entre deux pics.
        q_range : tuple or None
            Limite optionnelle (q_min, q_max) pour restreindre la détection.

        Retourne
        --------
        q_peaks : ndarray
            Tableau des positions q des pics détectés.
        """

        if window_length % 2 == 0:
            window_length += 1
        delta_q = q[1] - q[0]

        # Lissage + dérivée seconde
        d2I = savgol_filter(I, window_length=window_length, polyorder=polyorder, deriv=2, delta=delta_q)

        # Inverser la dérivée seconde : on cherche les minima => devient max
        inverted_d2I = -d2I

        # Restreindre la recherche si demandé
        mask = np.ones_like(q, dtype=bool)
        if q_range:
            mask &= (q >= q_range[0]) & (q <= q_range[1])

        # Détection des pics (minima de la dérivée seconde)
        #peaks, _ = find_peaks(inverted_d2I[mask],
        #                    prominence=prominence,
        #                    distance=distance_pts)
        peaks, properties = find_peaks(inverted_d2I[mask], prominence=prominence, distance=distance_pts)

        # Trier selon la proéminence décroissante
        sorted_indices = np.argsort(properties["prominences"])[::-1]
        top_peaks = peaks[sorted_indices[:nb_peaks]]
        q_detected = q[mask][top_peaks]
        if plot:
            plt.figure()
            # Affichage
            plt.loglog(q, I, label="I(q)")
            colors=['r','g','b','c','m','y']
            id=0
            for qp in q_detected[:(nb_peaks)]:
                plt.axvline(qp, color=colors[id], ls='--', label=f'd = {2*np.pi/qp:.4f}')
                id+=1
            plt.xlabel("q (Å⁻¹)")
            plt.ylabel("I(q)")
            plt.title("Détection multiple de pics par dérivée seconde")
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.show()
        return q_detected[:(nb_peaks)]
        
    def compute_correlation_distance(self,nb_peaks=1,azimuth:float=90,width:float=40,caving=False,plot=False):
        if caving:
            self.caving()
        q,I=self.pyFAI_extract_radialprofiles(azimuth=azimuth,width=width,caving=caving)
        q_peaks=self.detect_all_peaks_by_second_derivative(q,I,nb_peaks=nb_peaks,plot=plot)
        return 2*np.pi/q_peaks       

 
    
    def plot_azim_profiles(self):
        
        nb_plots=len(self.qvalues)
        fig,ax=plt.subplots(nb_plots)
        i=0
        for qvalue in self.qvalues:
            if nb_plots==1:
                subplt=ax
            else:
                subplt=ax[i]
            print('Plotting azimuthal profile for q=', qvalue)
            chi,I=self.pyFAI_extract_azimprofiles(qvalue)
            subplt.plot(chi,I,'.',label=f'Q value {qvalue:.2f}')
            subplt.set_xlabel('Azimuthal angle (°)',fontsize = 14)
            subplt.set_ylabel('Intensity',fontsize = 14)
            subplt.legend(fontsize = 14)
            i+=1
        plt.title(f"{self.samplename}")
        plt.legend(fontsize = 14)    
        plt.tight_layout()
        plt.show()

       
    
        
    def pseudo_voigt(self,x,y0,I, x0, gamma,eta,slope):
        pi=np.pi
        ln2=np.log(2)
        a=(2/gamma)*(ln2/pi)**(1/2)
        b=(4*ln2/(gamma**2))
        return y0+slope*x+I*(eta*(a*np.exp(-b*(x-x0)**2))+(1-eta)*((1/pi)*((gamma/2)/((x-x0)**2+(gamma/2)**2))))
    
    def myfunc(self,x,I, x0, gamma,eta):
        # f(beta)sin(beta)
        return self.pv_nobckgd(x, I, x0, gamma,eta)*np.sin(x)
    
    def P2_nobckgd(self,x,I, x0, gamma,eta):
        return ((1/2)*(3*np.cos(x)*np.cos(x)-1)*self.pv_nobckgd(x, I, x0, gamma,eta)*np.sin(x))
    
    def pv_nobckgd(self,x,I, x0, gamma,eta):
        pi=np.pi
        ln2=np.log(2)
        a=(2/gamma)*(ln2/pi)**(1/2)
        b=(4*ln2/(gamma**2))
        return I*(eta*(a*np.exp(-b*(x-x0)**2))+(1-eta)*((1/pi)*((gamma/2)/((x-x0)**2+(gamma/2)**2))))
     
    def calc_S(self,x,I,x0,gamma,eta):
        # calculate degree of order
        norm_factor , error = quad(self.myfunc,0,np.pi,args=(I,x0*np.pi/180,gamma*np.pi/180,eta))
        S , error = quad(self.P2_nobckgd,0,np.pi,args=(I,x0*np.pi/180,gamma*np.pi/180,eta))
        S/=norm_factor
        return S
    
    def compute_S(self):
        """
        performs fit of azimuthal profile for each reflection of a single image
        results are stored in a dictionnary {reflection:[y0,I,mean,sigma,aire]}
        """
        results={} 
        
        for qvalue in self.qvalues:  
            profile=self.pyFAI_extract_azimprofiles(qvalue=qvalue)
            
            # Smooth data using Savitzky-Golay filter
            x=profile[0];y=savgol_filter(profile[1],5,1)
            # find position of max and define fitting range with "width" variable
            index=np.argmax(y[x<150]) #index of max Intensity
            width=90
            a=x[index]-width/2; b=x[index]+width/2 
            
            # define fitting region in x
            test=np.argwhere((a<x)&(x<b))
            amin=test[0,0] #index where x is close to a
            amax=test[-1,0] # index where x is close to b
            #extract arrays corresponding to the fitting region
            x2fit=x[amin:amax+1]
            x2fit = np.asarray(x2fit)
            xmin=np.min(x2fit);xmax=np.max(x2fit)
            y2fit=y[test[:,0]] 
            y2fit=np.asarray(y2fit)
            # give a first estimation of the fitted parameters
            y0_guess=np.mean(y[amin:amin+5]) # flat horizonthal background
            x0_guess=x[index]
            I_guess=np.max(y)
            gamma_guess=5
            eta_guess=1
            slope_guess=0.001
            init_params=[y0_guess,I_guess,x0_guess,gamma_guess,eta_guess,slope_guess]
            
            # define bounds
            lb_G=[0,0,xmin,0,0,-np.inf] # sigma low bounds can be negative in the formula 
            ub_G=[np.inf,np.inf,xmax,np.inf,1,np.inf]
            bounds_G=(lb_G,ub_G) 
            
            # fit the parameters and extract sigmas

            try:
                params, _ =curve_fit(self.pseudo_voigt,x2fit,y2fit,p0=init_params,bounds=bounds_G,method='trf')
                y0=params[0]
                I=params[1]
                x0=params[2] 
                if x0<0 and x0 <-20:
                    x0_S = x0 + 180 # position must be between 0 and 180 (we only consider this range for S calculation)
                else:
                    x0_S = x0
                if x0_S >20:
                    x0_S -=90 # center position around 0 so that 0<S<1 
                gamma=params[3]
                eta=params[4]
                slope=params[5]
                # Calculate R² (coefficient of determination)
                residuals = y2fit - self.pseudo_voigt(x2fit, *params)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((y2fit - np.mean(y2fit))**2)
                r_squared = 1 - (ss_res / ss_tot)

                # calculate surface areas with refined values
                aire, erreur = quad(self.pv_nobckgd, a, b, args=(I, x0, gamma,eta))
                fitflag=True

                # calculate degree of order
                S = self.calc_S(x,I,x0_S,gamma,eta)
            except Exception as e:
                print(f"Fit failed for q = {qvalue}: {e}")
                y0=np.nan
                I=np.nan
                mean=np.nan
                sigma=np.nan
                aire=np.nan
                S=np.nan
                r_squared=np.nan
                fitflag=False
            results[str(qvalue)]=[y0,I,x0,x0_S,gamma,eta,slope,S,r_squared] 
                   
        return results

    def loglogline(self,q,a,scale,exp):
        return a+scale*q**(-exp)

    def slope_determination(self,qmin=0.01,qmax=0.1):
        """
        qmin, qmx: boundaries btewwen which slope is determined
        """
        q,I=self.pyFAI_extract_radialprofiles()
        qfit=[];I=savgol_filter(I,11,2)
        for qi in q:
            if qi>qmin and qi<qmax:
                qfit.append(qi)
        indices,_=np.where(qi>qmin and qi<qmax )
        Ifit=I[indices]
        
        #guess value
        a_guess=0;scale_guess=1;exp_guess=2
        init_params=[a_guess,scale_guess,exp_guess]
        #define bounds for fit
        lb=[0,0,-4.5];ub=[np.inf,np.inf,0]
        bounds_G=(lb,ub)
        
        #perform fit
        params, _ =curve_fit(self.loglogline,qfit,Ifit,p0=init_params,method='trf')
        a,scale,exp=params
        
        return exp
    
    
class BatchSAXSDataProcessor():
    def __init__(self, path, instrument='ID02',azimqvalues=None,qpeakrange = [0.02,0.053],file_filter='*_eiger2*_raw.h5',threshold=0.0001,binning=2,mask=None):
        """
        path: str path to the directory containing h5 files
        instrument: 'ID02', 'SWING' or 'LGC'
        azimqvalues: list of qvalues for which azimuthal profiles are extracted
        qpeakrange: [qmin,qmax] for peak on radial profiles

        file_filter:str wildcard file filter (default=*_waxs*_raw.h5)
        threshold (float): Relative tolerance for q values.
        binning (int): Factor for downsampling the image data. 
        mask: path to maskfile (optional)
        """
        self.path=path
        self.instrument=instrument
        self.qvalues=azimqvalues
        self.h5_filelist=glob.glob(os.path.join(path,file_filter))
        self.h5_filelist=sorted(self.h5_filelist,key=self.extract_number)
        self.threshold=threshold
        self.binning=binning
        self.plotflag=plotflag
        self.mask=mask

             

    def extract_number(self,file_path):
        """
        Extract the number from the filename.
        
        Args:
            file_path (str): Path to the file.
        
        Returns:
            int: Extracted number from the filename.
        """
        # Split the filename to get the number
        
        filename = file_path.split('/')[-1]  
        extension=filename.split('.')[-1]
        #print('l960',extension)
        basename=filename.split('.')[0]
        if self.instrument =='ID02':
            number = filename.split('_')[2]
        if self.instrument =='SWING':
            number = filename.split('_')[1]
        if extension =='edf':
            
            number= basename.split('_')[-1].split('-')[0]   
        return int(number)
    
    def extract_titles(self):
        titlelist=self.path+'/Sample_list.txt'
        line2write=''
        for file in self.h5_filelist:
            with h5py.File(file, "r") as f:
                group = list(f.keys())[0]  # Retrieve first group key
                title=str(f[group+'/instrument/id02-rayonixhs-waxs/header/Title'][()].decode('utf-8'))
            
            line2write+=f'{title}\n'
        with open(titlelist,'w') as f:
            f.write(line2write)
        print(line2write)

    def create_dataframe(self):
        logfile=self.path+'/BatchAzimProfileExtraction.log'
        line2write=''
        self.batch_results=[]
        for file in self.h5_filelist:
            try:
                filename=os.path.basename(file)
                map=SAXSDataProcessor(file,qvalues=self.qvalues,instrument=self.instrument,threshold=self.threshold,binning=self.binning,mask=self.mask)
                samplename=map.samplename;Bstring=map.B

                results=map.compute_S() #results are stored in a dictionnary {qvalue:[y0,I,x0,x0_S,gamma,eta,slope,S,R²]}
                
                d= map.compute_correlation_distance()
                
                for qvalue in self.qvalues:
                    background= results[str(qvalue)][0]
                    I=results[str(qvalue)][1]
                    position= results[str(qvalue)][2]
                    x0_S=results[str(qvalue)][3]
                    gamma =results[str(qvalue)][4]
                    eta=results[str(qvalue)][5]
                    slope = results[str(qvalue)][6]
                    S= results[str(qvalue)][7]
                    r_squared=results[str(qvalue)][8]
                    self.batch_results.append([filename,map.data,samplename,Bstring,qvalue,S, r_squared, d_derivative,R1,d_ratio,R2])

            except Exception as e:
                print(f'Failed: {filename} | Reason: {e}')
                line2write += f'Failed: {filename} | Reason: {e}\n'
        with open(logfile,'w') as f:
            f.write(line2write)
        
        # Create a DataFrame from the list self.batch_results
        column_names=[
            'File Name',
            'SAXS Data',
            'Sample name',
            'B (mT)',
            'qvalue',
            'Order Parameter S',
            'R² (S)',
            'Correlation_distance (angströms)'
        ]
        self.df = pd.DataFrame(self.batch_results, columns=column_names)
        self.df.to_csv(self.path+f'/SAXS_processed.csv', index=False)
        
        
        return self.df

       
    def plot_savedata(self,prefix='SAXS',vmin=None,vmax=None):
        print('Extracting and saving scattering data as png files')
            
        for file in self.h5_filelist:
            map=Mapping(file,cif_file=self.cif,reflections=self.reflections,qvalues=self.qvalues,instrument=self.instrument,threshold=self.threshold,binning=self.binning,mask=self.mask)
            samplename=map.samplename;Bstring=map.B;number=map.file_number
            if vmin is None:
                vmin=np.min(np.log10(map.data+1))
            if vmax is None:
                vmax=np.max(np.log10(map.data+1))

            # Apply log scale safely (avoid log(0))
            data_log = np.log10(map.data + 1)

            # Prepare output path
            figpath=self.path+'/png_images'
            os.makedirs(figpath,exist_ok=True)
            
            outputfile = os.path.join(figpath, f'{samplename}_{Bstring}mT_{prefix}_Img{int(number):05d}.png')
            print(f'{outputfile}')
            # Plot and save as PNG
            plt.clf()
            plt.figure()
            
            plt.imshow(data_log, cmap='jet',vmin=vmin,vmax=vmax)  
            plt.colorbar()
            plt.axis('off')
            plt.title(f'{samplename}_{Bstring}mT')
            #plt.legend()
            plt.savefig(outputfile, bbox_inches='tight', pad_inches=0)
            plt.close()
        


