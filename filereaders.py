import h5py
import numpy as np
import os
import re
import fabio

# ============================================================================
# FILE READERS - Unified interface for different beamlines
# ============================================================================

class EdfFile:
    """Handler for EDF format files (LGC beamline)"""
    
    def __init__(self, file, I_0=1):
        self.file = file
        self.lineEraser = 'vd' in self.file.split('_')
        
        if self.lineEraser:
            file1, file2 = self._get_individual_files_lineEraser(self.file)
            im1 = fabio.open(file1)
            header1 = im1.header
            im2 = fabio.open(file2)
            header2 = im2.header
            
            self.x_center = float(header1['Center_1'])
            date = header1['Date']
            annee = int(date[:4])
            if annee < 2025:
                self.z_center = (float(header2['Center_2']) + float(header1['Center_2'])) / 2
            else:
                self.z_center = float(header2['Center_2'])
        else:
            self.lineEraser = False
              
        image = fabio.open(self.file)
        header = image.header
        self.data =image.data.astype(float)
        #trés important pour pouvoir soustraire la référence
        
        shape = self.data.shape
        self.num_pixel_x = shape[0]
        self.num_pixel_z = shape[1]
               
        self.wl = float(header['WaveLength'])
        
        if not self.lineEraser:
            self.x_center = float(header['Center_1'])
            self.z_center = float(header['Center_2'])
            
        self.pixel_size_x = float(header['PSize_1'])
        self.pixel_size_z = float(header['PSize_2'])
        self.D = float(header['SampleDistance'])
        self.samplename = header['Comment']
        self.nb_frames = 1
        self.file_number = int(file.split('/')[-1].split('.')[0].split('_')[-1].split('-')[0])
        self.bin_x = 1
        self.bin_y = 1
        self.B = self._extract_B_value()
        # transmission
        self.transmission = float(header['TransmittedFlux'])/I_0
        # extract x and z motor positions
        try:
            self.x = header['x']
        except:
            self.x = 'N.A.'
        try:
            self.z = header['z']
        except:
            self.z = 'N.A.'

        
    def _get_individual_files_lineEraser(self, file):
        directory, basename = os.path.split(file)
        filename, _ = os.path.splitext(basename)

        try:
            prefix, filenumbers = re.split(r'_vd_', filename, maxsplit=1)
            n1, n2 = map(int, filenumbers.split('-'))
        except ValueError as e:
            raise ValueError(f"Invalid filename format: {basename}") from e

        file1 = os.path.join(directory, f"{prefix}_{n1:05d}.edf")
        file2 = os.path.join(directory, f"{prefix}_{n2:05d}.edf")

        return file1, file2
    
    def _extract_B_value(self):
        match = re.search(r'(\d+)\s*mT', self.file)
        if match:
            return int(match.group(1))  
        return 0


class h5File_ID02:
    """Handler for HDF5 files from ESRF-ID02 beamline"""
    
    def __init__(self, file):
        self.file = file
        self.file_number = self._extract_number()
               
        if file is None:
            raise ValueError("Please specify a data file path")
        if "_waxs_" in file:
            with h5py.File(file, "r") as f:
                group = list(f.keys())[0]  # Retrieve first group key
                self.title=str(f[group+'/instrument/id02-rayonixhs-waxs/header/Title'][()].decode('utf-8'))
                self.nb_frames = int(f[group + '/instrument/id02-rayonixhs-waxs/acquisition/nb_frames'][()])
                self.acq_time = float(f[group + '/instrument/id02-rayonixhs-waxs/acquisition/exposure_time'][()])
                
                # Retrieve image data
                target = group + '/measurement/data'
                data = np.array(f[target])
                self.data = np.mean(data, axis=0)  # Average over frames
                shape = np.shape(self.data)
                self.num_pixel_x = shape[0]
                self.num_pixel_z = shape[1]
                
                # Retrieve header information
                header = group + '/instrument/id02-rayonixhs-waxs/header'
                self.pixel_size_x = float(f[header + '/PSize_1'][()].decode('utf-8'))
                self.pixel_size_z = float(f[header + '/PSize_2'][()].decode('utf-8'))
                self.wl = float(f[header + '/WaveLength'][()])
                self.x_center = float(f[header + '/Center_1'][()].decode('utf-8'))
                self.z_center = float(f[header + '/Center_2'][()].decode('utf-8'))
                self.D = float(f[header + '/SampleDistance'][()].decode('utf-8'))

                # Retrieve binning info
                header = '/entry_0000/instrument/id02-rayonixhs-waxs/image_operation/binning'
                self.bin_x = f[header + '/x'][()]
                self.bin_y = f[header + '/y'][()]  
        elif "_eiger2_" in file:
            with h5py.File(file, "r") as f:
                group = list(f.keys())[0]
                self.title = str(f[group + '/instrument/id02-eiger2-saxs/header/Title'][()].decode('utf-8'))
                self.nb_frames = int(f[group + '/instrument/id02-eiger2-saxs/acquisition/nb_frames'][()])
                self.acq_time = float(f[group + '/instrument/id02-eiger2-saxs/acquisition/exposure_time'][()])
                
                target = group + '/measurement/data'
                self.data = np.array(f[target])
                shape = np.shape(self.data)
                
                if len(shape) == 2:
                    self.num_pixel_x = shape[0]
                    self.num_pixel_z = shape[1]
                elif len(shape) == 3:
                    self.num_pixel_x = shape[1]
                    self.num_pixel_z = shape[2]
                else:
                    raise ValueError(f"Data in file {self.file} should have 2 or 3 dimensions")
                
                header = '/entry_0000/instrument/id02-eiger2-saxs/header'
                self.pixel_size_x = float(f[header + '/PSize_1'][()].decode('utf-8'))
                self.pixel_size_z = float(f[header + '/PSize_2'][()].decode('utf-8'))
                self.wl = float(f[header + '/WaveLength'][()])
                self.x_center = float(f[header + '/Center_1'][()].decode('utf-8'))
                self.z_center = float(f[header + '/Center_2'][()].decode('utf-8'))
                self.D = float(f[header + '/SampleDistance'][()])
                self.transmission = float(f[header+ '/HSI0Factor'][()].decode('utf-8'))
                header = '/entry_0000/instrument/id02-eiger2-saxs/image_operation/binning'
                self.bin_x = f[header + '/x'][()]
                self.bin_y = f[header + '/y'][()]
                
        self.samplename = self._extract_sample_name()
        self.B = self._extract_magnetic_field()
        

    def _extract_magnetic_field(self):
        match = re.search(r'(\d+(\.\d+)?)(mT|T)', self.title)
        if match:
            value, _, unit = match.groups()
            value = float(value)
            if unit == 'T':
                value *= 1000
            return int(value)
        return 0
    
    def _extract_sample_name(self):
        pattern = re.compile(r"^(.*?)(?:_\d+(?:\.\d+)?(?:mT|T).*)$")
        match = pattern.match(self.title)
        if match:
            return match.group(1)
        return self.title
        
    def _extract_number(self):
        filename = self.file.split('/')[-1]  
        number = filename.split('_')[2]       
        return int(number)


class h5File_SWING:
    """Handler for HDF5 files from SOLEIL-SWING beamline"""
    
    def __init__(self, file: str, mean=True, force_linescan = False):
        self.file = file
        self.file_number = self._extract_number()
        self.mean = mean    
        if not os.path.exists(file):
            raise FileNotFoundError(f"File {file} not found.")
          
        self._extract_from_h5()
        
        self.data = self._extract_scatteringdata()
        self.B = self._extract_magnetic_field()
        
    def _extract_number(self):
        filename = os.path.basename(self.file)
        number = filename.split('_')[1]
        return int(number)

    def _extract_from_h5(self):
        with h5py.File(self.file, "r") as f:
            group = list(f.keys())[0]
            self.samplename = f[group + '/sample_info/ChemSAXS/sample_name'][()].decode('utf-8')
            target = group + '/scan_data/eiger_image'
            data_raw = np.array(f[target])
                       
            target = group + '/SWING/EIGER-4M'
            self.D = f[target + '/distance'][0] / 1000
            self.pixel_size_x = f[target + '/pixel_size_x'][0] * 1e-6
            self.pixel_size_z = f[target + '/pixel_size_z'][0] * 1e-6
            self.x_center = f[target + '/dir_beam_x'][0]
            self.z_center = f[target + '/dir_beam_z'][0]
            if self.mean:
                self.nb_frames=1
            else:
                self.nb_frames = f[target + '/nb_frames'][0]
            self.bin_x = f[target + '/binning_x'][0]
            self.bin_y = f[target + '/binning_y'][0]
            self.acq_time = f[target + '/exposure_time'][0]
                        
            target = group + '/SWING/i11-c-c03__op__mono'
            self.wl = f[target + '/wavelength'][0] * 1e-10
                        
            self.folder = os.path.dirname(self.file)

            # Retrieve Basler microscope image
            self.basler_image = f[group + '/SWING/i11-c-c08__dt__basler_analyzer/image'][()]
            # Retrieve positions (start and end) for X and Z (for possible mapping or alignment)
            self.position_x_start = f[group + '/SWING/i11-c-c08__ex__tab-mt_tx.4/position'][()]
           
            self.position_x_end = f[group+'/SWING/i11-c-c08__ex__tab-mt_tx.4/position_post'][()]
            self.position_z_start = f[group + '/SWING/i11-c-c08__ex__tab-mt_tz.4/position'][()]
            self.position_z_end = f[group + '/SWING/i11-c-c08__ex__tab-mt_tz.4/position_post'][()]
            self.position = {'X_start': self.position_x_start, 'X_end': self.position_x_end,
                        'Z_start': self.position_z_start, 'Z_end': self.position_z_end}
            print("position moteur :")
            #print(self.position)
            # Calculate step_x  and step_z
            self.step_x=(self.position_x_end-self.position_x_start)/data_raw.shape[0]
            #print(self.position)
            self.step_z=(self.position_z_end-self.position_z_start)/data_raw.shape[1]
            #print("step :")
            #print(self.step_x)
            #print(self.step_z)
            #print(self.nb_frames)
            self.averagemi5 = f[group + '/scan_data/averagemi5'][()][()]
            self.averagemi8a = f[group + '/scan_data/averagemi8a'][()][()]
            

            #if self.step_z < 0.1:
            #    self.step_z=0  

            # Retrive sample transmission
            self.transmission = f[group + '/sample_info/sample_transmission'][()] # transmission array for line scans
            if self.mean:
                self.transmission = np.mean(self.transmission)     
           
    def _extract_magnetic_field(self):
        match = re.search(r'(\d+)\s*mT', self.samplename)
        if match:
            return int(match.group(1))  
        return 0

    def _extract_scatteringdata(self):
        with h5py.File(self.file, "r") as f:
            group = list(f.keys())[0]
            target = group + '/scan_data/eiger_image'
            data_raw = np.array(f[target])
            self.raw_data = data_raw
                        
        print(f"Raw data shape: {data_raw.shape}")
        if (len(data_raw.shape)==3) & self.mean:
            data = np.mean(data_raw, axis=0) # average over lines
        elif (len(data_raw.shape)==4) & self.mean:
            data = np.mean(np.mean(data_raw, axis=0), axis=0) # average over lines & columns scans
        else:
            data = data_raw          
        
        self.num_pixel_x = data.shape[-2]
        self.num_pixel_z = data.shape[-1]
        self.Pts_x = data.shape[0]
        self.Pts_z = data.shape[1]
        #print("info")
        #print(self.Pts_x)
        #print(self.Pts_z)
        return data
    

    def convert2edf(self, outputdir=None):
        filelist = []
        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                print(j) 
                print("convert")   
                data2save = self.data[i][j]
                print(data2save)
                x = self.position_x_start + i * self.step_x
                z = self.position_z_start + j * self.step_z
                transmission_array = np.array(self.transmission).reshape((self.data.shape[1], self.data.shape[0]))
                print("transmission", transmission_array)
                header = {
                    "WaveLength": str(self.wl),
                    "Center_1": str(self.x_center),
                    "Center_2": str(self.z_center),
                    "PSize_1": str(self.pixel_size_x * self.bin_x),
                    "PSize_2": str(self.pixel_size_z * self.bin_y),
                    "SampleDistance": str(self.D),
                    "Comment": str(self.samplename),
                    "x":str(x[0]),
                    "z":str(z[0]),
                    "TransmittedFlux":str(transmission_array[j][i]) if not self.mean else str(self.transmission)
                } 
                img = fabio.edfimage.edfimage(data=data2save, header=header)
                
                if outputdir is None:
                    outputdir = self.folder
                else:
                    os.makedirs(outputdir, exist_ok=True)
                    
                filename = os.path.join(outputdir, f'{self.samplename}_File_{self.file_number}_Img_{j}_{i}.edf')
                img.write(filename)
                filelist.append(filename)
        return filelist
    
    '''def convert2edf(self, outputdir=None):
        filelist = []
        for i in range(self.nb_frames):
            data2save = self.data[i]
            x = self.position_x_start + i * self.step_x
            z = self.position_z_start + i * self.step_z # je comprends pas car on est sur une ligne
            header = {
                "WaveLength": str(self.wl),
                "Center_1": str(self.x_center),
                "Center_2": str(self.z_center),
                "PSize_1": str(self.pixel_size_x * self.bin_x),
                "PSize_2": str(self.pixel_size_z * self.bin_y),
                "SampleDistance": str(self.D),
                "Comment": str(self.samplename),
                "x":str(x[0]),
                "z":str(z[0]),
                "TransmittedFlux":str(self.transmission[i]) if not self.mean else str(self.transmission)
            } 
            img = fabio.edfimage.edfimage(data=data2save, header=header)
            
            if outputdir is None:
                outputdir = self.folder
            else:
                os.makedirs(outputdir, exist_ok=True)
                
            filename = os.path.join(outputdir, f'{self.samplename}_File_{self.file_number}_Img_{i}.edf')
            img.write(filename)
            filelist.append(filename)
        return filelist'''
    
    def convert_SWING_mask(self,maskfile):
        maskdata = np.loadtxt(maskfile, delimiter = ';')
        header = {
                "WaveLength": str(self.wl),
                "Center_1": str(self.x_center),
                "Center_2": str(self.z_center),
                "PSize_1": str(self.pixel_size_x*self.bin_x),
                "PSize_2": str(self.pixel_size_z*self.bin_y),
                "SampleDistance": str(self.D),
                "Comment": str(self.samplename)
            } 
        maskdata = 1 - maskdata
        if self. nb_frames==1:
            masksize = self.data.shape
        else:
            if len(self.data.shape) == 3:
                masksize = self.data.shape[1:]
                
            else:
                masksize = self.data.shape[2:]
        maskdata.reshape(masksize)
        # Write mask file
        outputname = os.path.join(self.folder,'mask_pyfai.edf')
        obj = fabio.edfimage.EdfImage(header=header,data=maskdata)
        obj.write(outputname)
        print(f'Mask file was successfully imported and converted in {outputname}')
        return outputname

        
        

