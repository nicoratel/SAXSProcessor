
from utilities_SWING import *
import pandas as pd
from saxsprocessor import SAXSProcessor
import glob
import os
import numpy as np
from matplotlib import pyplot as plt
import re
from utilities_SWING import *
import pandas as pd
from filereaders import h5File_SWING
from utilities import *
from pathlib import Path
from excel_parameters_extractor import get_all_params


excel_file = "Run_synchrotron_SOLEIL_02_2026_suivi_manips_after_run.xlsm" # path to the excel file containing the parameters 
path =  'T:/LPCNO/NCO/Manips/DATA_SAXS/SWING_align_fev26/20250369/2026/Run1/lacroix/S0000/' # Attention important pour stock
bg_path = 'T:/LPCNO/NCO/Manips/DATA_SAXS/SWING_align_fev26/20250369/2026/Run1/lacroix/ref'
csv_results_path = os.path.join(path,'analysis_results_batch.csv')
mask = 'mask_pyfai.edf'
instrument='SWING'


# ONGOING WORK: Generalize it more by finding automatically the parameters from the excel file and the h5path 
# Results file will be saved in the same folder as the h5 files, with the name "analysis_results_batch.csv". 
# It will contain the results for all the h5 files in the folder.
from excel_parameters_extractor import get_all_params
results = global_analysis(
    csv_file=csv_results_path,
    h5path=path,
    bg_path= bg_path,
    excel_file=excel_file, # new param
    instrument=instrument,
    autosubstract=True,
    mask=mask,
    nb_peaks=3,
    qmin=0.01,
    qmax=0.15,
    azimuth=90,
    width=360,
    method='hybrid',
    # in the case you use the excel to find it automatically, you can set radius and L to None, and it will be done in the function using the get_all_params function
    radius= None, 
    L= None,
    plot=False,
    verbose=False)