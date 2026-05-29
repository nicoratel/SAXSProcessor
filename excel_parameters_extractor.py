"""
Module to extract parameters (reference file, diameter, length and polydispersity on both) of the Excel file from a h5folder,
useful for computing the nematic order parameter S automatically.
The module also allows to create a final dataframe combining all the informations from the excel file and the results of 
the S computation.

The first main function is `get_all_params` which takes the path to the excel file,
the path to the folder containing reference files, and the path to the h5 files.
Parameters extracted include:
- Diameter [nm]
- Length [nm]
- Polydispersity on diameter [%]
- Polydispersity on length [%]
- Path of the reference file for background subtraction. This path can then be 
used in the SAXS processing functions such as `average_h5_processor`.

The second main function is `final_dataframe` which takes the path to the h5 folder,
the path to the excel file, and the path to the csv results file.
It creates a final dataframe combining all the parameters from the excel file and the results from the csv file.

Notes: Assumes a specific structure of the excel file with sheets 'MEASUREMENTS' and 'SAMPLES'.
The indexes of the h5 files must be in "FILE" column in "MEASUREMENTS" sheet. 
For example: for files lacroix_00782_2024-07-20_02-32-08.h5 
to lacroix_00800_2024-07-20_02-45-10.h5, they must be written as "00782-00800"
in the "FILE" column of the excel.

Library dependencies:
- pandas
- numpy
- os

Example usage:
----------
diameter, polydispersity, polydispersity_percent, reference_file = get_all_params(
    excel_file='path/to/excel.xlsx',
    bg_path='path/to/reference/files',
    h5path='path/to/h5/files'
)   

final_dataframe()
    h5path='path/to/h5/files',
    excel_file='path/to/excel.xlsx',
    csv_results_file='path/to/csv/results.csv'
)

"""


import pandas as pd
import numpy as np
import os


def excel_preprocess(excel_file, h5path):
    """
    Finds reference file, diameter and polydispersity from the excel file for a given measurement.
    Finds the index range of the h5 files. Uses them to find the experiments in the excel file.
    Once the experiments are found, it retrieves the diameter, polydispersity and reference file for background subtraction.
    Parameters
    ----------
    excel_file : str
        Path to the excel file.
    h5path : str
        Path to the h5 files.
        Example of file: lacroix_00782_2024-07-20_02-32-08.h5, the ID is 00782.  
    Returns
    -------
    diameter : float
        Diameter in nm.
    polydispersity : float
        Polydispersity in nm.
    polydispersity_percent : float
        Polydispersity in percent.
    reference : str
        Name of the reference file. Returns None if couldn't find it.
    """

    # Load the measurement dataframe
    df_measurement = pd.read_excel(excel_file, sheet_name= 'MEASUREMENTS', header=3) 

    # 1. Preprocess dataframe (remove useless columns, rename some columns)               
    # Issue: some columns are empty, remove them
    # Only do the preprocess if not already done (to do)

    # Preprocess dataframe (remove useless columns, rename some columns)               
    # Issue: the first column "name" is B + C so the second column is useless, remove it 
    df_measurement = df_measurement.drop(columns=['Unnamed: 0']) # empty column
    df_measurement = df_measurement.drop(columns=['Unnamed: 2']) # empty column
    df_measurement = df_measurement.drop(columns=['Unnamed: 29']) # empty column
    # rename some columns
    df_measurement = df_measurement.rename(columns={'Unnamed: 27': 'fail', # because it was written on multiple lines 
                                                    'Unnamed: 28': 'COMMENTS',
                                                    'Unnamed: 30': 'd [nm]',
                                                    'Unnamed: 31': 'S'})
    df_measurement = df_measurement.fillna(0)  # remplace with 0

    # 2. For each measurement find the reference (and its own measurement)
    # The reference measurement to substract later is in "REFERENCE",
    # Find the row in which that reference was measured 
    # = the row in which "SAMPLE" == "REFERENCE" and get the column "FILE" (which gives the ID of our background file)

    diameter, polydispersity_d, length, polydispersity_L, B, reference_ID = process_measurement(excel_file, df_measurement, h5path)

    return diameter, polydispersity_d, length, polydispersity_L, B, reference_ID

def process_measurement(excel_file, df_measurement, h5file):

    # 1. Find the measurement name from the h5path
    import os
    # ID_list = []
    f = os.path.basename(h5file)
    if f.endswith('.h5') or f.endswith('.nxs'):
    # assuming the format is 'name_ID_date_time.h5', ex: lacroix_00782_2024-07-20_02-32-08.h5
        ID = f.split('_')[1] # ID is 00782, str to 782.0, float
        ID = ID.lstrip('0') # remove leading zeros, 782.0 to 782, str
        ID = float(ID) # 782, int
    else : 
        print(f"File {f} is not an h5 or nxs file.")

    # Find the measurement name corresponding to that ID range
    try :
        measurement_name = df_measurement.loc[df_measurement['FILE'] == ID, 'NAME'].values[0]
        print("Measurement name found:", measurement_name, "for index of h5 file", (ID))


    except IndexError:
        print(f"Couldn't find the measurement name for the given ID {ID}.")
        return None, None, None, None, None, None   
    
    # Search the corresponding measurement of the reference (where reference_name in column "SAMPLE")
    reference_name = df_measurement.loc[df_measurement['NAME'] == measurement_name, 'REFERENCE'].values[0] # find reference
    try:
        raw_index = df_measurement.index[df_measurement['NAME'] == reference_name].tolist()[0] # find the index of the raw (NAME OR SAMPLE ?) 
        # Find the reference file name ID
        reference_ID = df_measurement.loc[raw_index, 'FILE']
        # print("Reference file ID:", reference_ID)
    except IndexError:
        # print("Couldn't find the reference measurement for", reference_name)
        reference_ID = None

    # Find magnetic feild value for that measurement : first case: directly a number (great)
    # second case (less great): it is written as "0 before" or "0 after", in this case retreive only the number and convert it to float !
    if isinstance(df_measurement.loc[df_measurement['NAME'] == measurement_name, 'VALUE [mT]'].values[0], str):
        try : 
            B_str = df_measurement.loc[df_measurement['NAME'] == measurement_name, 'VALUE [mT]'].values[0]
            B_str = B_str.split()[0] # take only the first part of the string, ex: "0 before" to "0"
            B = float(B_str) # convert to float
            # print("Magnetic field [mT]", B)
        except Exception as e:
            B = None
            # print(f"Couldn't find the magnetic field for measurement {measurement_name} because of error: {e}")   
    else :
        try :
            B = df_measurement.loc[df_measurement['NAME'] == measurement_name, 'VALUE [mT]'].values[0]
            # print("Magnetic field [mT]", B)
        except IndexError:
            B = None
            # print('Couldn\'t find the magnetic field for measurement  ', measurement_name)    

    # Find diameter and polydispersity in df_sample 
    sample_name = df_measurement.loc[df_measurement['NAME'] == measurement_name, 'SAMPLE'].values[0] 
    # Preprocess of the second dataframe
    df_sample = pd.read_excel(excel_file, sheet_name= 'SAMPLES', header=3)
    df_sample = df_sample.drop(columns=['Unnamed: 0']) # empty column
    df_sample = df_sample.drop(columns=['Unnamed: 2']) # empty column
    df_sample = df_sample.fillna(0)  # remplace with 0

    ############## Diameter and its polydispersity in % ################
    try:   
        diameter = df_sample.loc[df_sample['NAME'] == sample_name, 'DIAMETER [nm]'].values[0]
        # print('Diameter [nm]', diameter)
    except IndexError: 
        diameter = None
        # print('Couldn\'t find the diameter for sample  ', sample_name)
    try:
        polydispersity_d = df_sample.loc[df_sample['NAME'] == sample_name, 'POLYDISPERSITY d [%]'].values[0]
        # print("polydispersity d [%]", polydispersity_d)
    except IndexError:
        polydispersity_d = None
        # print('Couldn\'t find the polydispersity d [%] for sample  ', sample_name)
    
    ############## Length and its polydispersity in % #################
    try:
        aspect_ratio = df_sample.loc[df_sample['NAME'] == sample_name, 'AR'].values[0]
        length = diameter * aspect_ratio
    except IndexError:
        aspect_ratio = None
        length = None
        # print('Couldn\'t find the aspect ratio for sample  ', sample_name)
    try:
        polydispersity_L = df_sample.loc[df_sample['NAME'] == sample_name, 'POLYDISPERSITY L [%]'].values[0]
        # print("polydispersity L [%]", polydispersity_L)
    except IndexError:
        polydispersity_L = None
        # print('Couldn\'t find the polydispersity L [%] for sample  ', sample_name)

    return diameter, polydispersity_d, length, polydispersity_L, B, reference_ID


def find_ref_file_from_ID(ID, bg_path):
    """
    Finds the reference file name from its ID in the given path.
    Example of file name: "rodriguez_01093_2024-10-11_05-15-41.h5"
    Parameters
    ----------
    ID : float
        ID of the reference file: receive a float, ex: 62.0 and has to be transformed to "00062" to find the file.
    path : str
        Path to the folder containing the files.
    Returns
    -------
    reference_file : str
        Name of the reference file.
    """

    import os
    ID = str(int(ID)).rjust(5, '0') # transform 62.0 to "00062"
    for file in os.listdir(bg_path):
        if ID in file:
            reference_file = file
            return os.path.join(bg_path, reference_file)
    return None


def get_all_params(excel_file, bg_path, h5path):
    """
    Finds reference file, diameter and polydispersity from the excel file for a given measurement.
    Parameters
    ----------
    file : str
        Path to the excel file.
    measurement_name : str
        Optional : Name of the measurement to find the parameters for. 
        If not provided, defaults is all the measurements.
    path : str
        Path to the folder containing the reference files.
    Returns
    -------
    diameter : float
        Diameter in nm.
    polydispersity : float
        Polydispersity in nm.
    polydispersity_percent : float
        Polydispersity in percent.
    reference_file : str
        Name of the reference file.
    """

    diameter, polydispersity_d, length, polydispersity_L, B, reference_ID = excel_preprocess(excel_file, h5path)
    if reference_ID is None:
        reference_file = None
    else:
        reference_file = find_ref_file_from_ID(reference_ID, bg_path)

    return diameter, polydispersity_d, length, polydispersity_L, B, reference_file


def final_dataframe(h5path, excel_file, csv_results_path):
    """
    Creates the final dataframe with all the parameters from the excel file and the csv results for a given h5path.
    Concatenates the two sheets "SAMPLES" and "MEASUREMENTS" from the excel file with the results (S, d) from the csv file.
    
    Parameters
    ----------
    h5path : str
        path of the h5 folder (one experiment)
    excel_file : str
        path of the excel file containing the experiment informations
    csv_results_path : str
        path of the csv file containing the results

    Notes:
    1. Assumes a specific structure of the excel file with sheets 'MEASUREMENTS' and 'SAMPLES'.
    The indexes of the h5 files must be in "FILE" column in "MEASUREMENTS" sheet. 
    For example: for files lacroix_00782_2024-07-20_02-32-08.h5 
    to lacroix_00800_2024-07-20_02-45-10.h5, they must be written as "00782-00800"
    in the "FILE" column of the excel.
    2. "COMMENTS" column is removed.
    """

    # 1 Get the "MEASUREMENTS" sheet informations
    df_measurement = pd.read_excel(excel_file, sheet_name= 'MEASUREMENTS', header=3)
    # 1.1 Preprocess the dataframe (remove useless columns, rename some columns)               
    # Issue: the first column "name" is B + C so the second column is useless, remove it 
    # Only do the preprocess if not already done (to do)

    # Preprocess dataframe (remove useless columns, rename some columns)               
    # Issue: the first column "name" is B + C so the second column is useless, remove it 
    df_measurement = df_measurement.drop(columns=['Unnamed: 0']) # empty column
    df_measurement = df_measurement.drop(columns=['Unnamed: 2']) # empty column
    df_measurement = df_measurement.drop(columns=['Unnamed: 29']) # empty column
    # rename some columns
    df_measurement = df_measurement.rename(columns={'Unnamed: 27': 'fail', # because it was written on multiple lines 
                                                    'Unnamed: 28': 'COMMENTS',
                                                    'Unnamed: 30': 'd [nm]',
                                                    'Unnamed: 31': 'S'})
    df_measurement = df_measurement.fillna(0)  # remplace with 0

    # 1.2 Find the measurement name corresponding to that h5path
    try : 
        import os

        for f in os.listdir(h5path):
            if f.endswith('.h5') or f.endswith('.nxs'):
                ID = f.split('_')[1] # assuming the format is 'name_ID_date_time.h5'
                ID = ID.lstrip('0') # remove leading zeros, 782.0 to 782, str
                ID = float(ID) # 782, int
                print("ID found in h5path:", ID)
        measurement_raw = df_measurement.loc[df_measurement['FILE'] == ID]
        measurement_name = measurement_raw['NAME'].values[0]
    except IndexError:
        print(f"Couldn't find the measurement name for the given ID {ID}.")
        return None
    

    # 2. Get the "SAMPLES" sheet informations

    # 2.1 Preprocess of the second dataframe
        # Preprocess of the second dataframe
    try:
        df_sample = pd.read_excel(excel_file, sheet_name= 'SAMPLES', header=3)
        df_sample = df_sample.drop(columns=['Unnamed: 0']) # empty column
        df_sample = df_sample.drop(columns=['Unnamed: 2']) # empty column
        df_sample = df_sample.fillna(0)  # remplace with 0
    except Exception as e:
        print(f"Error reading or preprocessing SAMPLES sheet: {e}")
        return None

    # 2.2 Find the sample of the corresponding measurement
    try :
        sample_name = measurement_raw['SAMPLE'].values[0] 
        sample_raw = df_sample.loc[df_sample['NAME'] == sample_name]
        if sample_raw.empty:
            print("Couldn't find the sample name for the given measurement.")
            return None
    except Exception as e:
        print(f"Error finding sample name: {e}")
        return None
    

    # 3. Get the results from the csv file
    
    # 3.1 Open csv results containing S and d values
    df_results = pd.read_csv(csv_results_path)
 
    # Find the name of the h5 folder in "samplename" column (link between h5 folder and csv results)
    try : 
        # sample_name = os.path.basename(h5path) not anymore, S0000_M0119 now
        sample_name = f"{sample_name}_{measurement_name}" # because in the csv file, the samplename is "S0000_M0119" for example, not "M0119"s
        print('sample_name', sample_name)
        row_results_list = df_results.index[df_results['samplename'] == sample_name].tolist()
        if not row_results_list:
            print(f" Couldn't find the h5 folder {sample_name} in the results csv file.")
            return False
        elif len(row_results_list) > 1:
            print(f" Found multiple {sample_name} in the results csv file, take first (index {row_results_list[0]})")
        row_results = row_results_list[0]
    except Exception as e:
        print(f"Error finding h5 folder in results csv: {e}")
        return None

    # 3.2 Get S and d values
    # list in a csv is a string ... not good: retreat it as string and convert to array
    try :
        new_value_S = df_results.loc[row_results, 'nematic_parameter']
        print('S value = ', new_value_S, 'type of S value = ', type(new_value_S))
        # if S is a list in the csv, it is read as a string, so we need to convert it back to a list
        if isinstance(new_value_S, str) and new_value_S.startswith('[') and new_value_S.endswith(']'):
            S_str = new_value_S.strip("[]")  # enlève les crochets
            S_array = np.fromstring(S_str, sep=' ')
            print('S array[0] = ', S_array[0], 'type of S array = ', type(S_array))
        else :
            S_value = float(new_value_S) # if it's not a list, just convert it to float
        # list in a csv is a string... not good: retreat it as string and convert to array
        new_value_distance = df_results.loc[row_results, 'distance (A)']
        print('distance [A] = ', new_value_distance, 'type of distance value = ', type(new_value_distance))
        if isinstance(new_value_distance, str) and new_value_distance.startswith('[') and new_value_distance.endswith(']'):
            d_str = new_value_distance.strip("[]")  # enlève les crochets
            d_array = np.fromstring(d_str, sep=' ')
            print('d array[0] = ', d_array[0], 'type of d array = ', type(d_array))   
        else :
            d_value = float(new_value_distance) # if it's not a list, just convert it to float
    except Exception as e:
        print(f"Error extracting S and d values from results csv: {e}")
        return None


    # 4. Concatenate all data into a final dataframe
    try:
        final_df = pd.concat([measurement_raw.reset_index(drop=True), 
                              sample_raw.reset_index(drop=True)], axis=1)
        # Add S and d values to the final dataframe
        if isinstance(new_value_S, str) and new_value_S.startswith('[') and new_value_S.endswith(']'):
            final_df['nematic_parameter'] = [S_array]
        if isinstance(new_value_distance, str) and new_value_distance.startswith('[') and new_value_distance.endswith(']'):
            final_df['distance (A)'] = [d_array]
        if not (isinstance(new_value_S, str) and new_value_S.startswith('[') and new_value_S.endswith(']')):
            final_df['nematic_parameter'] = S_value
        if not (isinstance(new_value_distance, str) and new_value_distance.startswith('[') and new_value_distance.endswith(']')):
            final_df['distance (A)'] = d_value
        return final_df
    
    except Exception as e:
        print(f"Error creating final dataframe: {e}")

        return None 
