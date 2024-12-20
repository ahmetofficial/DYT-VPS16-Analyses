import pandas as pd
import numpy as np
import glob
import pickle

import utils

# load patient dataframe
PATH_reconstruction = "2D Reconstructions/"
patient_files       = glob.glob(PATH_reconstruction +"*.txt")

LEAD_dataframe      = []
for file in patient_files:
    patient_dataframe = utils.read_2D_coordinate_file_into_dataframe(file)
    if(len(LEAD_dataframe)==0):
        LEAD_dataframe = patient_dataframe
    else:
        LEAD_dataframe = pd.concat([LEAD_dataframe, patient_dataframe], ignore_index=False)

# Apply the function to create a new column for stimulation voltages
LEAD_dataframe['stimulation_voltage'] = LEAD_dataframe.apply(utils.get_stimulation_voltage, axis=1)

patient_responsiveness              = {}
patient_responsiveness["patient1"]  = "hyper"
patient_responsiveness["patient3"]  = "hyper"
patient_responsiveness["patient6"]  = "hyper"
patient_responsiveness["patient7"]  = "hyper"
patient_responsiveness["patient9"]  = "hyper"
patient_responsiveness["patient13"] = "hyper"
patient_responsiveness["patient15"] = "hyper"
patient_responsiveness["patient16"] = "hyper"
patient_responsiveness["patient17"] = "hyper"
patient_responsiveness["patient18"] = "hyper"
patient_responsiveness["patient20"] = "hyper"
patient_responsiveness["patient23"] = "hyper"
patient_responsiveness["patient2"]  = "high"
patient_responsiveness["patient4"]  = "high"
patient_responsiveness["patient14"] = "high"
patient_responsiveness["patient26"] = "high"
patient_responsiveness["patient5"]  = "no"
patient_responsiveness["patient8"]  = "no"
patient_responsiveness["patient11"] = "no"
patient_responsiveness["patient12"] = "no"
patient_responsiveness["patient22"] = "no"
patient_responsiveness["patient24"] = "no"
LEAD_dataframe['responsiveness']    = LEAD_dataframe['patient_id'].map(patient_responsiveness)

stimulation_centers = {}

for patient_id in LEAD_dataframe.patient_id.unique():
    
    stimulation_centers[patient_id] = {}
    
    patient_data = LEAD_dataframe[LEAD_dataframe.patient_id==patient_id] 
    
    try:
        right_center = utils.calculate_electric_field(patient_data[patient_data.hemisphere=="right"])
        stimulation_centers[patient_id]["right"] = right_center
        
    except:
        print("--->" + patient_id + ": right hemisphere does not contain LEAD contacts...")
    
    try:
        
        left_center = utils.calculate_electric_field(patient_data[patient_data.hemisphere=="left"])
        left_center = (np.abs(left_center[0]),left_center[1],left_center[2]) #mirror to right hemisphere
        stimulation_centers[patient_id]["left"] = left_center
        
    except:
        print("--->" + patient_id + ": left hemisphere does not contain LEAD contacts...")    
        
    print(patient_id + " is completed...")
    
with open('stimulation_centers.pickle', 'wb') as handle:
    pickle.dump(stimulation_centers, handle, protocol=pickle.HIGHEST_PROTOCOL)