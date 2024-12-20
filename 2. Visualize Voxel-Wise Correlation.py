# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 14:22:30 2024

@author: a.kaymak
"""

import nibabel as nib
import pyvista as pv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import utils_VAT

import warnings
warnings.filterwarnings('ignore')

# load basal ganglia nuclei meshes
GP_meshes       = utils_VAT.load_GP_meshes()
STN_meshes      = utils_VAT.load_STN_meshes()

# color codes for basal ganglia nuclei
colors          = {}
colors["stn"]   = "sandybrown"
colors["gpi"]   = "lightgreen"
colors["gpe"]   = "turquoise"

clinical_data   = pd.read_csv("clinical_data.csv")

data_efield = pd.DataFrame(columns=["patient", "hemisphere", "coordinates", "vector_magnitude"])

for p_id in range(25):
    
    patient = "P"+str(p_id+1)
    
    for hemisphere in ["right", "left"]:
        try:
            VAT, efield = utils_VAT.load_VAT_for_patients(path="data_VAT/"+patient+"/vat_efield_" + hemisphere + ".nii")
            row         = {"patient": patient, "hemisphere": hemisphere, 
                           "coordinates": efield["coordinates"], 
                           "vector_magnitude": efield["vector_magnitude"]}
            
            data_efield.loc[len(data_efield)] = row
        except:
            print("Patient " + patient + " - " + hemisphere + " hemisphere e-field was not found in the directory...")
            
            
# threshold for voxels that were at least covered by 30% of E-fields across patients with a vector magnitude above 150 V/m
N_patient                                = data_efield.patient.nunique()
threshold                                = 30

# filter non-zero e-field vector magnitudes, in previous step we set all e-field points having vector
# magnitude less than 150 V/m to 0, in this step we filte them out
data_efield_filtered                     = data_efield.explode(["coordinates", "vector_magnitude"])
data_efield_filtered                     = data_efield_filtered[data_efield_filtered["vector_magnitude"] > 0]

# expand coordinates into separate columns
data_efield_filtered[["x", "y", "z"]]    = pd.DataFrame(data_efield_filtered["coordinates"].tolist(), index=data_efield_filtered.index)
data_efield_filtered                     = data_efield_filtered[["patient", "hemisphere", "x", "y", "z", "vector_magnitude"]] 

# we assign each e-field position to a voxel
data_efield_filtered                     = utils_VAT.assign_efields_to_voxels(data_efield_filtered)
# we take the mean of all the e-field belonging to same voxel by taking the mean for each patient and hemisphere seperately
data_aggregated_efield                   = utils_VAT.aggregate_voxels(data_efield_filtered)

# group by coordinates and hemisphere, and count how many patients acceptable e-field in the selected voxel
valid_voxels                             = data_aggregated_efield.groupby(['x_voxel', 'y_voxel', 'z_voxel', 'hemisphere'])['patient'].nunique().reset_index()
# calculate percentage of patients having e-field of more than 150 V/m for each voxel
valid_voxels['patient_coverage']         = (valid_voxels['patient'] / N_patient) * 100
valid_voxels                             = valid_voxels[valid_voxels.patient_coverage>=threshold]

filtered_df = data_aggregated_efield.merge(valid_voxels[['x_voxel', 'y_voxel', 'z_voxel', 'hemisphere']], 
                                           on=['x_voxel', 'y_voxel', 'z_voxel', 'hemisphere'],
                                           how='inner')
filtered_df = filtered_df.merge(clinical_data, on='patient', how='left')

data_corr   = utils_VAT.calculate_spearman_correlation(filtered_df)

plotter     = pv.Plotter()

grid        = utils_VAT.create_3D_grid_for_correlation_values(data_corr, hemisphere="left")
plotter.add_mesh(GP_meshes["left"]["gpi"], color=colors["gpi"], opacity=0.15)
plotter.add_mesh(GP_meshes["left"]["gpe"], color=colors["gpe"], opacity=0.15)
plotter.add_mesh(grid, scalars="correlation", clim=[-0.5, 0.5], cmap='coolwarm', render_points_as_spheres=True,  point_size=10, opacity=1)
plotter.set_background("white")
plotter.show()
