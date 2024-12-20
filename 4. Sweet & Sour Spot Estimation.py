import nibabel as nib
import pyvista as pv
import numpy as np
import pandas as pd

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

###############################################################################
# 1. Data Loading and Voxel-Based Correlation Estimation ######################
###############################################################################

clinical_data   = pd.read_csv("clinical_data.csv")

data_efield = pd.DataFrame(columns=["patient", "hemisphere", "coordinates", "vector_magnitude"])

for p_id in range(25):
    
    patient = "P"+str(p_id+1)
    
    for hemisphere in ["right", "left"]:
        try:
            VAT, efield = utils_VAT.load_efields_for_patients(path="data_VAT/"+patient+"/vat_efield_" + hemisphere + ".nii")
            row         = {"patient": patient, "hemisphere": hemisphere, 
                           "coordinates": efield["coordinates"], 
                           "vector_magnitude": efield["vector_magnitude"]}
            
            data_efield.loc[len(data_efield)] = row
        except:
            print("Patient " + patient + " - " + hemisphere + " hemisphere e-field was not found in the directory...")
            
# threshold for voxels that were at least covered by 30% of E-fields across patients with a vector magnitude above 150 V/m
N_patient                                = data_efield.patient.nunique()
threshold                                = 30 # %
voxel_size                               = 0.25

# filter non-zero e-field vector magnitudes, in the previous step we set all e-field points having vector
# magnitude less than 150 V/m to 0, in this step, we filter them out
data_efield_filtered                     = data_efield.explode(["coordinates", "vector_magnitude"])
data_efield_filtered                     = data_efield_filtered[data_efield_filtered["vector_magnitude"] > 0]

# expand coordinates into separate columns
data_efield_filtered[["x", "y", "z"]]    = pd.DataFrame(data_efield_filtered["coordinates"].tolist(), index=data_efield_filtered.index)
data_efield_filtered                     = data_efield_filtered[["patient", "hemisphere", "x", "y", "z", "vector_magnitude"]] 

# we assign each e-field position to a voxel
data_efield_filtered                     = utils_VAT.assign_efields_to_voxels(data_efield_filtered, voxel_size=voxel_size)
# we take the mean of all the e-fields belonging to the same voxel by taking the mean for each patient and hemisphere separately
data_aggregated_efield                   = utils_VAT.aggregate_voxels(data_efield_filtered)

# group by coordinates and hemisphere, and count how many patients acceptable e-field in the selected voxel
valid_voxels                             = data_aggregated_efield.groupby(['x_voxel', 'y_voxel', 'z_voxel', 'hemisphere'])['patient'].nunique().reset_index()
# calculate the percentage of patients having an e-field of more than 150 V/m for each voxel
valid_voxels['patient_coverage']         = (valid_voxels['patient'] / N_patient) * 100
valid_voxels                             = valid_voxels[valid_voxels.patient_coverage>=threshold]

filtered_EFields = data_aggregated_efield.merge(valid_voxels[['x_voxel', 'y_voxel', 'z_voxel', 'hemisphere']], 
                                           on=['x_voxel', 'y_voxel', 'z_voxel', 'hemisphere'], how='inner')
filtered_EFields = filtered_EFields.merge(clinical_data, on='patient', how='left')
data_corr        = utils_VAT.calculate_spearman_correlation(filtered_EFields)

###############################################################################
# 2. Plot Sweet and Sour Spots Visulalization #################################
###############################################################################

hemisphere       = "right"
voxel_size       = 0.25
plane            = "xz"

sweet_spot_voxels, sour_spot_voxels = utils_VAT.extract_significant_voxels_for_sweet_and_sour_regions(data_corr, hemisphere, voxel_size)

sweet_spot_map   = utils_VAT.define_3D_ROI(sweet_spot_voxels, voxel_size)
sour_spot_map    = utils_VAT.define_3D_ROI(sour_spot_voxels, voxel_size)

# Visualize with PyVista
plotter = pv.Plotter()

plotter.add_mesh(sour_spot_map, scalars="density", opacity="density", cmap="Blues", clim=[0,1])
plotter.add_mesh(sweet_spot_map, scalars="density", opacity="density", cmap="Reds", clim=[0,1])

plotter.add_mesh(GP_meshes[hemisphere]["gpi"], color=colors["gpi"], opacity=0.15)
plotter.add_mesh(GP_meshes[hemisphere]["gpe"], color=colors["gpe"], opacity=0.15)
plotter.add_mesh(STN_meshes[hemisphere]["stn"], color=colors["stn"], opacity=1)

for index, row in sweet_spot_voxels.iterrows():
    plotter.add_mesh(pv.Sphere(center=(row.x, row.y, row.z), radius=0.05), color="r", opacity=1)

for index, row in sour_spot_voxels.iterrows():
    plotter.add_mesh(pv.Sphere(center=(row.x, row.y, row.z), radius=0.05), color="b", opacity=1)
    
if(hemisphere=="right"):
    if(plane=="xy")   : plotter.camera_position = (0.0, 0.0, 1.0)
    elif(plane=="yz") : plotter.camera_position = (-1.0, 0.0, 0.0)
    elif(plane=="xz") : plotter.camera_position = (0.0, -1.0, 0.0)

elif(hemisphere=="left"):
    if(plane=="xy")   : plotter.camera_position = (0.0, 0.0, 1.0)
    elif(plane=="yz") : plotter.camera_position = (1.0, 0.0, 0.0)
    elif(plane=="xz") : plotter.camera_position = (0.0, -1.0, 0.0)
    
plotter.set_background("white")
plotter.show()
