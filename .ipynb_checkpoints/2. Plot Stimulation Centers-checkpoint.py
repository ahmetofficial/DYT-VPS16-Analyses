import pandas as pd
import numpy as np
import pyvista as pv
import glob
import pickle

import utils

# load patient dataframe
with open('stimulation_centers.pickle', 'rb') as handle:
    stimulation_centers = pickle.load(handle)
    
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

# color codes for basal ganglia nuclei
colors        = {}
colors["stn"] = "sandybrown"
colors["gpi"] = "lightgreen"
colors["gpe"] = "turquoise"

# load basal ganglia nuclei meshes
GP_meshes     = utils.load_GP_meshes()
STN_meshes    = utils.load_STN_meshes()


# Plot the cortex mesh with the corresponding scalars and alpha values
plotter = pv.Plotter()
plotter.add_mesh(GP_meshes["right"]["gpi"], color=colors["gpi"], opacity=0.15)
#plotter.add_mesh(GP_meshes["left"]["gpi"], color=colors["gpi"], opacity=0.5)
plotter.add_mesh(GP_meshes["right"]["gpe"], color=colors["gpe"], opacity=0.10)
#plotter.add_mesh(GP_meshes["left"]["gpe"], color=colors["gpe"], opacity=0.1)
plotter.add_mesh(STN_meshes["right"]["stn"], color=colors["stn"], opacity=1)
#plotter.add_mesh(STN_meshes["left"]["stn"], color=colors["stn"], opacity=0.5)

for patient_id in stimulation_centers.keys():
    for hemisphere in stimulation_centers[patient_id].keys():
        if((patient_responsiveness[patient_id]=="hyper") & (stimulation_centers[patient_id][hemisphere] is not None)):
            plotter.add_mesh(pv.Sphere(center=stimulation_centers[patient_id][hemisphere], radius=0.30), color="olivedrab", opacity=1)
        elif((patient_responsiveness[patient_id]=="high") & (stimulation_centers[patient_id][hemisphere] is not None)):
            plotter.add_mesh(pv.Sphere(center=stimulation_centers[patient_id][hemisphere], radius=0.30), color="gold", opacity=1)
        elif((patient_responsiveness[patient_id]=="no") & (stimulation_centers[patient_id][hemisphere] is not None)):
            plotter.add_mesh(pv.Sphere(center=stimulation_centers[patient_id][hemisphere], radius=0.30), color="orangered", opacity=1)


plotter.background_color = "white"
plotter.add_axes(line_width=5, labels_off=True)
plotter.view_yz()
plotter.add_light(pv.Light(position=(0, -1, 0), color='white', intensity=0.35))
# Set the camera position for viewing along the XZ plane

plotter.show()