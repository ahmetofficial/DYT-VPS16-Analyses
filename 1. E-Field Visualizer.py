import nibabel as nib
import pyvista as pv
import numpy as np

import utils_VAT

# load basal ganglia nuclei meshes
GP_meshes       = utils_VAT.load_GP_meshes()
STN_meshes      = utils_VAT.load_STN_meshes()

# color codes for basal ganglia nuclei
colors          = {}
colors["stn"]   = "sandybrown"
colors["gpi"]   = "lightgreen"
colors["gpe"]   = "turquoise"

patient         = "P2"
VAT_R, efield_R = utils_VAT.load_VAT_for_patients(path="data_VAT/"+patient+"/vat_efield_right.nii")
VAT_L, efield_L = utils_VAT.load_VAT_for_patients(path="data_VAT/"+patient+"/vat_efield_left.nii")

plotter = pv.Plotter()
plotter.add_mesh(GP_meshes["right"]["gpi"], color=colors["gpi"], opacity=0.2)
plotter.add_mesh(GP_meshes["left"]["gpi"], color=colors["gpi"], opacity=0.2)
plotter.add_mesh(GP_meshes["right"]["gpe"], color=colors["gpe"], opacity=0.2)
plotter.add_mesh(GP_meshes["left"]["gpe"], color=colors["gpe"], opacity=0.2)
plotter.add_mesh(VAT_R, scalars="e_field", cmap="hot", opacity="linear", clim=[150, 1000])
plotter.add_mesh(VAT_L, scalars="e_field", cmap="hot", opacity="linear", clim=[150, 1000])
plotter.show()