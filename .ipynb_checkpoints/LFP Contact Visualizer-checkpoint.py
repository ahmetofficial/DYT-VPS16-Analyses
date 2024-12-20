import pandas as pd
import numpy as np
import pyvista as pv

import utils

pallidum_meshes = utils.load_pallidal_meshes()

plotter = pv.Plotter()

# Plot the cortex mesh with the corresponding scalars and alpha values
plotter.add_mesh(pallidum_meshes["right"]["gpi"], color='dimgray', opacity=0.05, specular=10, specular_power=50)
plotter.add_mesh(pallidum_meshes["left"]["gpi"], color='white', opacity=0.05, specular=10, specular_power=50)
plotter.background_color = "white"
plotter.add_axes(line_width=5, labels_off=True)

plotter.view_xz()
plotter.add_light(pv.Light(position=(0, -1, 0), color='white', intensity=0.5))
plotter.show()