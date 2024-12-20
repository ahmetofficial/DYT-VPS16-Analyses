import pandas as pd
import pickle
import pyvista as pv 
import numpy as np
import math
import nibabel as nib
from scipy.stats import spearmanr
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

def load_GP_meshes():
    pallidum_meshes                 = {}
    pallidum_meshes["left"]         = {}
    pallidum_meshes["right"]        = {}
    pallidum_meshes["right"]["gpi"] = pv.read("DISTAL Atlas/gpi_right.vtk")
    pallidum_meshes["right"]["gpe"] = pv.read("DISTAL Atlas/gpe_right.vtk")
    pallidum_meshes["left"]["gpi"]  = pv.read("DISTAL Atlas/gpi_left.vtk")
    pallidum_meshes["left"]["gpe"]  = pv.read("DISTAL Atlas/gpe_left.vtk")
    return pallidum_meshes

def load_STN_meshes():
    subthalamic_meshes                 = {}
    subthalamic_meshes["left"]         = {}
    subthalamic_meshes["right"]        = {}
    subthalamic_meshes["right"]["stn"] = pv.read("DISTAL Atlas/stn_right.vtk")
    subthalamic_meshes["left"]["stn"]  = pv.read("DISTAL Atlas/stn_left.vtk")
    return subthalamic_meshes


def load_efields_for_patients(path):
    
    file                         = nib.load(path)
    data                         = file.get_fdata()      # extract the E-field values and affine matrix
    affine                       = file.affine
    shape                        = data.shape            # get the shape of the data (dimensions)
    x, y, z                      = np.indices(shape)     # create a grid of indices (x, y, z) for the voxel locations

    indices                      = np.vstack((x.ravel(), y.ravel(), z.ravel()))
    coordinates                  = np.dot(affine[:3, :3], indices) + affine[:3, 3, np.newaxis]

    points                       = coordinates.T         # Transpose to get (num_voxels, 3)
    mesh                         = pv.PolyData(points)   # create a PyVista PolyData object

    # Assign the E-field values as point scalars (each point gets an E-field value)
    e_field                      = data.ravel()
    e_field[e_field<150]         = 0
    mesh.point_arrays['e_field'] = e_field
    
    field                        = dict()
    field["coordinates"]         = points
    field["vector_magnitude"]    = e_field
    
    return mesh, field

def assign_efields_to_voxels(dataset, voxel_size=1):

    if voxel_size <= 0:
        raise ValueError("Voxel size must be greater than 0")
    
    # assign voxel coordinates based on the floor to the nearest voxel size
    dataset['x_voxel'] = (np.floor(dataset['x'] / voxel_size) * voxel_size).astype(float)
    dataset['y_voxel'] = (np.floor(dataset['y'] / voxel_size) * voxel_size).astype(float)
    dataset['z_voxel'] = (np.floor(dataset['z'] / voxel_size) * voxel_size).astype(float)
    
    return dataset

def aggregate_voxels(dataset):
    
    # aggregates the DataFrame to merge rows with identical voxel coordinates 
    # and hemisphere for each patient, taking the mean of vector magnitude of e-field

    # group by patient, hemisphere, and voxel coordinates, then take the mean of vector_magnitude
    aggregated_dataset = (dataset.groupby(['patient', 'hemisphere', 'x_voxel', 'y_voxel', 'z_voxel'], as_index=False).agg({'vector_magnitude': 'mean'}))
    
    return aggregated_dataset

def calculate_spearman_correlation(dataset):
    
    # group by the unique combinations of hemisphere and voxel coordinates
    grouped = dataset.groupby(['hemisphere', 'x_voxel', 'y_voxel', 'z_voxel'])
    results = []
    
    for group, data in grouped:
        
        correlation, pvalue = spearmanr(data['vector_magnitude'], data['BFMDRS_M_improvement'])
        results.append({'hemisphere'   : group[0],
                        'x_voxel'      : group[1],
                        'y_voxel'      : group[2],
                        'z_voxel'      : group[3],
                        'correlation'  : correlation,
                        'pvalue'       : pvalue,
                        'patient_count': len(data)})
    
    # convert the results to a DataFrame
    results_df = pd.DataFrame(results)
    return results_df

def create_3D_grid_for_correlation_values(dataset, hemisphere):
    
    # create a structured grid for the voxel data
    x           = dataset[dataset.hemisphere==hemisphere]['x_voxel'].values
    y           = dataset[dataset.hemisphere==hemisphere]['y_voxel'].values
    z           = dataset[dataset.hemisphere==hemisphere]['z_voxel'].values
    correlation = dataset[dataset.hemisphere==hemisphere]['correlation'].values

    # convert voxel corners to centers
    x_centers = x + 0.5
    y_centers = y + 0.5
    z_centers = z + 0.5

    # create a PyVista structured grid
    grid = pv.PolyData(np.c_[x_centers, y_centers, z_centers])
    grid['correlation'] = correlation  # add correlation as a scalar field

    return grid

def create_2D_plane_for_correlation_values(dataset, hemisphere, plane_axis, constant):

    dataset = dataset[dataset.hemisphere==hemisphere].copy()
    
    if(plane_axis == 'xy'):
        # group by X-Y and average correlation across Z axis
        grouped = dataset.groupby(['x_voxel', 'y_voxel']).correlation.mean().reset_index()
        # grouped = grouped[grouped.correlation.abs()>=0.2]
        plane_z = constant  # Set Z to a constant value for the X-Y plane
        x_vals  = grouped['x_voxel'].values
        y_vals  = grouped['y_voxel'].values
        z_vals  = np.full(len(grouped), plane_z)
        scalars = grouped['correlation'].values
        
    elif(plane_axis == 'yz'):
        # group by Y-Z and average correlation across X axis
        grouped = dataset.groupby(['y_voxel', 'z_voxel']).correlation.mean().reset_index()
        # grouped = grouped[grouped.correlation.abs()>=0.2]
        plane_x = constant  # Set X to a constant value for the Y-Z plane
        x_vals  = np.full(len(grouped), plane_x)
        y_vals  = grouped['y_voxel'].values
        z_vals  = grouped['z_voxel'].values
        scalars = grouped['correlation'].values
        
    elif(plane_axis == 'xz'):
        # group by X-Z and average correlation across Y axis
        grouped = dataset.groupby(['x_voxel', 'z_voxel']).correlation.mean().reset_index()
        # grouped = grouped[grouped.correlation.abs()>=0.2]
        plane_y = constant  # Set Y to a constant value for the X-Z plane
        x_vals  = grouped['x_voxel'].values
        y_vals  = np.full(len(grouped), plane_y)
        z_vals  = grouped['z_voxel'].values
        scalars = grouped['correlation'].values
    
    else:
        raise ValueError("Invalid plane_axis. Choose from 'yz', 'xz', or 'xy'.")

    # create a PyVista StructuredGrid
    grid = pv.PolyData(np.c_[x_vals, y_vals, z_vals])
    grid['correlation'] = scalars
    return grid


def extract_plane_correlation(data_corr, plane_x, plane_y, plane_value=np.nan):

    # extract axial view
    if((plane_x == "x") & (plane_y == "y")):
        
        if(np.isnan(plane_value)):
            R_plane = data_corr[(data_corr.hemisphere=="right") & (data_corr.correlation.abs()>=0.2)]
            R_plane = R_plane.pivot_table(index="y_voxel", columns="x_voxel", values='correlation', aggfunc='mean')
            R_plane = R_plane.sort_index(ascending=False)
            L_plane = data_corr[(data_corr.hemisphere=="left") & (data_corr.correlation.abs()>=0.2)]
            L_plane = L_plane.pivot_table(index="y_voxel", columns="x_voxel", values='correlation', aggfunc='mean')
            L_plane = L_plane.sort_index(ascending=False)
        else:
            R_plane = data_corr[(data_corr.hemisphere=="right") & (data_corr.z_voxel==plane_value) & (data_corr.correlation.abs()>=0.2)]
            R_plane = R_plane.pivot_table(index="y_voxel", columns="x_voxel", values='correlation', aggfunc='mean')
            R_plane = R_plane.sort_index(ascending=False)
            L_plane = data_corr[(data_corr.hemisphere=="left") & (data_corr.z_voxel==plane_value) & (data_corr.correlation.abs()>=0.2)]
            L_plane = L_plane.pivot_table(index="y_voxel", columns="x_voxel", values='correlation', aggfunc='mean')
            L_plane = L_plane.sort_index(ascending=False)

        # to align voxels for both hemispheres
        x_range = sorted(set(np.array(R_plane.columns)*-1).union(L_plane.columns))
        y_range = sorted(set(R_plane.index).union(L_plane.index))

        # reindex both dataframes with the full range of x_voxel and y_voxel
        L_plane = L_plane.reindex(index=y_range, columns=x_range)
        R_plane = R_plane.reindex(index=y_range, columns=sorted(np.array(x_range)*-1))
        
        L_plane = L_plane.sort_index(ascending=False)
        R_plane = R_plane.sort_index(ascending=False)

    # extract coronal view
    elif((plane_x == "x") & (plane_y == "z")):
        
        if(np.isnan(plane_value)):
            R_plane = data_corr[(data_corr.hemisphere=="right") & (data_corr.correlation.abs()>=0.2)]
            R_plane = R_plane.pivot_table(index="z_voxel", columns="x_voxel", values='correlation', aggfunc='mean')
            R_plane = R_plane.sort_index(ascending=False)
            L_plane = data_corr[(data_corr.hemisphere=="left") & (data_corr.correlation.abs()>=0.2)]
            L_plane = L_plane.pivot_table(index="z_voxel", columns="x_voxel", values='correlation', aggfunc='mean')
            L_plane = L_plane.sort_index(ascending=False)
        else:
            R_plane = data_corr[(data_corr.hemisphere=="right") & (data_corr.y_voxel==plane_value) & (data_corr.correlation.abs()>=0.2)]
            R_plane = R_plane.pivot_table(index="z_voxel", columns="x_voxel", values='correlation', aggfunc='mean')
            R_plane = R_plane.sort_index(ascending=False)
            L_plane = data_corr[(data_corr.hemisphere=="left") & (data_corr.y_voxel==plane_value) & (data_corr.correlation.abs()>=0.2)]
            L_plane = L_plane.pivot_table(index="z_voxel", columns="x_voxel", values='correlation', aggfunc='mean')
            L_plane = L_plane.sort_index(ascending=False)

        # to align voxels for both hemispheres
        x_range = sorted(set(np.array(R_plane.columns)*-1).union(L_plane.columns))
        y_range = sorted(set(R_plane.index).union(L_plane.index))

        # reindex both dataframes with the full range of x_voxel and y_voxel
        L_plane = L_plane.reindex(index=y_range, columns=x_range)
        R_plane = R_plane.reindex(index=y_range, columns=sorted(np.array(x_range)*-1))
        
        L_plane = L_plane.sort_index(ascending=False)
        R_plane = R_plane.sort_index(ascending=False)

    # extract sagittal view
    elif((plane_x == "y") & (plane_y == "z")):
        
        if(np.isnan(plane_value)):
            R_plane = data_corr[(data_corr.hemisphere=="right")]
            R_plane = R_plane.pivot_table(index="z_voxel", columns="y_voxel", values='correlation', aggfunc='mean')
            R_plane = R_plane.sort_index(ascending=False)
            L_plane = data_corr[(data_corr.hemisphere=="left")]
            L_plane = L_plane.pivot_table(index="z_voxel", columns="y_voxel", values='correlation', aggfunc='mean')
            L_plane = L_plane.sort_index(ascending=False)
        else:
            R_plane = data_corr[(data_corr.hemisphere=="right") & (data_corr.x_voxel==plane_value)]
            R_plane = R_plane.pivot_table(index="z_voxel", columns="y_voxel", values='correlation', aggfunc='mean')
            R_plane = R_plane.sort_index(ascending=False)
            L_plane = data_corr[(data_corr.hemisphere=="left") & (data_corr.x_voxel==plane_value)]
            L_plane = L_plane.pivot_table(index="z_voxel", columns="y_voxel", values='correlation', aggfunc='mean')
            L_plane = L_plane.sort_index(ascending=False)

        # to align voxels for both hemispheres
        x_range = sorted(set(R_plane.columns).union(L_plane.columns))
        y_range = sorted(set(R_plane.index).union(L_plane.index))

        # reindex both dataframes with the full range of x_voxel and y_voxel
        L_plane = L_plane.reindex(index=y_range, columns=x_range)
        R_plane = R_plane.reindex(index=y_range, columns=x_range)
        
        L_plane = L_plane.sort_index(ascending=False)
        R_plane = R_plane.sort_index(ascending=False)

        #L_plane = L_plane.mask(L_plane.abs() < 0.2, np.nan)
        #R_plane = R_plane.mask(R_plane.abs() < 0.2, np.nan)

    return R_plane, L_plane


def extract_significant_voxels_for_sweet_and_sour_regions(dataset, hemisphere, voxel_size):

    sweet_spots      = dataset[(dataset.hemisphere==hemisphere) & (dataset.correlation>=0) & (dataset.pvalue<=0.05)]
    # convert voxel edges to voxel centers
    sweet_spots["x"] = sweet_spots["x_voxel"] + voxel_size/2
    sweet_spots["y"] = sweet_spots["y_voxel"] + voxel_size/2
    sweet_spots["z"] = sweet_spots["z_voxel"] + voxel_size/2
    sweet_spots.reset_index(drop=True, inplace=True)



    sour_spots       = dataset[(dataset.hemisphere==hemisphere) & (dataset.correlation<0) & (dataset.pvalue<=0.05)]
    # convert voxel edges to voxel centers
    sour_spots["x"]  = sour_spots["x_voxel"] + voxel_size/2
    sour_spots["y"]  = sour_spots["y_voxel"] + voxel_size/2
    sour_spots["z"]  = sour_spots["z_voxel"] + voxel_size/2
    sour_spots.reset_index(drop=True, inplace=True)
    
    return sweet_spots, sour_spots
    
def define_3D_ROI(data, voxel_size):
    
    points          = data[['x', 'y', 'z']].values
    # create a PyVista PolyData object from voxel centers (sweet or sour based on passed dataframe)
    point_cloud     = pv.PolyData(points)
    
    # define a 3D grid over the boundaries of  ROI
    bounds          = point_cloud.bounds  
    
    bounds          = (bounds[0] - 10, bounds[1] + 10, bounds[2] - 10, bounds[3] + 10, bounds[4] - 10, bounds[5] + 10)
    
    spacing         = (voxel_size, voxel_size, voxel_size)   # grid resolution equal to voxel size
    grid            = pv.UniformGrid()
    grid.dimensions = (int((bounds[1] - bounds[0]) / spacing[0]),
                       int((bounds[3] - bounds[2]) / spacing[1]),
                       int((bounds[5] - bounds[4]) / spacing[2]))
    grid.origin     = (bounds[0], bounds[2], bounds[4])
    grid.spacing    = spacing
    cell_centers    = grid.cell_centers().points
    
    # calculate 3D point density using a KDTree
    tree            = cKDTree(points)
    radius          = 2.0  # radius parameter for the algorithm
    density_3D      = np.array([len(tree.query_ball_point(p, radius)) for p in cell_centers])
    
    # normalize the density to range [0, 1] (Standard Scaling)
    density_min     = density_3D.min()
    density_max     = density_3D.max()
    density_3D_norm = (density_3D - density_min) / (density_max - density_min)
    
    # add the normalized density field to the grid as cell data
    grid.cell_data["density"] = density_3D_norm
    
    # convert cell data to point data
    grid = grid.cell_data_to_point_data()
    
    # apply Gaussian smoothing to the point data
    smooth_grid = grid.gaussian_smooth(scalars="density", radius_factor=1)
    
    # extract an isosurface (contour) from the smoothed grid
    isosurface = smooth_grid.contour(isosurfaces=10, scalars="density")

    return isosurface
