import pandas as pd
import pickle
import pyvista as pv 
import numpy as np
import math

def load_GP_meshes():
    pallidum_meshes                 = {}
    pallidum_meshes["left"]         = {}
    pallidum_meshes["right"]        = {}
    pallidum_meshes["right"]["gpi"] = pv.read("atlases/DISTAL Atlas/gpi_right.vtk")
    pallidum_meshes["right"]["gpe"] = pv.read("atlases/DISTAL Atlas/gpe_right.vtk")
    pallidum_meshes["left"]["gpi"]  = pv.read("atlases/DISTAL Atlas/gpi_left.vtk")
    pallidum_meshes["left"]["gpe"]  = pv.read("atlases/DISTAL Atlas/gpe_left.vtk")
    return pallidum_meshes

def load_STN_meshes():
    pallidum_meshes                 = {}
    pallidum_meshes["left"]         = {}
    pallidum_meshes["right"]        = {}
    pallidum_meshes["right"]["stn"] = pv.read("atlases/DISTAL Atlas/stn_right.vtk")
    pallidum_meshes["left"]["stn"]  = pv.read("atlases/DISTAL Atlas/stn_left.vtk")
    return pallidum_meshes

def dict_to_dataframe(patient_id, data):
    # Using a list comprehension to flatten the nested dictionary
    rows = [{'hemisphere': hemisphere, 'contact': contact, **coords}
            for hemisphere, contacts in data.items()
            for contact, coords in contacts.items()]

    df               = pd.DataFrame(rows)
    df["patient_id"] = patient_id
    df               = df[["patient_id","hemisphere","contact","x","y","z"]]
    return df

def read_2D_coordinate_file_into_dataframe(patient_path):

    patient_id = patient_path.split("\\")[1].split(".")[0]
    
    # read the txt. file
    with open(patient_path) as file:
        lines = [line.rstrip() for line in file]
    
    data          = {}
    data["right"] = {}
    data["left"]  = {}
     
    for line in lines:
        contact = line.split(",")[0].split(" ")[1]
        axis    = line.split(",")[1].split(" = ")[0].split(": ")[1]
        value   = float(line.split(",")[1].split(" = ")[1].split(" mm.")[0])
        if(contact in ["k0","k1","k2","k3"]):
           if(contact not in data["right"].keys()):
               data["right"][contact]       = {}
               data["right"][contact][axis] = value
           else:
               data["right"][contact][axis] = value
        else:
           if(contact not in data["left"].keys()):
               data["left"][contact]       = {}
               data["left"][contact][axis] = value
           else:
               data["left"][contact][axis] = value
        
    return dict_to_dataframe(patient_id, data)   

# Function to get the stimulation voltage for each contact
def get_stimulation_voltage(row):
    with open('data_DBS_Stimulation/stimulation_parameters.pickle', 'rb') as handle:
        stimulation_voltages = pickle.load(handle)
    patient = row['patient_id']
    contact = row['contact']
    return stimulation_voltages.get(patient, {}).get(contact, None)  # Returns None if patient or contact is not found

def calculate_electric_field(df, grid_size=50, search_range=1):
    
    k = 8.99e9  # Coulomb's constant in N m^2/C^2

    # Prepare to find the maximum electric field
    max_field = 0
    max_point = None

    # Create a grid of points to search
    x_range = np.linspace(df['x'].min()-1 - search_range, df['x'].max()+1 + search_range, grid_size)
    y_range = np.linspace(df['y'].min()-1 - search_range, df['y'].max()+1 + search_range, grid_size)
    z_range = np.linspace(df['z'].min()-1 - search_range, df['z'].max()+1 + search_range, grid_size)

    # iterate through all points in the grid (in x,y,z, axes)
    for x in x_range:
        for y in y_range:
            for z in z_range:
                E_net = np.array([0.0, 0.0, 0.0])  # initialize net electric field vector
                
                # Calculate electric field contributions from each point in the DataFrame
                for index, row in df.iterrows():
                    voltage = row['stimulation_voltage']
                    contact_pos = np.array([row['x'], row['y'], row['z']])
                    r_vec = np.array([x, y, z]) - contact_pos  # Vector from charge to the point
                    r_mag = np.linalg.norm(r_vec)  # Magnitude of the distance
                    if r_mag > 0:  # Prevent division by zero
                        E_contribution = k * voltage * r_vec / r_mag**3  # Electric field contribution
                        E_net += E_contribution  # Sum up the contributions
                
                # Calculate the magnitude of the net electric field
                E_magnitude = np.linalg.norm(E_net)
                
                # Update maximum electric field and corresponding point
                if E_magnitude > max_field:
                    max_field = E_magnitude
                    max_point = (x, y, z)

    return max_point


def transform_data(data):
    data["is_bursting"]            = data.spike_pattern == "bursting"
    data["is_bursting"]            = data["is_bursting"].astype(int)
    data["is_tonic"]               = data.spike_pattern == "tonic"
    data["is_tonic"]               = data["is_tonic"].astype(int)
    data["is_irregular"]           = data.spike_pattern == "irregular"
    data["is_irregular"]           = data["is_irregular"].astype(int)
    data["delta_band_oscillatory"] = ~np.isnan(data.oscillation_frequency_delta_band)
    data["delta_band_oscillatory"] = data["delta_band_oscillatory"].astype(int)
    data["theta_band_oscillatory"] = ~np.isnan(data.oscillation_frequency_theta_band)
    data["theta_band_oscillatory"] = data["theta_band_oscillatory"].astype(int)
    data["alpha_band_oscillatory"] = ~np.isnan(data.oscillation_frequency_alpha_band)
    data["alpha_band_oscillatory"] = data["alpha_band_oscillatory"].astype(int)
    data["beta_band_oscillatory"]  = ~np.isnan(data.oscillation_frequency_beta_band)
    data["beta_band_oscillatory"]  = data["beta_band_oscillatory"].astype(int)
    data["gamma_band_oscillatory"] = ~np.isnan(data.oscillation_frequency_gamma_band)

    data["is_oscillatory"]         = data.delta_band_oscillatory  + data.theta_band_oscillatory + data.alpha_band_oscillatory + data.beta_band_oscillatory + data.gamma_band_oscillatory
    data.is_oscillatory[data.is_oscillatory != 0] = 1

    data.drop(columns = ['coherence_delta_band','coherence_frequency_delta_band',
                                      'coherence_theta_band','coherence_frequency_theta_band',
                                      'coherence_alpha_band','coherence_frequency_alpha_band',
                                      'coherence_beta_band','coherence_frequency_beta_band',
                                      'coherence_gamma_band','coherence_frequency_gamma_band'], inplace=True)

    data.drop(columns = ['mean_power_delta_band', 'mean_power_theta_band',
                                       'mean_power_alpha_band', 'mean_power_beta_band',
                                       'mean_power_gamma_band', 'mean_power_spectrum'], inplace=True)

    data.columns                = data.columns.str.replace('bspike_proportion', 'burst_spike_proportion')
    data.columns                = data.columns.str.replace('burst_int', 'burst_duration')
    data.columns                = data.columns.str.replace('burst_avg_spikes', 'burst_average_spikes')

    data.burst_count            = data.burst_count.astype(float)
    data.pause_count            = data.pause_count.astype(float)

    data.is_bursting            = data.is_bursting.astype(float)
    data.is_tonic               = data.is_tonic.astype(float)
    data.is_irregular           = data.is_irregular.astype(float)
    data.delta_band_oscillatory = data.delta_band_oscillatory.astype(float)
    data.theta_band_oscillatory = data.theta_band_oscillatory.astype(float)
    data.alpha_band_oscillatory = data.alpha_band_oscillatory.astype(float)
    data.beta_band_oscillatory  = data.beta_band_oscillatory.astype(float)
    data.gamma_band_oscillatory = data.gamma_band_oscillatory.astype(float)
    return data

def oscillatory_characteristics(dataset, group_variable, group_value):
    osc_delta  = np.sum(dataset[dataset[group_variable] == group_value][['delta_band_oscillatory']].sum(axis=1) == 1)
    osc_theta  = np.sum(dataset[dataset[group_variable] == group_value][['theta_band_oscillatory']].sum(axis=1) == 1)
    osc_alpha  = np.sum(dataset[dataset[group_variable] == group_value][['alpha_band_oscillatory']].sum(axis=1) == 1)
    osc_beta   = np.sum(dataset[dataset[group_variable] == group_value][['beta_band_oscillatory']].sum(axis=1) == 1)
    osc_gamma  = np.sum(dataset[dataset[group_variable] == group_value][['gamma_band_oscillatory']].sum(axis=1) == 1)
    osc_non    = np.sum(dataset[dataset[group_variable] == group_value][['delta_band_oscillatory', 'theta_band_oscillatory','alpha_band_oscillatory', 'beta_band_oscillatory','gamma_band_oscillatory']].sum(axis=1) == 0)
    
    oscillations = [osc_delta, osc_theta, osc_alpha, osc_beta, osc_gamma, osc_non]
    oscillations = [ x / len(dataset[dataset[group_variable] == group_value]) *100 for x in oscillations]
    return oscillations