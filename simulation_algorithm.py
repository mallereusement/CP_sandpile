import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
from scipy.stats import linregress, t
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm


def detect_steady_state(data, window_size=20, confidence_level=0.99):
    """
    Detects the steady state in the given data by fitting a line to a window of data
    and considering the uncertainty of the slope.

    Parameters:
        data (array-like): Input data array.
        window_size (int): Size of the window for fitting the line.
        confidence_level (float): Confidence level for determining the uncertainty of the slope.

    Returns:
        int or None: Index at which steady state is reached, or None if not reached.
    """
    for i in range(len(data) - window_size + 1):
        window_data = data[i:i + window_size]
        x = np.arange(window_size)
        slope, _, std_err_slope, _, _ = linregress(x, window_data)

        # Calculate critical t-value for the given confidence level and degrees of freedom (window_size - 2)
        t_critical = np.abs(t.ppf((1 + confidence_level) / 2, window_size - 2))

        # Calculate margin of error for the slope
        margin_of_error = t_critical * std_err_slope

        if np.abs(slope) < std_err_slope:
            return True  # Steady state reached at this index

    return False  # Steady state not reached within the given data

def set_bound_0(grid: np.ndarray, N: int, boundary: str='open', d: int=2) -> np.ndarray:
    """Set all borders to 0 as per the specified boundary condition.

    Args:
        grid (np.ndarray): Current grid.
        N (int): Size of the grid.
        boundary (str, optional): Type of boundary condition used. Defaults to 'open'.
                                  It can take values 'open' and 'closed'.
        d (int, optional): Dimension of the grid. Defaults to 2.

    Returns:
        np.ndarray: Grid with the boundary condition applied (with borders set to 0).
    """
    # Set borders to 0 for each axis
    for axis in range(d):
        slices = [slice(None)] * d
        slices[axis] = 0
        grid[tuple(slices)] = 0
    # If boundary is 'closed', set the opposite borders to 0 as well
    if boundary == 'closed':
        for axis in range(d):
            slices = [slice(None)] * d
            slices[axis] = N-1
            grid[tuple(slices)] = 0
    return grid

def generate_unit_vecs(d:int) -> list:
    """Generate unit vectors in d-dimensional space.

    Args:
        d (int): Dimension of the space.

    Returns:
        list: List of unit vectors in the specified dimension.
    """
    # Create an empty vector
    empty = np.zeros(d, dtype=int)
    # Initialize a list to store unit vectors
    vectors = []
    # Generate unit vectors along each axis
    for i in range(d):
        vec = copy.copy(empty)
        # Set the current axis to 1 to generate the unit vector
        vec[i] = 1
        vectors.append(vec)
    return vectors
           
def set_up_grid(N: int, d: int=2) -> np.ndarray:
    """Generates a grid with zeros.

    Args:
        N (int): Size of the grid in one axis.
        d (int, optional): Dimension of the grid. Defaults to 2.

    Returns:
        np.ndarray: Grid initialized with zeros.
    """
    shape = tuple([N] * d)
    return np.zeros(shape=shape)

def pertubation_mech(grid: np.ndarray, N: int, p: np.ndarray, d:int=2, type: str = 'conservative', boundary_condition: str = 'open') -> np.ndarray:  
    """Applies a perturbation to a specific point in the grid.

    Args:
        grid (np.ndarray): Current grid.
        N (int): Size of the grid.
        p (np.ndarray): Position of the perturbation.
        d (int, optional): Dimension of the grid. Defaults to 2.
        type (str, optional): Type of perturbation, takes 'conservative' or 'non_conservative'. Defaults to 'conservative'.
        boundary_condition (str, optional): Type of boundary condition used, takes 'open' and 'closed'. Defaults to 'open'.

    Returns:
        np.ndarray: Grid with the perturbation applied.
    """
    # Randomly set position if not provided
    p[p==False] = np.random.randint(1,N)

    if type == 'conservative':
        # Apply conservative perturbation
        grid[tuple(p)] += d 
        vecs = generate_unit_vecs(d)
        for i in range(d):
            grid[tuple(p-vecs[i])] -= 1
              
    if type == 'non_conservative':
        # Apply non-conservative perturbation
        grid[tuple(p)] += 1

    # Apply boundary condition
    grid = set_bound_0(grid, N, boundary_condition, d=d)
    return grid

def relax(grid, N, crit_val, boundary_condition='open', use_abs_val=False, d:int=2) -> np.ndarray:   
    """Applies relaxation to the grid.

    Args:
        grid (np.ndarray): Current grid.
        N (int): Size of the grid.
        crit_val (int): Value of the critical slope.
        boundary_condition (str, optional): Type of boundary condition used, takes 'open' and 'closed'. Defaults to 'open'.
        use_abs_val (bool, optional): Determines whether the condition |z| > z_crit or z > z_crit is used. Defaults to False (for the latter case).
        d (int, optional): Dimension of the grid. Defaults to 2.

    Returns:
        np.ndarray: Relaxed grid.
    """
    # Set boundary condition
    grid = set_bound_0(grid, N, boundary_condition, d)

    # Determine mask based on use_abs_val
    if use_abs_val:
        mask = np.where(abs(grid) > crit_val)
    else:
        mask = np.where(grid > crit_val)

    # Apply relaxation to each point in the mask
    for i in range(len(mask[0])):   
        vecs = generate_unit_vecs(d)
        p = np.array(mask)[:, i]

        # |z| > z_crit
        if use_abs_val:
            if grid[tuple(p)] > 0:
                grid[tuple(p)] += -2*d + sum([1 if mask[j][i] == N-1 else 0 for j in range(0,d)])
                
                for j in range(d):
                    
                    if p[j] < N-1:
                        grid[tuple(p + vecs[j])] += 1
                    grid[tuple(p - vecs[j])] += 1

            elif grid[tuple(p)] < 0:
                grid[tuple(p)] += 2*d - sum([1 if mask[j][i] == N-1 else 0 for j in range(0,d)])
                
                for j in range(d):
                    if p[j] < N-1:
                        grid[tuple(p + vecs[j])] -= 1
                    grid[tuple(p - vecs[j])] -= 1   

        # z > z_crit
        else:
            if grid[tuple(p)] > 0:
                grid[tuple(p)] += -2*d + sum([1 if mask[j][i] == N-1 else 0 for j in range(0,d)])
                
                for j in range(d):
                    if p[j] < N-1:
                        grid[tuple(p + vecs[j])] += 1
                    grid[tuple(p - vecs[j])] += 1         
    # Set boundary condition again and return the relaxed grid
    grid = set_bound_0(grid, N, boundary_condition, d)
    return grid

def spatial_linear_distance(crit_grid, p:tuple, d:int=2) -> float:
    """Calculates the distance between the point of pertubation and the point furthest away but still reached by the avalanche.

    Args:
        crit_grid (np.ndarray): Grid indicating critical sites.
        p (tuple): Coordinates of the point.
        d (int, optional): Dimension of the grid. Defaults to 2.

    Returns:
        float: Maximum spatial linear distance
    """
    # Find coordinates of critical sites
    truth = np.where(crit_grid >= 1)
    
    dist = []
    # Calculate distances to each critical site
    for i in range(len(truth[0])):
        pos = np.array(truth)[:, i]
        dist.append(np.sqrt(np.dot(p-pos, p-pos)))
    # Return the maximum distance
    return np.max(dist)


def write_data_for_power_spectrum_to_file(filepath: str, filename: str, list_of_avalanches :list) -> None:
    """Write avalanche data to a text file for power spectrum calculation.

    Args:
        filepath (str): Path to the directory where the file will be saved.
        filename (str): Name of the file to be saved.
        list_of_avalanches (list): List of avalanche data to be written to the file.
    """
    with open(f'{filepath}/{filename}.txt', 'w') as f:
        for line in list_of_avalanches:
            f.write(f'{line} \n')

def write_data_for_exponent_calculation_to_file(filepath: str, filename: str, data: dict) -> None:
    """Write data for exponent calculation to a CSV file.

    Args:
        filepath (str): Path to the directory where the file will be saved.
        filename (str): Name of the file to be saved.
        data (dict): Dictionary containing the data to be written to the file.
                     Keys represent column names, and values represent lists of data for each column.
    """
    df = pd.DataFrame(data)
    df.to_csv(f'{filepath}/{filename}.csv', sep=';', encoding='utf8')

def plot_heatmap(crit_grid_sum, type, boundary_condition):
    """Plot heatmap of the summed critical grid.

    Args:
        crit_grid_sum (np.ndarray): Summed critical grid data to be plotted.
        type (str): Type of perturbation applied.
        boundary_condition (str): Type of boundary condition used.
    """
    fig, ax = plt.subplots(figsize=(6,5))
    
    sns.heatmap(crit_grid_sum)#, norm=LogNorm())
    ax.text(0.95, 0.95, f'perturbation = {type} \n boundary = {boundary_condition} ', ha='right', va='top', color='white', transform=ax.transAxes)
    #plt.tight_layout()



def run_simulation(simulation_parameter: dict, filepath_datastorage: str, simulation_name: str, safe_data_for_animations: bool=False):
    """Runs a sandpile simulation based on the provided parameters.

    Args:
        simulation_parameter (dict): Dictionary containing simulation parameters.
        filepath_datastorage (str): Path to the directory where simulation data will be stored.
        simulation_name (str): Name of the simulation.
        safe_data_for_animations (bool): safe data for animations (Default to False)

    Returns:
        None
    """

    ### Set simulation parameter ######################################################
    N = simulation_parameter['size of grid']
    d = simulation_parameter['dimension']
    use_abs_value = simulation_parameter['use absolute value']
    pertubation_mechanism = simulation_parameter['pertubation mechanism']
    boundary_condition = simulation_parameter['boundary condition']
    crit_val = simulation_parameter['crititcal value of z']
    maximum_avalanches = simulation_parameter['number of activated avalanches']
    max_t = simulation_parameter['maximum time steps']
    track_after_steady_state = simulation_parameter['track avalanches after steady state']
    if track_after_steady_state: # Start data acquisition after steady state is reached 
        mean_tracker = []
    steady_state = simulation_parameter['steady state'] # Point where steady state is reached
    if steady_state > 0:
        track_after_steady_state = True
    ###################################################################################

    ### Initialize storage for power spectrum data and/or exponent calculation data and/or mean data ###
    if simulation_parameter['save file for exponent calculation']:
        exp_data = {
            'timestep': [],
            'number': [],
            'lifetime': [],
            'total dissipation': [],
            'spatial linear size': []
                    }

    if simulation_parameter['save file for power spectrum calculation']:
        list_avalanches = []

    if simulation_parameter['save mean value of grid']:
        means = {
            'mean': [],
            'time': []
        }
    ####################################################################################
    
    if track_after_steady_state:
        reached_steady_state = False
        pbar1 = tqdm(total = maximum_avalanches, desc =f"Running Simulation {simulation_name}: Steady state not reached")
    else:
        pbar1 = tqdm(total = maximum_avalanches, desc =f"Running Simulation {simulation_name}: Number of activated avalances")
    pbar2 = tqdm(total = max_t, desc =f"Running Simulation {simulation_name}: Number of time steps")
    c = 0
    ### Simulation algorithm ###########################################################

    grid = set_up_grid(N=N, d=d) # Initialize grid with z = 0
    grid_pertubation_tracker = set_up_grid(N, d) # Initialize grid that tracks pertubations
    grid_dissipation_tracker = set_up_grid(N, d)  ## Initialize grid that tracks dissipations
    
    t = 0 # Initialize time tracker
    count_avalanches = 0 # Track number of triggered avalanches

    if safe_data_for_animations: # Only to create animations
        if simulation_parameter['save animation of evolution to steady state']:
            evo_to_steady_state = []
    if safe_data_for_animations: # Only to create animations
        if simulation_parameter['save animation of avalanches']:
            ani_counter = 0
            ava_tracker = []

    while (count_avalanches < maximum_avalanches) and (t < max_t): # Run simulation until maximum time steps or maximum avalanches are reached



        if c < 2:
            if track_after_steady_state:
                if t > steady_state:
                    reached_steady_state = True
                    c = 2
                    pbar1.set_description(f"Running Simulation {simulation_name}: Steady state reached after t={t}, Number of activated avalances")
                    pbar1.refresh()

        random_point = np.random.randint(1, N, d) # Random point where pertubation is applied
        
        grid_pertubation_tracker[tuple(random_point)] +=1 # Track point of pertubation
        
        grid = pertubation_mech(grid, N, type=pertubation_mechanism, p=random_point, d=d, boundary_condition = boundary_condition) # Apply pertubation


        # Only to create animations #####
        if safe_data_for_animations:
            if simulation_parameter['save animation of evolution to steady state']:
                evo_to_steady_state.append(grid.copy())
                if reached_steady_state:
                    evo_to_steady_state = np.array(evo_to_steady_state)
                    np.save(f'{filepath_datastorage}/{simulation_name}/evo_to_steady_state.npy', evo_to_steady_state)
                    break
        #################################
        
        if use_abs_value: # Check either if there are points with z > critical value or |z| > critical value, depends on chooses simulation parameter
            z = abs(grid) > crit_val
        else:
            z = grid > crit_val

        t_pre = t
        crit_grid = set_up_grid(N, d)  # Grid that shows all points for which z or |z| > critical value during one avalanche

        if simulation_parameter['save file for power spectrum calculation']:
            dissipation_tau = []
        if simulation_parameter['save mean value of grid']:
            means['mean'].append(np.mean(grid))
            means['time'].append(t)


        while np.any(z):
            crit_grid_tau = set_up_grid(N, d)
            crit_grid_tau[z] += 1 # Track current dissipation rate
            
            crit_grid += crit_grid_tau # Track points of grid that are above the critical value
            
            grid = relax(grid, N, crit_val, boundary_condition = boundary_condition, use_abs_val=use_abs_value, d=d)  ## Do one step of relaxation
            
            if use_abs_value:
                z = abs(grid) > crit_val
            else:
                z = grid > crit_val
                
            if simulation_parameter['save file for power spectrum calculation']:
                dissipation_tau.append(np.sum(crit_grid_tau)) # Store dissipation rate at time t
            
            t +=1 # update time
            pbar2.update(1)
            
            if simulation_parameter['save mean value of grid']:
                means['mean'].append(np.mean(grid))
                means['time'].append(t)

            # Only to create animations #####
            if safe_data_for_animations:
                if simulation_parameter['save animation of avalanches']:
                    ava_tracker.append(crit_grid.copy())
            ##################################

        # Only to create animations #####
        if safe_data_for_animations: 
            if simulation_parameter['save animation of avalanches']:
                if simulation_parameter['save multiple avalanches']:
                    if t > simulation_parameter['minimum time of avalanche']:
                        ava_tracker = np.array(ava_tracker)
                        np.save(f'{filepath_datastorage}/{simulation_name}/ava_tracker.npy', ava_tracker)
                        break
                else:
                    if (t - t_pre) > simulation_parameter['minimum time of avalanche']:
                        ava_tracker = np.array(ava_tracker)
                        np.save(f'{filepath_datastorage}/{simulation_name}/ava_tracker.npy', ava_tracker)
                        break
                    else:
                        ava_tracker = []
        ###################################
                               
        t_post = t 

        grid_dissipation_tracker += crit_grid
        
        if t_post - t_pre != 0: # if avalanche occurs, then this is valid

            
            if track_after_steady_state:
                if reached_steady_state:
                    pbar1.update(1)
                    if simulation_parameter['save file for exponent calculation']:
                        exp_data['timestep'].append(t_pre)
                        exp_data['number'].append(count_avalanches)
                        count_avalanches +=1 # update number of avalanche

                        exp_data['lifetime'].append(t_post - t_pre) # lifetime of the avalanche
                        exp_data['total dissipation'].append(np.sum(crit_grid)) # Calculate total dissipation of avalanche
                        exp_data['spatial linear size'].append(spatial_linear_distance(crit_grid, p=random_point, d=d)) # Calculate maximum spatial linear distance of avalanche

                    if simulation_parameter['save file for power spectrum calculation']:
                        list_avalanches.append(dissipation_tau) # Save dissipation rate of avalanche
            else:
                pbar1.update(1)

                if simulation_parameter['save file for exponent calculation']:
                    exp_data['timestep'].append(t_pre)
                    exp_data['number'].append(count_avalanches)
                    count_avalanches +=1 # update number of avalanche

                    exp_data['lifetime'].append(t_post - t_pre) # lifetime of the avalanche
                    exp_data['total dissipation'].append(np.sum(crit_grid)) # Calculate total dissipation of avalanche
                    exp_data['spatial linear size'].append(spatial_linear_distance(crit_grid, p=random_point, d=d)) # Calculate maximum spatial linear distance of avalanche

                if simulation_parameter['save file for power spectrum calculation']:
                    list_avalanches.append(dissipation_tau) # Save dissipation rate of avalanche


        t += 1 # update time
        pbar2.update(1)
    pbar2.close()
    pbar1.close()

    #### Save Data ####
    if not safe_data_for_animations:
        if simulation_parameter['save file for exponent calculation']:
            write_data_for_exponent_calculation_to_file(f'{filepath_datastorage}/{simulation_name}/simulation_data', 'data_for_exponent_calculation', exp_data)

        if simulation_parameter['save file for power spectrum calculation']:
            write_data_for_power_spectrum_to_file(f'{filepath_datastorage}/{simulation_name}/simulation_data', 'data_for_power_spectrum_calculation', list_avalanches)

        if simulation_parameter['save mean value of grid']:
            write_data_for_exponent_calculation_to_file(f'{filepath_datastorage}/{simulation_name}/simulation_data', 'data_mean', means)
            
    if not safe_data_for_animations:
        if d == 2:
            ## plot perturbations in the grid
            plot_heatmap(grid_dissipation_tracker, pertubation_mechanism, boundary_condition)
            plt.savefig(f'{filepath_datastorage}/plots/{simulation_name}/heatmap_dissipation_{simulation_name}.jpg')