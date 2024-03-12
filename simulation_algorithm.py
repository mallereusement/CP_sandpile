import numpy as np
import pandas as pd
from tqdm import tqdm
import copy

def set_bound_0(grid: np.ndarray, N: int, boundary: str='open', d: int=2) -> np.ndarray:
    """set all borders to 0

    Args:
        grid (np.ndarray): current grid
        N (int): number of rows/columns of the grid
        boundary (str, optional): type of boundary condition used, takes 'open' and 'closed'. Defaults to 'open'.
        d (int, optional): Dimension of Grid. Defaults to 2.

    Returns:
        np.ndarray: grid with the boundary condition applied (with borders set to 0)
    """
    for axis in range(d):
        slices = [slice(None)] * d
        slices[axis] = 0
        grid[tuple(slices)] = 0

    if boundary == 'closed':
        for axis in range(d):
            slices = [slice(None)] * d
            slices[axis] = N-1
            grid[tuple(slices)] = 0
    return grid

def generate_unit_vecs(d:int) -> list:
    empty = np.zeros(d, dtype=int)

    vectors = []
    for i in range(d):
        vec = copy.copy(empty)
        vec[i] = 1
        vectors.append(vec)
    return vectors
           
def set_up_grid(N: int, d: int=2) -> np.ndarray:
    """Generates flat grid 

    Args:
        N (int): Size of grid in one axis
        d (int, optional): Dimension of Grid. Defaults to 2.

    Returns:
        np.ndarray: grid
    """
    shape = tuple([N] * d)
    return np.zeros(shape=shape)

def pertubation_mech(grid: np.ndarray, N: int, p: np.ndarray, d:int=2, type: str = 'conservative', boundary_condition: str = 'open') -> np.ndarray:  
    """applies a pertubation to a specific point in the grid

    Args:
        grid (np.ndarray): current grid
        N (int): number of rows/columns of the grid
        type (str, optional): type of perturbation, takes 'conservative' or 'non_conservative'. Defaults to 'conservative'.
        boundary_condition (str, optional): type of boundary condition used, takes 'open' and 'closed'. Defaults to 'open'.
        px: x-position of the perturbation
        py: y-position of the perturbation

    Returns:
        np.ndarray: grid with the perturbation applied
    """
    
    p[p==False] = np.random.randint(1,N)

    if type == 'conservative':
        grid[tuple(p)] += d 
        vecs = generate_unit_vecs(d)
        for i in range(d):
            grid[tuple(p-vecs[i])] -= 1
              
    if type == 'non_conservative':
        grid[tuple(p)] += 1    
    
    grid = set_bound_0(grid, N, boundary_condition, d=d)
    return grid

def relax(grid, N, crit_val, boundary_condition='open', use_abs_val=False, d:int=2) -> np.ndarray:   
    """relaxation applied to the grid (only one time)

    Args:
        grid (np.ndarray): current grid
        N (int): number of rows/columns of the grid
        crit_val (int): value of the critical slope
        boundary_condition (str, optional): type of boundary condition used, takes 'open' and 'closed'. Defaults to 'open'.
        use_abs_val (bool, optional): Determines wether the condition |z| > z_crit or z > z_crit is used. Defaults to False. (for the latter case)

    Returns:
        np.ndarray: relaxed grid
    """
    grid = set_bound_0(grid, N, boundary_condition, d)
    
    if use_abs_val:
        mask = np.where(abs(grid) > crit_val)
    else:
        mask = np.where(grid > crit_val)
    
    for i in range(len(mask[0])):   
        vecs = generate_unit_vecs(d)
        p = np.array(mask)[:, i]

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

        else:
            if grid[tuple(p)] > 0:
                grid[tuple(p)] += -2*d + sum([1 if mask[j][i] == N-1 else 0 for j in range(0,d)])
                
                for j in range(d):
                    if p[j] < N-1:
                        grid[tuple(p + vecs[j])] += 1
                    grid[tuple(p - vecs[j])] += 1         
    
    grid = set_bound_0(grid, N, boundary_condition, d)
    return grid

def spatial_linear_distance(crit_grid, p:tuple, d:int=2) -> float:
    """_summary_

    Args:
        crit_grid (_type_): _description_
        px (_type_): _description_
        py (_type_): _description_

    Returns:
        float: _description_
    """
    truth = np.where(crit_grid >= 1)
    
    dist = []
    for i in range(len(truth[0])):
        pos = np.array(truth)[:, i]
        dist.append(np.sqrt(np.dot(p-pos, p-pos)))
    return np.max(dist)


def write_data_for_power_spectrum_to_file(filepath: str, filename: str, list_of_avalanches :list) -> None:
    with open(f'{filepath}/{filename}.txt', 'w') as f:
        for line in list_of_avalanches:
            f.write(f'{line} \n')

def write_data_for_exponent_calculation_to_file(filepath: str, filename: str, data: dict) -> None:
    df = pd.DataFrame(data)
    df.to_csv(f'{filepath}/{filename}.csv', sep=';', encoding='utf8')


def run_simulation(simulation_parameter: dict, filepath_datastorage: str, simulation_name: str):

    ### Set simulation parameter ######################################################
    N = simulation_parameter['size of grid']
    d = simulation_parameter['dimension']
    use_abs_value = simulation_parameter['use absolute value']
    pertubation_mechanism = simulation_parameter['pertubation mechanism']
    boundary_condition = simulation_parameter['boundary condition']
    crit_val = simulation_parameter['crititcal value of z']
    maximum_avalanches = simulation_parameter['number of activated avalanches']
    max_t = simulation_parameter['maximum time steps']
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
    
    pbar = tqdm(total = max_t, desc =f"Running Simulation {simulation_name}")

    ### Simulation algorithm ###########################################################

    grid = set_up_grid(N=N, d=d) # Initialize grid with z = 0
    grid_pertubation_tracker = set_up_grid(N, d) # Initialize grid that tracks pertubations
    t = 0 # Initialize time tracker
    count_avalanches = 0 # Track number of triggered avalanches

    while count_avalanches < maximum_avalanches and t < max_t: # Run simulation until maximum time steps or maximum avalanches are reached

        random_point = np.random.randint(1, N, d) # Random point where pertubation is applied
        
        grid_pertubation_tracker[tuple(random_point)] +=1 # Track point of pertubation
        
        grid = pertubation_mech(grid, N, type=pertubation_mechanism, p=random_point, d=d, boundary_condition = boundary_condition) # Apply pertubation
        
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
            pbar.update(1)
            
            if simulation_parameter['save mean value of grid']:
                means['mean'].append(np.mean(grid))
                means['time'].append(t)
            
        t_post = t 

        
        if t_post - t_pre != 0: # if avalanche occurs, then this is valid
            

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
        pbar.update(1)
    pbar.close()

    #### Save Data ####
    if simulation_parameter['save file for exponent calculation']:
        write_data_for_exponent_calculation_to_file(f'{filepath_datastorage}/simulation_data', 'data_for_exponent_calculation', exp_data)

    if simulation_parameter['save file for power spectrum calculation']:
        write_data_for_power_spectrum_to_file(f'{filepath_datastorage}/simulation_data', 'data_for_power_spectrum_calculation', list_avalanches)
    if simulation_parameter['save mean value of grid']:
        write_data_for_exponent_calculation_to_file(f'{filepath_datastorage}/simulation_data', 'data_mean', means)