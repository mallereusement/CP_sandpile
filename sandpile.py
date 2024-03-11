import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns
import copy
from tqdm import tqdm
from matplotlib.colors import LogNorm

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

def run_simulation():
    pass



## data frame to store results in
df_results = pd.DataFrame(columns = ["number", "lifetime", "total dissipation", "spatial linear size"])

## input values
N = 10
crit_val = 3
t_max = 2e6
d=3

## settings for the algorithm
use_abs_value = True
type = 'non_conservative'
boundary_condition = 'closed'

grid = set_up_grid(N=N, d=d)         ## set up grid
pert_grid = set_up_grid(N, d)      ## array that shows where the grid was perturbed during the entire run

means = []              ## list where the mean z value of the grid at each point is stored
t_avalanche = []        ## list where the lifetimes of the avalanches are stored
n_avalanche = []        ## list where the number of avalanche is stored
total_dissipation = []  ## list where the total dissipation of each avalanche is stored
distance = []           ## list where the spatial linear distance of each avalanche is stored
t = 0                   ## time steps
n = 0                   ## current number of avalanches
file_for_simulation_data = 'simulation_data'

pbar = tqdm(total = t_max, desc ="Running Simulation")

crit_grid_sum = set_up_grid(N, d)

while t < t_max:
    ## choose point where the perturbation occurs    
    #px = np.random.randint(1, N)
    #py = np.random.randint(1, N)
    p = np.random.randint(1,N, d)
    
    pert_grid[tuple(p)] +=1
    
    grid = pertubation_mech(grid, N, type=type, p=p, d=d, boundary_condition = boundary_condition)
    
    if use_abs_value:
        z = abs(grid) > crit_val
    else:
        z = grid > crit_val
    
    t_pre = t
    crit_grid = set_up_grid(N, d)  ## grid that shows all points for which z > z_crit during one avalanche
    while np.any(z):
        
        crit_grid[z] += 1   ## increase all points where z > z_crit in an empty grid / the previous grid
        
        
        grid = relax(grid, N, crit_val, boundary_condition = boundary_condition, use_abs_val=use_abs_value, d=d)  ## relax the grid
        
        if use_abs_value:
            z = abs(grid) > crit_val
        else:
            z = grid > crit_val
            
        t +=1
        pbar.update(1)
        means.append(np.mean(grid))
        
        crit_grid_sum += crit_grid
        
        
    t_post = t 
    

    
    if t_post - t_pre != 0:     ## if avalanche occurs, then this is valid
        t_avalanche.append(t_post - t_pre)          ## lifetime of the avalanche
        n +=1                                       ## update number of avalanche
        n_avalanche.append(n)
        total_dissipation.append(np.sum(crit_grid))
        distance.append(spatial_linear_distance(crit_grid, p=p, d=d))
  
    ## plot heatmaps of the crit_grid of avalanches with lifetimes larger than some value  
    #if t_post - t_pre > 40:  
    #    sns.heatmap(crit_grid)
    #    plt.text(95, 5, f'$\\tau$ = {t_post - t_pre} \n $f_{{tot}}$ = {np.sum(crit_grid)} \n $s$ = {spatial_linear_distance(crit_grid, px, py):.2f}', ha='right', va='top', color='white')
    #    plt.show()
    
    
    
    means.append(np.mean(grid))
    t += 1
    pbar.update(1)

pbar.close()

#sns.heatmap(crit_grid_sum, norm=LogNorm())
#plt.text(95, 5, f'perturbation = {type} \n boundary = {boundary_condition} ', ha='right', va='top', color='white')
#plt.show()

## write results into data frame
df_results["lifetime"] = t_avalanche
df_results["number"] = n_avalanche
df_results['total dissipation'] = total_dissipation
df_results["spatial linear size"] = distance

df_results.to_csv(f'{file_for_simulation_data}/results_{type}_{boundary_condition}_{d}.csv', index=False, sep=';')

print(df_results)

## plot average z value of grid versus time
plt.plot(np.arange(t)[::1000], means[::1000])
plt.show()

## plot lifetime of avalanches 
plt.hist(t_avalanche, bins=np.arange(1000))
plt.yscale('log')
plt.xscale('log')
plt.show()

