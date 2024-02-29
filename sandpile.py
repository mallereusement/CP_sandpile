import numpy as np
import matplotlib.pyplot as plt 
import copy

def set_bound_0(grid, N, boundary='open'):
    grid[0,:] = 0
    grid[:,0] = 0
    if boundary == 'closed':
        grid[N-1,:] = 0
        grid[:,N-1] = 0
    return grid
        
    
def set_up_grid(N):
    """Sets up the grid

    Args:
        N (_type_): number of rows/colums
    """
    
    return np.zeros(shape=(N,N))

def pertubation_mech(grid, N, point = 'random', type = 'conservative', boundary_condition = 'open'):
   
    if point == 'random':
        px = np.random.randint(1, N)
        py = np.random.randint(1, N)
    
    if type == 'conservative':
        grid[px,py] += 2 
        grid[px - 1, py] -= 1
        grid[px, py - 1] -= 1
        
    if type == 'non_conservative':
        grid[px,py] += 1    
    
    grid = set_bound_0(grid, N, boundary_condition)
    return grid

def relax(grid, N, crit_val, boundary_condition='open', use_abs_val=False):   
    
    grid = set_bound_0(grid, N, boundary_condition)
    
    if use_abs_val:
        mask = np.where(abs(grid) > crit_val)
    else:
        mask = np.where(grid > crit_val)
        
    #if boundary_condition == 'open':
   
    for i in range(len(mask[0])):
        if use_abs_val:
            if grid[mask[0][i], mask[1][i]] > 0:
                grid[mask[0][i], mask[1][i]] += -2*2 + sum([1 if mask[j][i] == N-1 else 0 for j in range(0,2)])
                if mask[0][i] < N-1:
                    grid[mask[0][i]+1, mask[1][i]] += 1
                if mask[1][i] < N-1:
                    grid[mask[0][i], mask[1][i]+1] += 1
                grid[mask[0][i]-1, mask[1][i]] += 1
                grid[mask[0][i], mask[1][i]-1] += 1
                
                    
            elif grid[mask[0][i], mask[1][i]] < 0:
                grid[mask[0][i], mask[1][i]] += 2*2 - sum([1 if mask[j][i] == N-1 else 0 for j in range(0,2)])
                if mask[0][i] < N-1:
                    grid[mask[0][i]+1, mask[1][i]] -= 1
                if mask[1][i] < N-1:
                    grid[mask[0][i], mask[1][i]+1] -= 1
                grid[mask[0][i]-1, mask[1][i]] -= 1
                grid[mask[0][i], mask[1][i]-1] -= 1
        else:
            grid[mask[0][i], mask[1][i]] += -2*2 + sum([1 if mask[j][i] == N-1 else 0 for j in range(0,2)])
            if mask[0][i] < N-1:
                grid[mask[0][i]+1, mask[1][i]] += 1
            if mask[1][i] < N-1:
                grid[mask[0][i], mask[1][i]+1] += 1
            grid[mask[0][i]-1, mask[1][i]] += 1
            grid[mask[0][i], mask[1][i]-1] += 1           
            
        
    #if boundary_condition == 'closed':
    #    pass
    

    
    grid = set_bound_0(grid, N, boundary_condition)
    return grid

N = 40
crit_val = 3
grid = set_up_grid(N=N)
t = 0
use_abs_value = False
means = []

while t < 2e5:
#for i in range(int(5e5)):
    if use_abs_value:
        z = abs(grid) > crit_val
    else:
        z = grid > crit_val
    
    t_pre = t
    while np.any(z):
        
        grid = relax(grid, N, crit_val)
        
        if use_abs_value:
            z = abs(grid) > crit_val
        else:
            z = grid > crit_val
        t +=1
        means.append(np.mean(grid))
        
    t_post = t 
    
    t_avalanche = t_post - t_pre
    
    grid = pertubation_mech(grid, N, type='non_conservative')
    means.append(np.mean(grid))
    t += 1

plt.plot(np.arange(t)[::1000], means[::1000])
plt.show()