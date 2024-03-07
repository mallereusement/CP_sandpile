def pertubation_mech(grid: np.ndarray, N: int, point: np.ndarray, d: int=2, type: str = 'conservative', boundary_condition: str = 'open') -> np.ndarray:  

    point[point == False] = np.random.randint(1, N)
    
    if type == 'conservative':
        grid[point] += d 
        for i in range(d):
            grid[point - ] -= 1
        grid[px, py - 1] -= 1
        
    if type == 'non_conservative':
        grid[px,py] += 1    
    
    grid = set_bound_0(grid, N, boundary_condition)
    return grid