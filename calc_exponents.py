import numpy as np
import pandas as pd
import static_definitions
from iminuit.cost import LeastSquares
from iminuit import Minuit
import uncertainties as unc


keys_of_fit_functions = ['P_of_S', 'P_of_T', 'P_of_L', 'E_of_S_T', 'E_of_T_S', 'E_of_S_L', 'E_of_L_S', 'E_of_T_L', 'E_of_L_T']

fit_functions = static_definitions.exponent_functions()


def exclude_outside_interval(bin_edges: np.ndarray, histogram_data: np.ndarray, x_min: int, x_max: int) -> list[np.ndarray, np.ndarray]:
    """ Return histogram data from interval 

    Args:
        bin_edges (np.ndarray): bins
        histogram_data (np.ndarray): simulation data
        x_min (int): left bound
        x_max (int): right bound

    Returns:
        _type_: _description_
    """
    min_index = np.searchsorted(bin_edges, x_min, side='left')
    max_index = np.searchsorted(bin_edges, x_max, side='right')

    truncated_bin_edges = bin_edges[min_index:max_index+1]
    truncated_histogram_data = histogram_data[min_index:max_index]

    return truncated_bin_edges, truncated_histogram_data


def get_exponent_from_simulation_data(fit_function: str, bins: np.ndarray, data: np.ndarray, bootstrap_size: int, x_limit: list=[]) -> unc.ufloat:
    """Get exponent with uncertainty for specified distribution

    Args:
        fit_function (str): specified distribution
        bins (np.ndarray): bins
        data (np.ndarray): simulation data
        bootstrap_size (int): number of bootstrap samples for error estimation
        x_limit (list): define interval for fit. Default interval is provided by bins

    Returns:
        unc.ufloat: exponent with uncertainty
    """
    if x_limit:
        bins, data = exclude_outside_interval(bins, data, *x_limit)

    samples = generate_bootstrap_samples(data, bootstrap_size)
    parameters = []

    for sample in samples:
        parameter = fit_data(fit_function, bins, sample)
        parameters.append(parameter)
    parameters = np.array(parameters)
    value = np.mean(parameters)
    std = np.std(parameters)
    return unc.ufloat(value, std)

def generate_bootstrap_samples(data: np.ndarray, bootstrap_size: int) -> np.ndarray:
    """Generate bootstrap sample of size bootstrap_size for given simulation data

    Args:
        data (np.ndarray): simulation data
        bootstrap_size (int): number of bootstrap samples

    Returns:
        np.ndarray: bootstrap_size bootstrap samples
    """
    return np.array([np.random.choice(data, size=data.size, replace=True) for _ in range(R)])

def fit_data(fit_function: str, bins: np.ndarray, data: np.ndarray) -> float:
    """Fits fit_function to binned data and returns exponent

    Args:
        fit_function (str): fit function
        bins (np.ndarray): bin edges
        data (np.ndarray): simulation data

    Returns:
        float: returns exponent of fit function
    """
    bin_centers = (bins[:-1] + bins[1:]) / 2
    errors = np.sqrt(data)
    least_squares = LeastSquares(bin_centers, data, errors, fit_functions[fit_function])
    m = Minuit(least_squares, amp=data[0], exponent=1)
    m.migrad()
    return m.values['exponent']




if __name__ == '__main__':
    
    f_function = 'P_of_T'
    data_name = 'lifetime'
    type = 'non_conservative'
    boundary_condition = 'closed'
    df = pd.read_csv(f'results_{type}_{boundary_condition}.csv', sep=';')
    bins = np.arange(1,1000)
    data, bins = np.histogram(df[data_name], bins=bins)
    bootstrap_size = 1000
    x_limit = [0, 1000]


    exponent = get_exponent_from_simulation_data(f, bins, data, bootstrap_size, x_limit = x_limit)

    print(exponent)

