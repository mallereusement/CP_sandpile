import numpy as np
import pandas as pd
import static_definitions
from iminuit.cost import LeastSquares
from iminuit import Minuit
import uncertainties as unc
import matplotlib.pyplot as plt
from uncertainties import unumpy as unp


keys_of_fit_functions = ['P_of_S', 'P_of_T', 'P_of_L', 'E_of_S_T', 'E_of_T_S', 'E_of_S_L', 'E_of_L_S', 'E_of_T_L', 'E_of_L_T']

fit_functions = static_definitions.exponent_functions()

def conditional_expectation_value(variable: str, condition: str, bins: np.ndarray, df: pd.DataFrame, x_limit: list=[]):
    """calculates the conditional expectation value E(variable|condition)

    Args:
        variable (str): variable
        condition (str): condition
        bins (np.ndarray): bins to use
        df (pd.DataFrame): simulation data
        x_limit (list): Default to  []. Set interval to use

    Returns:
        np.ndarray, uarray: array with x values and the conditional expectation values
    """
    if x_limit:
        bins = exclude_outside_interval(bins, False, *x_limit)

    bin_centers = (bins[:-1] + bins[1:]) / 2

    expectation_list = []
    expectation_list_err = []
    
    for left_edge, right_edge in zip(bins[:-1], bins[1:]):
        list = df[(df[condition] >= left_edge) & (df[condition] < right_edge)][variable].to_numpy()
        err = np.sqrt(list)
        list = unp.uarray(list, err)
        
        if np.any(list):
            
            expectation = np.sum(list) / len(list)
            
            expectation_list.append(expectation.n)
            expectation_list_err.append(expectation.s)
        else:
            expectation_list.append(0)   
            expectation_list_err.append(1)     
    expectation_list = unp.uarray(expectation_list, expectation_list_err)
    return bin_centers, expectation_list


def exclude_outside_interval(bin_edges: np.ndarray, histogram_data: np.ndarray, x_min: int, x_max: int) -> list[np.ndarray, np.ndarray]:
    """ Return histogram data from interval 

    Args:
        bin_edges (np.ndarray): bins
        histogram_data (np.ndarray): simulation data
        x_min (int): left bound
        x_max (int): right bound

    Returns:
        _type_: histogram data inside interval
    """
    min_index = np.searchsorted(bin_edges, x_min, side='left')
    max_index = np.searchsorted(bin_edges, x_max, side='right')

    truncated_bin_edges = bin_edges[min_index:max_index+1]
    if np.any(histogram_data):
        truncated_histogram_data = histogram_data[min_index:max_index]
        return truncated_bin_edges, truncated_histogram_data
    else:
        return truncated_bin_edges


def get_exponent_from_simulation_data_power_spectrum() -> dict: # ToDo
    pass


def get_exponent_from_simulation_data_conditional_exp_value(fit_function: str, bins: np.ndarray, df: pd.DataFrame, variable: str, condition: str, bootstrap_size: int, x_limit: list=[], starting_values: list = [1, 1]) -> dict: # ToDo
    """Get exponent of conditional expectation value with uncertainty for specified distribution and covariance matrix

    Args:
        fit_function (str): specified distribution
        bins (np.ndarray): bins
        df (pd.DataFrame): simulation data
        variable (str): set variable you want to look at
        condition (str): set condition you want to look at
        bootstrap_size (int): number of bootstrap samples for error estimation
        x_limit (list): define interval for fit. Default interval is provided by bins

    Returns:
        dict: ["parameters"] Fit parameters with errors, ["covariance_matrix"] covariance matrix
    """
    samples = generate_bootstrap_samples(df, bootstrap_size)


    parameters_amp = []
    parameters_exp = []
    x_org, data_org = conditional_expectation_value(variable, condition, bins, df, x_limit)

    for sample in samples:

        x, data = conditional_expectation_value(variable, condition, bins, sample, x_limit)
        m = fit_data(fit_function, x, unp.nominal_values(data), unp.std_devs(data), starting_values)
        parameter_amp, parameter_exp = m.values['amp'], m.values['exponent']
        parameters_amp.append(parameter_amp)
        parameters_exp.append(parameter_exp)

    parameters_amp = np.array(parameters_amp)
    parameters_exp = np.array(parameters_exp)

    cov_mat = np.cov(np.stack((parameters_amp, parameters_exp), axis = 0))
    value_amp = np.mean(parameters_amp)
    std_amp = np.std(parameters_amp)
    value_exp = np.mean(parameters_exp)
    std_exp = np.std(parameters_exp)

    return {"parameters": [unc.ufloat(value_amp, std_amp), unc.ufloat(value_exp, std_exp)], "covariance_matrix": cov_mat, "x": x, "data": unp.nominal_values(data_org), "errors": unp.std_devs(data_org), "model": fit_functions[fit_function](x, value_amp, value_exp), "samples": samples}


def generate_bootstrap_samples(data, bootstrap_size: int):
    """Generate bootstrap sample of size bootstrap_size for given simulation data

    Args:
        data (pd.DataFrame): simulation data
        bootstrap_size (int): number of bootstrap samples

    Returns:
        pd.DataFrame: bootstrap_size bootstrap samples
    """ 
    return [data.sample(data.shape[0], replace=True) for _ in range(bootstrap_size)]

def get_exponent_from_simulation_data(fit_function: str, bins: np.ndarray, df: pd.DataFrame, variable: str, bootstrap_size: int, x_limit: list=[], starting_values = [1,1]) -> dict:
    """Get exponent with uncertainty for specified distribution and covariance matrix

    Args:
        fit_function (str): specified distribution
        bins (np.ndarray): bins
        df (pd.DataFrame): simulation data
        variable (str): set variable you want to look at
        bootstrap_size (int): number of bootstrap samples for error estimation
        x_limit (list): define interval for fit. Default interval is provided by bins
        starting_values (list): Starting values for fit. Default to [1,1]

    Returns:
        dict: ["parameters"] Fit parameters with errors, ["covariance_matrix"] covariance matrix
    """

    data, bins = np.histogram(df[variable], bins=bins)

    if x_limit:
        bins, data = exclude_outside_interval(bins, data, *x_limit)
        df = df[(df[variable] >= x_limit[0]) & (df[variable] <= x_limit[1])]

    bin_centers = (bins[:-1] + bins[1:]) / 2
    samples = generate_bootstrap_samples(df, bootstrap_size)
    hist_data = [np.histogram(i[variable], bins=bins)[0] for i in samples]
    samples = np.array([i for i in hist_data])
    errors = np.sqrt(samples)
    parameters_amp = []
    parameters_exp = []

    for sample, error in zip(samples, errors):
        m = fit_data(fit_function, bin_centers, sample, error, starting_values)
        parameter_amp, parameter_exp = m.values['amp'], m.values['exponent']
        parameters_amp.append(parameter_amp)
        parameters_exp.append(parameter_exp)

    parameters_amp = np.array(parameters_amp)
    parameters_exp = np.array(parameters_exp)

    cov_mat = np.cov(np.stack((parameters_amp, parameters_exp), axis = 0))
    value_amp = np.mean(parameters_amp)
    std_amp = np.std(parameters_amp)
    value_exp = np.mean(parameters_exp)
    std_exp = np.std(parameters_exp)

    return {"parameters": [unc.ufloat(value_amp, std_amp), unc.ufloat(value_exp, std_exp)], "covariance_matrix": cov_mat, "x": bin_centers, "data": data, "errors": errors, "model": fit_functions[fit_function](bin_centers, value_amp, value_exp), "samples": samples}


def fit_data(fit_function: str, x:np.ndarray, data: np.ndarray, errors: np.ndarray, starting_values: list) -> float:
    """Fits fit_function to binned data and returns exponent

    Args:
        fit_function (str): fit function
        bins (np.ndarray): bin edges
        data (np.ndarray): simulation data
        starting_values (list): Starting values for fit

    Returns:
        float: returns exponent of fit function
    """
    least_squares = LeastSquares(x, data, errors, fit_functions[fit_function])
    m = Minuit(least_squares, *starting_values)
    m.migrad()
    return m




if __name__ == '__main__':


    #### To check #####
    ## look if std from numpy really creates the 68% confidence interval in this case
    ## look if we can you least square fitting for this case
    ## add error calculation for conditional expectation values, this is not implemented at the moment
    
    #f_function = 'P_of_T'
    #variable = 'lifetime'
    #x_limit = [0, 100]

    #f_function = 'P_of_S'
    #variable = 'total dissipation'
    #x_limit = [0, 200]

    #f_function = 'P_of_L'
    #variable = 'spatial linear size'
    #x_limit = [0, 60]

    f_function = 'E_of_T_S'
    variable = 'lifetime'
    condition = 'total dissipation'
    x_limit = [10, 100]

    type = 'non_conservative'
    boundary_condition = 'closed'
    df = pd.read_csv(f'results_{type}_{boundary_condition}.csv', sep=';')
    bins = np.arange(1,1000)
    bootstrap_size = 100


    result = get_exponent_from_simulation_data(f_function, bins, df, variable, bootstrap_size, x_limit = x_limit)
    result = get_exponent_from_simulation_data_conditional_exp_value(f_function, bins, df, variable, condition, bootstrap_size, x_limit = x_limit)


    print(result["parameters"])
    print(result["covariance_matrix"])

    fig, ax = plt.subplots()
    ax.scatter(result["x"], result["data"], color="blue")
    ax.plot(result["x"], result["model"], color="orange")
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.show()


