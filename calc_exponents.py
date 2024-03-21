import numpy as np
import pandas as pd
import static_definitions
from iminuit.cost import LeastSquares
from iminuit import Minuit
import uncertainties as unc
import matplotlib.pyplot as plt
from uncertainties import unumpy as unp
from tqdm import tqdm
import math


fit_functions = static_definitions.exponent_functions()

keys_of_fit_functions = ['P_of_S', 'P_of_T', 'P_of_L', 'E_of_S_T', 'E_of_T_S', 'E_of_S_L', 'E_of_L_S', 'E_of_T_L', 'E_of_L_T', 'gamma1_gamma3_1', 'gamma1_gamma3_2']

def conditional_expectation_value(variable: str, condition: str, bins: np.ndarray, df: pd.DataFrame, x_limit: list=[], get_error_with_bootstrapping: bool=False, bootstrap_size=200):
    """Calculates the conditional expectation value E(variable|condition).

    Args:
        variable (str): Name of the variable.
        condition (str): Name of the condition.
        bins (np.ndarray): Array of bin edges.
        df (pd.DataFrame): DataFrame containing simulation data.
        x_limit (list, optional): Interval to use. Defaults to [].
        get_error_with_bootstrapping (bool, optional): Whether to estimate error with bootstrapping. Defaults to False.
        bootstrap_size (int, optional): Number of bootstrap samples. Defaults to 200.

    Returns:
        tuple: Tuple containing:
            np.ndarray: Array with x values (bin centers).
            unp.uarray: Array containing conditional expectation values with uncertainties.
            np.ndarray: Array containing statistics (number of data points) for each bin.
    """
    # Apply x limit if specified
    if x_limit:
        bins = exclude_outside_interval(bins, False, *x_limit)
    # Compute bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2
    # Initialize lists for expectation values and statistics
    expectation_list = []
    expectation_list_err = []
    statistics = []
    # Iterate over bins
    for left_edge, right_edge in zip(bins[:-1], bins[1:]):
        # Filter data within current bin
        list = df[(df[condition] > left_edge) & (df[condition] <= right_edge)][variable].to_numpy()
        # Check if there is any data within the bin
        if np.any(list):

            list = unp.uarray(list, np.sqrt(list))
            # Compute conditional expectation value
            expectation = np.sum(list) / len(list)
            expectation_list.append(unp.nominal_values(expectation))
            expectation_list_err.append(unp.std_devs(expectation))
            statistics.append(len(list))
        else:
            # Append zero values if no data within the bin
            expectation_list.append(0)   
            expectation_list_err.append(1)
            statistics.append(0)   
    expectation_list = unp.uarray(expectation_list, expectation_list_err)
    statistics = np.array(statistics)

    # Estimate error with bootstrapping if specified
    if get_error_with_bootstrapping:
        pbar = tqdm(total = bootstrap_size, desc ="Running Bootstrap for Error Estimation of conditional expectation value")
        samples = generate_bootstrap_samples(df, bootstrap_size)
        expectation_list_bootstrap_global = []
        # Iterate over bootstrap samples
        for sample in samples:
            expectation_list_bootstrap = []
            # Iterate over bins
            for left_edge, right_edge in zip(bins[:-1], bins[1:]):
                list = sample[(sample[condition] >= left_edge) & (sample[condition] < right_edge)][variable].to_numpy()
                # Check if there is any data within the bin
                if np.any(list):
                    # Compute conditional expectation value
                    expectation = np.sum(list) / len(list)
                    
                    expectation_list_bootstrap.append(expectation)
                else:
                    expectation_list_bootstrap.append(0)   
            pbar.update(1)
            expectation_list_bootstrap = np.array(expectation_list_bootstrap)
            expectation_list_bootstrap_global.append(expectation_list_bootstrap)
        pbar.close()
        expectation_list_bootstrap_global = np.array(expectation_list_bootstrap_global)
        # Compute standard deviations
        stds = np.std(expectation_list_bootstrap_global, axis=0)
        # Replace small standard deviations that cause problems in the fit with 1% of nominal values
        indices = np.where(stds < 1e-3)
        one_percent_data = 0.01 * unp.nominal_values(expectation_list)
        stds[indices] = one_percent_data[indices]

        expectation_list = unp.uarray(unp.nominal_values(expectation_list), stds)
    return bin_centers, expectation_list, statistics


def exclude_outside_interval(bin_edges: np.ndarray, histogram_data: np.ndarray, x_min: int, x_max: int) -> list[np.ndarray, np.ndarray]:
    """Return histogram data within the specified interval.

    Args:
        bin_edges (np.ndarray): The bin edges.
        histogram_data (np.ndarray): The histogram data corresponding to bin edges.
        x_min (int): The left bound of the interval.
        x_max (int): The right bound of the interval.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing truncated bin edges and histogram data within the interval.
    """
    # Find indices corresponding to x_min and x_max
    min_index = np.searchsorted(bin_edges, x_min, side='left')
    max_index = np.searchsorted(bin_edges, x_max, side='right')
    # Truncate bin edges and histogram data based on the interval
    truncated_bin_edges = bin_edges[min_index:max_index]
    if np.any(histogram_data):
        truncated_histogram_data = histogram_data[min_index:max_index-1]
        return truncated_bin_edges, truncated_histogram_data
    else:
        # If histogram data is empty, return only truncated bin edges
        return truncated_bin_edges


def get_exponent_from_simulation_data_power_spectrum(fit_function: str, bins: np.ndarray, df: pd.DataFrame, bootstrap_size: int, x_limit: list=[], starting_values: list = [1, 1]) -> dict: # ToDo
    pass

def get_exponent_product_from_simulation_data_conditional_exp_value(fit_function: str, bins1: np.ndarray, bins2: np.ndarray, df: pd.DataFrame, bootstrap_size: int, x_limit1: list=[], x_limit2: list=[], starting_values: list = [1, 1], block_size: int=0) -> dict: # ToDo
    """Get product of exponents gamma1 and gamma3 with uncertainty.

    Args:
        fit_function (str): Specified fit function.
        bins1 (np.ndarray): Bins for the first variable.
        bins2 (np.ndarray): Bins for the second variable.
        df (pd.DataFrame): DataFrame containing simulation data.
        bootstrap_size (int): Number of bootstrap samples for error estimation.
        x_limit1 (list, optional): Interval for fit for the first variable. Default is [].
        x_limit2 (list, optional): Interval for fit for the second variable. Default is [].
        starting_values (list, optional): Starting values for fitting parameters. Default is [1, 1].
        block_size (int, optional): Size of blocks for block bootstrapping. Default is 0.

    Returns:
        dict: A dictionary containing fit parameters, chi-square values, degrees of freedom, exponent product with uncertainty, products from bootstrap, samples, and number of valid fits.
    """
    # Generate bootstrap samples
    samples = generate_bootstrap_samples(df, bootstrap_size, block_size=block_size)
    product = []

    if fit_function == 'gamma1_gamma3_1':
        # Calculate conditional expectation values for the first variable and fit data
        x_org1, data_org1, _ = conditional_expectation_value('total dissipation', 'lifetime', bins1, df, x_limit1, get_error_with_bootstrapping=True, bootstrap_size=bootstrap_size)
        m_org1 = fit_data(fit_funtion_mapping[fit_function][0], x_org1, unp.nominal_values(data_org1), unp.std_devs(data_org1), starting_values)
        # Calculate conditional expectation values for the second variable and fit data
        x_org2, data_org2, _ = conditional_expectation_value('lifetime', 'spatial linear size', bins2, df, x_limit2, get_error_with_bootstrapping=True, bootstrap_size=bootstrap_size)
        m_org2 = fit_data(fit_funtion_mapping[fit_function][1], x_org2, unp.nominal_values(data_org2), unp.std_devs(data_org2), starting_values)
    elif fit_function == 'gamma1_gamma3_2':
        # Calculate conditional expectation values for the first variable and fit data
        x_org1, data_org1, _ = conditional_expectation_value('lifetime', 'total dissipation', bins1, df, x_limit1, get_error_with_bootstrapping=True, bootstrap_size=bootstrap_size)
        m_org1 = fit_data(fit_funtion_mapping[fit_function][0], x_org1, unp.nominal_values(data_org1), unp.std_devs(data_org1), starting_values)
        # Calculate conditional expectation values for the second variable and fit data
        x_org2, data_org2, _ = conditional_expectation_value('spatial linear size', 'lifetime', bins2, df, x_limit2, get_error_with_bootstrapping=True, bootstrap_size=bootstrap_size)
        m_org2 = fit_data(fit_funtion_mapping[fit_function][1], x_org2, unp.nominal_values(data_org2), unp.std_devs(data_org2), starting_values)
    # Initialize counter for valid fits    
    valid_counter = 0

    pbar = tqdm(total = bootstrap_size, desc ="Running Bootstrap for Error Estimation of fit parameters")
    # Iterate over bootstrap samples
    for sample in samples:
        if fit_function == 'gamma1_gamma3_1':  
            # Calculate conditional expectation values for the first variable in the bootstrap sample    
            x1, data1, _ = conditional_expectation_value('total dissipation', 'lifetime', bins1, sample, x_limit1)
            # Fit Data
            m1 = fit_data('E_of_S_T', x1, unp.nominal_values(data1), unp.std_devs(data_org1), starting_values)
             # Calculate conditional expectation values for the second variable in the bootstrap sample  
            x2, data2, _ = conditional_expectation_value('lifetime', 'spatial linear size', bins2, sample, x_limit2)
            # Fit Data
            m2 = fit_data('E_of_T_L', x2, unp.nominal_values(data2), unp.std_devs(data_org2), starting_values)
            # Check if both fits are valid and update counter
            if (m1.valid == True) and (m2.valid == True):
                valid_counter = valid_counter + 1
                # Get parameters for the first fit
                parameter_amp1, parameter_exp1 = m1.values['amp'], m1.values['exponent']
                # Get parameters for the second fit
                parameter_amp2, parameter_exp2 = m2.values['amp'], m2.values['exponent']
                product.append(parameter_exp1*parameter_exp2)
        elif fit_function == 'gamma1_gamma3_2':
            # Calculate conditional expectation values for the first variable in the bootstrap sample
            x1, data1, _ = conditional_expectation_value('lifetime', 'total dissipation', bins1, sample, x_limit1)
            # Fit Data
            m1 = fit_data('E_of_S_T', x1, unp.nominal_values(data1), unp.std_devs(data_org1), starting_values)
            # Calculate conditional expectation values for the second variable in the bootstrap sample
            x2, data2, _ = conditional_expectation_value('spatial linear size', 'lifetime', bins2, sample, x_limit2)
            # Fit Data
            m2 = fit_data('E_of_T_L', x2, unp.nominal_values(data2), unp.std_devs(data_org2), starting_values)
            # Check if both fits are valid and update counter
            if (m1.valid == True) and (m2.valid == True):
                valid_counter = valid_counter + 1
                # Get parameters for the first fit
                parameter_amp1, parameter_exp1 = m1.values['amp'], m1.values['exponent']
                # Get parameters for the second fit
                parameter_amp2, parameter_exp2 = m2.values['amp'], m2.values['exponent']
                product.append(parameter_exp1*parameter_exp2)        
        pbar.update(1)
    pbar.close()   

    product = np.array(product)
    org_product = m_org1.values['exponent'] * m_org2.values['exponent']
    # Calculate uncertainties
    std = np.std(product)
    print(product)
    return {"chi_square1": m_org1.fval, "degrees_of_freedom1": m_org1.ndof, "chi_square2": m_org2.fval, "degrees_of_freedom2": m_org2.ndof, "product": unc.ufloat(org_product, std), "products_from_bootstrap": product, "samples": samples, "number_of_valid_fits": valid_counter}



def get_exponent_from_simulation_data_conditional_exp_value(fit_function: str, bins: np.ndarray, df: pd.DataFrame, variable: str, condition: str, bootstrap_size: int, x_limit: list=[], starting_values: list = [1, 1], block_size: int=0) -> dict: # ToDo
    """Get exponent of conditional expectation value with uncertainty.

    Args:
        fit_function (str): Specified fit function.
        bins (np.ndarray): Bins for the variable.
        df (pd.DataFrame): DataFrame containing simulation data.
        variable (str): Variable.
        condition (str): Condition.
        bootstrap_size (int): Number of bootstrap samples for error estimation.
        x_limit (list, optional): Interval for fit. Default is [].
        starting_values (list, optional): Starting values for fitting parameters. Default is [1, 1].
        block_size (int, optional): Size of blocks for block bootstrapping. Default is 0.

    Returns:
        dict: A dictionary containing fit parameters, chi-square values, degrees of freedom, exponent values from bootstrap, parameters with errors, covariance matrix, x values, data, errors, model, and samples.
    """
    # Generate bootstrap samples
    samples = generate_bootstrap_samples(df, bootstrap_size, block_size=block_size)

    # Initialize lists for storing fit parameters
    parameters_amp = []
    parameters_exp = []
    # Calculate conditional expectation values for original data and fit data
    x_org, data_org, _ = conditional_expectation_value(variable, condition, bins, df, x_limit, get_error_with_bootstrapping=True, bootstrap_size=bootstrap_size)
    m_org = fit_data(fit_function, x_org, unp.nominal_values(data_org), unp.std_devs(data_org), starting_values)
    pbar = tqdm(total = bootstrap_size, desc ="Running Bootstrap for Error Estimation of fit parameters")
    # Iterate over bootstrap samples
    for sample in samples:
        # Calculate conditional expectation values for each bootstrap sample and fit sample
        x, data, _ = conditional_expectation_value(variable, condition, bins, sample, x_limit)
        m = fit_data(fit_function, x, unp.nominal_values(data), unp.std_devs(data_org), starting_values)
        pbar.update(1)
        # Store parameters if fit is valid
        if m.valid:
            parameter_amp, parameter_exp = m.values['amp'], m.values['exponent']
            parameters_amp.append(parameter_amp)
            parameters_exp.append(parameter_exp)
    pbar.close()
    parameters_amp = np.array(parameters_amp)
    parameters_exp = np.array(parameters_exp)
    # Calculate covariance matrix
    cov_mat = np.cov(np.stack((parameters_amp, parameters_exp), axis = 0))
    print(parameters_exp)

    return {"chi_square": m_org.fval, "degrees_of_freedom": m_org.ndof, "exponent_values_from_bootrstrap": parameters_exp, "parameters": [unc.ufloat(m_org.values['amp'], cov_mat[0,0]**0.5), unc.ufloat(m_org.values['exponent'], cov_mat[1,1]**0.5)], "covariance_matrix": cov_mat, "x": x, "data": unp.nominal_values(data_org), "errors": unp.std_devs(data_org), "model": fit_functions[fit_function](x, m_org.values['amp'], m_org.values['exponent']), "samples": samples}


def generate_following_indices(indices: list, n: int) -> list:
    """Generate a list of consecutive indices following the given indices.

    Args:
        indices (list): List of starting indices.
        n (int): Number of consecutive indices to generate following each starting index.

    Returns:
        list: List containing consecutive indices following each starting index.
    """
    result = []
    for index in indices:
        result.extend(range(index, index + n ))
    return result


def generate_bootstrap_samples(data: pd.DataFrame, bootstrap_size: int, block_size: int=0):
    """Generate bootstrap samples for given simulation data. Number of samples = bootstrap_size

    Args:
        data (pd.DataFrame): Simulation data.
        bootstrap_size (int): Number of bootstrap samples.
        block_size (int): Block size for bootstrapping due to correlation of time series. Defaults to 0.

    Returns:
        List[pd.DataFrame]: list of bootstrap samples.
    """
    if block_size == 0:
        # If block size is 0, perform simple bootstrap sampling
        return [data.sample(data.shape[0], replace=True) for _ in range(bootstrap_size)]
    else:
        # Perform block bootstrap sampling
        # Calculate the number of blocks
        numb = math.floor(data.shape[0] / block_size)
        samples = []
        for x in range(bootstrap_size):
            # Generate random integers to select blocks
            sample_f_d = np.random.randint(0, numb, numb)
            # Generate indices for following blocks
            sample_f_d = generate_following_indices(sample_f_d, block_size)
            # Select data based on generated indices
            sample = df.iloc[sample_f_d]
            samples.append(sample)
        return samples

def get_exponent_from_simulation_data(fit_function: str, bins: np.ndarray, df: pd.DataFrame, variable: str, bootstrap_size: int, x_limit: list=[], starting_values = [1,1], block_size: int=0) -> dict:
    """Get exponent with uncertainty for specified fit function

    Args:
        fit_function (str): Specified fit function.
        bins (np.ndarray): Bins.
        df (pd.DataFrame): Simulation data.
        variable (str): Variable you want to look at.
        bootstrap_size (int): Number of bootstrap samples for error estimation.
        x_limit (list): Define interval for fit. Default interval is provided by bins.
        starting_values (list): Starting values for fit. Default to [1,1].
        block_size (int): Block size for bootstrapping due to correlation of time series. Defaults to 0.

    Returns:
        dict: Fit parameters with errors, covariance matrix, and other information.
    """
    # Get histogram data
    data, bins = np.histogram(df[variable], bins=bins)

    # Exclude data outside specified interval
    if x_limit:
        bins, data = exclude_outside_interval(bins, data, *x_limit)
        df = df[(df[variable] >= x_limit[0]) & (df[variable] <= x_limit[1])]

    bin_centers = (bins[:-1] + bins[1:]) / 2
    # Generate bootstrap samples
    samples = generate_bootstrap_samples(df, bootstrap_size, block_size=block_size)
    # Calculate histogram data for each bootstrap sample
    hist_data = [np.histogram(i[variable], bins=bins)[0] for i in samples]
    samples = np.array([i for i in hist_data])
    # Avoid problems with too small errors
    errors = np.std(samples, axis=0)
    indices = np.where(errors < 1e-4)
    one_percent_data = 0.01 * data
    errors[indices] = one_percent_data[indices]
    # Initialize lists to store fit parameters
    parameters_amp = []
    parameters_exp = []
    # Perform fit on original data
    m_org = fit_data(fit_function, bin_centers, data, errors, starting_values)

    # Iterate over bootstrap samples
    for sample in samples:

        sample[sample == 0] = 1e-7 # avoid division by zero
        sample = sample.astype(np.longdouble)
        # Perform fit on bootstrap sample
        m = fit_data(fit_function, bin_centers, sample, errors, starting_values)
        if m.valid:
            parameter_amp, parameter_exp = m.values['amp'], m.values['exponent']
            parameters_amp.append(parameter_amp)
            parameters_exp.append(parameter_exp)

    parameters_amp = np.array(parameters_amp)
    parameters_exp = np.array(parameters_exp)
    print(parameters_exp)
    # Calculate covariance matrix
    cov_mat = np.cov(np.stack((parameters_amp, parameters_exp), axis = 0))

    return {"chi_square": m_org.fval, "degrees_of_freedom": m_org.ndof, "exponent_values_from_bootrstrap": parameters_exp, "parameters": [unc.ufloat(m_org.values['amp'], cov_mat[0,0]**0.5), unc.ufloat(m_org.values['exponent'], cov_mat[1,1]**0.5)], "covariance_matrix": cov_mat, "x": bin_centers, "data": data, "errors": errors, "model": fit_functions[fit_function](bin_centers, m_org.values['amp'], m_org.values['exponent']), "samples": samples}


def fit_data(fit_function: str, x: np.ndarray, data: np.ndarray, errors: np.ndarray, starting_values: list) -> float:
    """Fit fit_function to binned data and return fitted parameters.

    Args:
        fit_function (str): Specified fit function.
        x (np.ndarray): Array of x values.
        data (np.ndarray): Array of data points.
        errors (np.ndarray): Array of errors corresponding to data points.
        starting_values (list): List of starting values for fit.

    Returns:
        Minuit: Fitted Minuit object containing fit parameters.
    """
    # Define least squares object
    least_squares = LeastSquares(x, data, errors, fit_functions[fit_function])
    # Create Minuit object and set initial parameter values
    m = Minuit(least_squares, *starting_values)
    # Set parameter limits to make sure fit converges
    m.limits = [(0, None), (0.2, 10)]
    # Perform fitting
    m.migrad()
    return m



def save_exponent_data(fit_function: str, bins: dict, bootstrap_size: str, fit_results: dict, file_to_save, file_to_load=False, bins2=False) -> None:
    """Save exponent data to a CSV file.

    Args:
        fit_function (str): Specified fit function.
        bins (dict): Dictionary containing bin edges.
        bootstrap_size (int): Number of bootstrap samples.
        fit_results (dict): Dictionary containing fit results.
        file_to_save (str): File path to save the data.
        file_to_load (bool, optional): Whether to load data from an existing file. Defaults to False.
        bins2 (bool, optional): Additional bins for the second variable. Defaults to False.
    """
    # Determine if second set of bins is provided, this is only the case for products of exponents
    if bins2 == False:
        bins = get_bins_from_parameter_settings(*bins)
        # Create DataFrame with fit results
        temp_df = pd.DataFrame({'chi_square': [fit_results['chi_square']], 'degrees_of_freedom': [fit_results['degrees_of_freedom']], 'fit function': [fit_function], 'variable': [fit_funtion_mapping[fit_function][0]], 'condition': [fit_funtion_mapping[fit_function][1]], 'left bin edge': [bins[0]], 'right bin edge': [bins[-1]], 'count of bins': [len(bins)-1], 'bootstrap size': [bootstrap_size], 'amplitude from fit result': [fit_results['parameters'][0]], 'exponent from fit result': [fit_results['parameters'][1]], 'covariance c_11': [fit_results['covariance_matrix'][0,0]], 'covariance c_12': [fit_results['covariance_matrix'][0,1]], 'covariance c_22': [fit_results['covariance_matrix'][1,1]]})
        
        if not file_to_load:
            temp_df.to_csv(file_to_save, sep=';', encoding='utf8', index=False)
        else:
            df = pd.read_csv(file_to_load, sep=';', encoding='utf8')
            df = pd.concat([df, temp_df], ignore_index=True)
            df.to_csv(file_to_save, sep=';', encoding='utf8', index=False)
    else:
        bins = get_bins_from_parameter_settings(*bins)
        bins2 = get_bins_from_parameter_settings(*bins2)
        # Create DataFrame with fit results
        temp_df = pd.DataFrame({'chi_square1': [fit_results['chi_square1']], 'degrees_of_freedom1': [fit_results['degrees_of_freedom1']], 'chi_square2': [fit_results['chi_square2']], 'degrees_of_freedom2': [fit_results['degrees_of_freedom2']], 'fit function': [fit_function], 'left bin edge 1': [bins[0]], 'right bin edge 1': [bins[-1]], 'count of bins 1': [len(bins)-1], 'left bin edge 2': [bins2[0]], 'right bin edge 2': [bins2[-1]], 'count of bins 2': [len(bins2)-1], 'bootstrap size': [bootstrap_size], 'product from fit result': [fit_results['product']], 'number of valid fits': [fit_results['number_of_valid_fits']]})
        
        # Check if data should be appended to an existing file or saved to a new file
        if not file_to_load:
            temp_df.to_csv(file_to_save, sep=';', encoding='utf8', index=False)
        else:
            df = pd.read_csv(file_to_load, sep=';', encoding='utf8')
            df = pd.concat([df, temp_df], ignore_index=True)
            df.to_csv(file_to_save, sep=';', encoding='utf8', index=False)


def load_simulation_data(sim_data: dict) -> pd.DataFrame:
    """_summary_

    Args:
        sim_data (dict): _description_

    Returns:
        pd.DataFrame: _description_
    """
    file = f"results_{sim_data['type']}_{sim_data['boundary_condition']}.csv"
    return pd.read_csv(file, sep=';', encoding='utf8')


def get_bins_from_parameter_settings(start_bin: int, end_bin: int, bin_width: int):
    """Generate bin edges based on provided parameters.

    This function generates bin edges for histogram bins based on the specified
    start bin, end bin, and bin width.

    Args:
        start_bin (int): The starting value of the bins.
        end_bin (int): The ending value of the bins.
        bin_width (int): The width of each bin.

    Returns:
        np.ndarray: An array containing the bin edges.
    """
    return np.linspace(start_bin, end_bin, int((end_bin-start_bin) / float(bin_width) + 1))

# Dictionary mapping fit functions to corresponding variable and condition pairs
fit_funtion_mapping = {
    'S_of_f': ['power spectrum', '-'],
    'P_of_S': ['total dissipation', '-'],
    'P_of_T': ['lifetime', '-'],
    'P_of_L': ['spatial linear size', '-'],
    'E_of_S_T': ['total dissipation', 'lifetime'],
    'E_of_T_S': ['lifetime', 'total dissipation'],
    'E_of_S_L': ['total dissipation', 'spatial linear size'],
    'E_of_L_S': ['spatial linear size', 'total dissipation'],
    'E_of_T_L': ['lifetime', 'spatial linear size'],
    'E_of_L_T': ['spatial linear size', 'lifetime'],
    'gamma1_gamma3_1': ['E_of_S_T', 'E_of_T_L'],
    'gamma1_gamma3_2': ['E_of_T_S', 'E_of_L_T']

}

def run_calculation(fit_function: str, bootrstrap_size: int, bins: list, df: pd.DataFrame, bins2=False, block_size: int = 0):
    """Run calculation to determine the exponent with uncertainty based on the fit function.

    Args:
        fit_function (str): Name of the fit function for the exponent calculation.
        bootrstrap_size (int): Number of bootstrap samples for error estimation of the fit and the data.
        bins (list): List containing [start bin, end bin, width of bin].
        df (pd.DataFrame): DataFrame containing the simulated data.
        bins2 (list, optional): Secondary bins for product of exponents. Defaults to False.
        block_size (int, optional): Block size for bootstrapping due to time series correlation. Defaults to 0.

    Returns:
        dict: Fit results containing chi-square, degrees of freedom, and exponent values with uncertainties etc.
    """
    bins = get_bins_from_parameter_settings(*bins)
    # Check if the fit function requires conditional expectation calculation
    if fit_funtion_mapping[fit_function][1] == '-':
        if fit_function == 'S_of_f':
            # If fit function is S_of_f, perform power spectrum calculation (To Do)
            result = get_exponent_from_simulation_data_power_spectrum()
        # Otherwise, perform standard conditional expectation calculation
        else:
            result = get_exponent_from_simulation_data(fit_function, bins, df, fit_funtion_mapping[fit_function][0], bootrstrap_size, block_size=block_size)

    else:
        # If product of exponents is required, calculate it
        if fit_function == 'gamma1_gamma3_1':
            bins2 = get_bins_from_parameter_settings(*bins2)
            result = get_exponent_product_from_simulation_data_conditional_exp_value('gamma1_gamma3_1', bins, bins2, df, bootrstrap_size, block_size=block_size)
        elif fit_function == 'gamma1_gamma3_2':
            bins2 = get_bins_from_parameter_settings(*bins2)
            result = get_exponent_product_from_simulation_data_conditional_exp_value('gamma1_gamma3_2', bins, bins2, df, bootrstrap_size, block_size=block_size)
        else:
            # Otherwise, perform standard conditional expectation calculation
            result = get_exponent_from_simulation_data_conditional_exp_value(fit_function, bins, df, fit_funtion_mapping[fit_function][0], fit_funtion_mapping[fit_function][1], bootrstrap_size, block_size=block_size)
    return result




if __name__ == '__main__':
    keys_of_fit_functions = ['P_of_S', 'P_of_T', 'P_of_L', 'E_of_S_T', 'E_of_T_S', 'E_of_S_L', 'E_of_L_S', 'E_of_T_L', 'E_of_L_T']

    simulation_parameter = {
        'type': 'non_conservative',
        'boundary_condition': 'closed',
    }

    analysis_parameter = {
    'setting1': {
        'fit_function': 'E_of_T_S',
        'variable': 'lifetime',
        'condition': 'total dissipation',
        'xlabel': 's',
        'ylabel': 'E($\\tau$|s)'
    },
    'setting2': {
        'fit_function': 'E_of_S_T',
        'variable': 'total dissipation',
        'condition': 'lifetime',
        'xlabel': '$\\tau$',
        'ylabel': 'E(s|$\\tau$)'
    },
    'setting3': {
        'fit_function': 'E_of_T_L',
        'variable': 'lifetime',
        'condition': 'spatial linear size',
        'xlabel': 'l',
        'ylabel': 'E($\\tau$|l)'
    },
    'setting4': {
        'fit_function': 'E_of_L_T',
        'variable': 'spatial linear size',
        'condition': 'lifetime',
        'xlabel': '$\\tau$',
        'ylabel': 'E(l|$\\tau$)'
    },
    'setting5': {
        'fit_function': 'E_of_L_S',
        'variable': 'spatial linear size',
        'condition': 'total dissipation',
        'xlabel': 's',
        'ylabel': 'E(l|s)'
    },
    'setting6': {
        'fit_function': 'E_of_S_L',
        'variable': 'total dissipation',
        'condition': 'spatial linear size',
        'xlabel': 'l',
        'ylabel': 'E(s|l)'
    },
    'setting7': {
        'fit_function': 'P_of_T',
        'variable': 'lifetime',
        'condition': '-',
        'xlabel': '$\\tau$',
        'ylabel': 'N($\\tau$)'
    },
    'setting8': {
        'fit_function': 'P_of_L',
        'variable': 'spatial linear size',
        'condition': '-',
        'xlabel': 'l',
        'ylabel': 'N(l)'
    },
    'setting9': {
        'fit_function': 'P_of_S',
        'variable': 'total dissipation',
        'condition': '-',
        'xlabel': 's',
        'ylabel': 'N(s)'
    }}

    fit_parameter = {
    'setting1': {
        'bins': np.arange(1,1000),
        'x_limit': [10, 400],
        'bootstrap_size': 100
    },
    'setting2': {
        'bins': np.arange(1,1000),
        'x_limit': [10, 400],
        'bootstrap_size': 100
    },
    'setting3': {
        'bins': np.arange(1,1000),
        'x_limit': [10, 20],
        'bootstrap_size': 100
    },
    'setting4': {
        'bins': np.arange(1,1000),
        'x_limit': [10, 100],
        'bootstrap_size': 100
    },
    'setting5': {
        'bins': np.arange(1,1000),
        'x_limit': [10, 400],
        'bootstrap_size': 100
    },
    'setting6': {
        'bins': np.arange(1,1000),
        'x_limit': [10, 20],
        'bootstrap_size': 100
    },
    'setting7': {
        'bins': np.arange(1,1000),
        'x_limit': [10, 200],
        'bootstrap_size': 100
    },
    'setting8': {
        'bins': np.arange(1,1000),
        'x_limit': [10, 60],
        'bootstrap_size': 100
    },
    'setting9': {
        'bins': np.arange(1,1000),
        'x_limit': [10, 500],
        'bootstrap_size': 100
    }}

    df = load_simulation_data(simulation_parameter)

    file_for_saving = 'exponent_calculation/20240311.csv'

    pbar = tqdm(total = 9, desc ="Running Exponent Calculation")
    for ana_para, fit_para in zip(analysis_parameter, fit_parameter):
        if analysis_parameter[ana_para]['condition'] == '-':
            result = get_exponent_from_simulation_data(analysis_parameter[ana_para]['fit_function'], fit_parameter[fit_para]['bins'], df, analysis_parameter[ana_para]['variable'], fit_parameter[fit_para]['bootstrap_size'], x_limit = fit_parameter[fit_para]['x_limit'])
            save_exponent_data(simulation_parameter, analysis_parameter[ana_para], fit_parameter[fit_para], result, file_for_saving, file_to_load=file_for_saving)
            pbar.update(1)
        else:
            result = get_exponent_from_simulation_data_conditional_exp_value(analysis_parameter[ana_para]['fit_function'], fit_parameter[fit_para]['bins'], df, analysis_parameter[ana_para]['variable'], analysis_parameter[ana_para]['condition'], fit_parameter[fit_para]['bootstrap_size'], x_limit = fit_parameter[fit_para]['x_limit'])
            save_exponent_data(simulation_parameter, analysis_parameter[ana_para], fit_parameter[fit_para], result, file_for_saving, file_to_load=file_for_saving)

        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8, 6), sharex=True)

     
        ax1.errorbar(result["x"], result["data"], yerr=result["errors"], fmt='o', color='blue', capsize=3, markersize=4, label='Simulation Data', zorder=1)
        ax1.plot(result["x"], result["model"], color="orange", linewidth=2, label="Model", zorder=2)
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        ax1.legend()
        ax1.set_ylabel(analysis_parameter[ana_para]['ylabel'])

        
        error_ratio = (result["data"] - result["model"]) / result["errors"]

    
        ax2.scatter(result["x"], error_ratio, color='red')
        ax2.set_xscale('log')
        ax2.set_xlabel(analysis_parameter[ana_para]['xlabel'])
        ax2.set_ylabel("(Data - Model) / Errors")
        ax2.set_ylim(-5, 5)

        plt.tight_layout()
        plt.savefig(f"exponent_calculation/plots/{analysis_parameter[ana_para]['fit_function']}.jpg", dpi=300)
        
        pbar.update(1)
    pbar.close()






