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
    """calculates the conditional expectation value E(variable|condition)

    Args:
        variable (str): variable
        condition (str): condition
        bins (np.ndarray): bins to use
        df (pd.DataFrame): simulation data
        x_limit (list): Default to  []. Set interval to use

    Returns:
        np.ndarray, unp.uarray: array with x values and the conditional expectation values
    """
    if x_limit:
        bins = exclude_outside_interval(bins, False, *x_limit)

    bin_centers = (bins[:-1] + bins[1:]) / 2

    expectation_list = []
    expectation_list_err = []

    for left_edge, right_edge in zip(bins[:-1], bins[1:]):
        list = df[(df[condition] >= left_edge) & (df[condition] < right_edge)][variable].to_numpy()

        if np.any(list):

            list = unp.uarray(list, np.sqrt(list))    
            expectation = np.sum(list) / len(list)
                
            expectation_list.append(unp.nominal_values(expectation))
            expectation_list_err.append(unp.std_devs(expectation))
        else:
            expectation_list.append(0)   
            expectation_list_err.append(1)     
    expectation_list = unp.uarray(expectation_list, expectation_list_err)


    if get_error_with_bootstrapping:
        pbar = tqdm(total = bootstrap_size, desc ="Running Bootstrap for Error Estimation of conditional expectation value")
        samples = generate_bootstrap_samples(df, bootstrap_size)
        expectation_list_bootstrap_global = []
        for sample in samples:
            expectation_list_bootstrap = []
            for left_edge, right_edge in zip(bins[:-1], bins[1:]):
                list = sample[(sample[condition] >= left_edge) & (sample[condition] < right_edge)][variable].to_numpy()

                if np.any(list):
                    
                    expectation = np.sum(list) / len(list)
                    
                    expectation_list_bootstrap.append(expectation)
                else:
                    expectation_list_bootstrap.append(0)   
            pbar.update(1)
            expectation_list_bootstrap = np.array(expectation_list_bootstrap)
            expectation_list_bootstrap_global.append(expectation_list_bootstrap)
        pbar.close()
        expectation_list_bootstrap_global = np.array(expectation_list_bootstrap_global)
        stds = np.std(expectation_list_bootstrap_global, axis=0)
        expectation_list = unp.uarray(unp.nominal_values(expectation_list), stds)

    


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

    truncated_bin_edges = bin_edges[min_index:max_index]
    if np.any(histogram_data):
        truncated_histogram_data = histogram_data[min_index:max_index-1]
        return truncated_bin_edges, truncated_histogram_data
    else:
        return truncated_bin_edges


def get_exponent_from_simulation_data_power_spectrum(fit_function: str, bins: np.ndarray, df: pd.DataFrame, bootstrap_size: int, x_limit: list=[], starting_values: list = [1, 1]) -> dict: # ToDo
    pass
    #samples = generate_bootstrap_samples(df, bootstrap_size)

def get_exponent_product_from_simulation_data_conditional_exp_value(fit_function: str, bins1: np.ndarray, bins2: np.ndarray, df: pd.DataFrame, bootstrap_size: int, x_limit1: list=[], x_limit2: list=[], starting_values: list = [1, 1], block_size: int=0) -> dict: # ToDo
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
    samples = generate_bootstrap_samples(df, bootstrap_size, block_size=block_size)
    product = []

    if fit_function == 'gamma1_gamma3_1':
        x_org1, data_org1 = conditional_expectation_value('total dissipation', 'lifetime', bins1, df, x_limit1, get_error_with_bootstrapping=True, bootstrap_size=bootstrap_size)
        m_org1 = fit_data(fit_function, x_org1, unp.nominal_values(data_org1), unp.std_devs(data_org1), starting_values)
    
        x_org2, data_org2 = conditional_expectation_value('lifetime', 'spatial linear size', bins2, df, x_limit2, get_error_with_bootstrapping=True, bootstrap_size=bootstrap_size)
        m_org2 = fit_data(fit_function, x_org2, unp.nominal_values(data_org2), unp.std_devs(data_org2), starting_values)
    elif fit_function == 'gamma1_gamma3_2':
        x_org1, data_org1 = conditional_expectation_value('lifetime', 'total dissipation', bins1, df, x_limit1, get_error_with_bootstrapping=True, bootstrap_size=bootstrap_size)
        m_org1 = fit_data(fit_function, x_org1, unp.nominal_values(data_org1), unp.std_devs(data_org1), starting_values)
    
        x_org2, data_org2 = conditional_expectation_value('spatial linear size', 'lifetime', bins2, df, x_limit2, get_error_with_bootstrapping=True, bootstrap_size=bootstrap_size)
        m_org2 = fit_data(fit_function, x_org2, unp.nominal_values(data_org2), unp.std_devs(data_org2), starting_values)
    valid_counter = 0
    for sample in samples:
        if fit_function == 'gamma1_gamma3_1':      
            x1, data1 = conditional_expectation_value('total dissipation', 'lifetime', bins1, sample, x_limit1)
            m1 = fit_data('E_of_S_T', x1, unp.nominal_values(data1), unp.std_devs(data_org1), starting_values)
            x2, data2 = conditional_expectation_value('lifetime', 'spatial linear size', bins2, sample, x_limit2)
            m2 = fit_data('E_of_T_L', x2, unp.nominal_values(data2), unp.std_devs(data_org2), starting_values)
            if (m1.valid == True) and (m2.valid == True):
                valid_counter = valid_counter + 1
                parameter_amp1, parameter_exp1 = m1.values['amp'], m1.values['exponent']
                parameter_amp2, parameter_exp2 = m2.values['amp'], m2.values['exponent']
                product.append(parameter_exp1*parameter_exp2)
        elif fit_function == 'gamma1_gamma3_2':
            x1, data1 = conditional_expectation_value('lifetime', 'total dissipation', bins1, sample, x_limit1)
            m1 = fit_data('E_of_S_T', x1, unp.nominal_values(data1), unp.std_devs(data_org1), starting_values)
            x2, data2 = conditional_expectation_value('spatial linear size', 'lifetime', bins2, sample, x_limit2)
            m2 = fit_data('E_of_T_L', x2, unp.nominal_values(data2), unp.std_devs(data_org2), starting_values)
            if (m1.valid == True) and (m2.valid == True):
                valid_counter = valid_counter + 1
                parameter_amp1, parameter_exp1 = m1.values['amp'], m1.values['exponent']
                parameter_amp2, parameter_exp2 = m2.values['amp'], m2.values['exponent']
                product.append(parameter_exp1*parameter_exp2)           

    product = np.array(product)
    org_product = m_org1.values['exponent'] * m_org2.values['exponent']

    std = np.std(product)

    return {"product": unc.ufloat(org_product, std), "products_from_bootstrap": product, "samples": samples, "number_of_valid_fits": valid_counter}



def get_exponent_from_simulation_data_conditional_exp_value(fit_function: str, bins: np.ndarray, df: pd.DataFrame, variable: str, condition: str, bootstrap_size: int, x_limit: list=[], starting_values: list = [1, 1], block_size: int=0) -> dict: # ToDo
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
    samples = generate_bootstrap_samples(df, bootstrap_size, block_size=block_size)


    parameters_amp = []
    parameters_exp = []
    x_org, data_org = conditional_expectation_value(variable, condition, bins, df, x_limit, get_error_with_bootstrapping=True, bootstrap_size=bootstrap_size)
    m_org = fit_data(fit_function, x_org, unp.nominal_values(data_org), unp.std_devs(data_org), starting_values)

    for sample in samples:
        x, data = conditional_expectation_value(variable, condition, bins, sample, x_limit)
        m = fit_data(fit_function, x, unp.nominal_values(data), unp.std_devs(data_org), starting_values)
        if m.valid:
            parameter_amp, parameter_exp = m.values['amp'], m.values['exponent']
            parameters_amp.append(parameter_amp)
            parameters_exp.append(parameter_exp)

    parameters_amp = np.array(parameters_amp)
    parameters_exp = np.array(parameters_exp)

    cov_mat = np.cov(np.stack((parameters_amp, parameters_exp), axis = 0))

    return {"exponent_values_from_bootrstrap": parameters_exp, "parameters": [unc.ufloat(m_org.values['amp'], cov_mat[0,0]**0.5), unc.ufloat(m_org.values['exponent'], cov_mat[1,1]**0.5)], "covariance_matrix": cov_mat, "x": x, "data": unp.nominal_values(data_org), "errors": unp.std_devs(data_org), "model": fit_functions[fit_function](x, m_org.values['amp'], m_org.values['exponent']), "samples": samples}


def generate_following_indices(indices, n):
    result = []
    for index in indices:
        result.extend(range(index, index + n ))
    return result


def generate_bootstrap_samples(data: pd.DataFrame, bootstrap_size: int, block_size: int=0):
    """Generate bootstrap sample of size bootstrap_size for given simulation data

    Args:
        data (pd.DataFrame): simulation data
        bootstrap_size (int): number of bootstrap samples
        block_size (int): block_size for bootrstrapping because of correlation of time series

    Returns:
        pd.DataFrame: bootstrap_size bootstrap samples
    """
    if block_size == 0:
        return [data.sample(data.shape[0], replace=True) for _ in range(bootstrap_size)]
    else:
        numb = math.floor(data.shape[0] / block_size)
        samples = []
        for x in range(bootstrap_size):
            sample_f_d = np.random.randint(0, numb, numb)
            sample_f_d = generate_following_indices(sample_f_d, block_size)
            sample = df.iloc[sample_f_d]
            samples.append(sample)
        return samples

def get_exponent_from_simulation_data(fit_function: str, bins: np.ndarray, df: pd.DataFrame, variable: str, bootstrap_size: int, x_limit: list=[], starting_values = [1,1], block_size: int=0) -> dict:
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
    samples = generate_bootstrap_samples(df, bootstrap_size, block_size=block_size)
    hist_data = [np.histogram(i[variable], bins=bins)[0] for i in samples]
    samples = np.array([i for i in hist_data])
    errors = np.std(samples, axis=0)
    parameters_amp = []
    parameters_exp = []
    m_org = fit_data(fit_function, bin_centers, data, errors, starting_values)

    for sample in samples:

        sample[sample == 0] = 1e-7 # avoid division by zero
        sample = sample.astype(np.longdouble)
        m = fit_data(fit_function, bin_centers, sample, errors, starting_values)
        if m.valid:
            parameter_amp, parameter_exp = m.values['amp'], m.values['exponent']
            parameters_amp.append(parameter_amp)
            parameters_exp.append(parameter_exp)

    parameters_amp = np.array(parameters_amp)
    parameters_exp = np.array(parameters_exp)

    cov_mat = np.cov(np.stack((parameters_amp, parameters_exp), axis = 0))

    return {"exponent_values_from_bootrstrap": parameters_exp, "parameters": [unc.ufloat(m_org.values['amp'], cov_mat[0,0]**0.5), unc.ufloat(m_org.values['exponent'], cov_mat[1,1]**0.5)], "covariance_matrix": cov_mat, "x": bin_centers, "data": data, "errors": errors, "model": fit_functions[fit_function](bin_centers, m_org.values['amp'], m_org.values['exponent']), "samples": samples}


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
    m.limits = [(0, None), (0.2, 5)]
    m.migrad()
    return m



def save_exponent_data(fit_function: str, bins: dict, bootstrap_size: str, fit_results: dict, file_to_save, file_to_load=False, bins2=False) -> None:
    """_summary_

    Args:
        simulation_parameters (dict): _description_
        analysis_parameters (dict): _description_
        fit_parameter (dict): _description_
        fit_results (dict): _description_
        file_to_save (_type_): _description_
        file_to_load (bool, optional): _description_. Defaults to False.
    """
    #if fit_parameter['x_limit']:
    #    bins_start = fit_parameter['x_limit'][0]
    #    bins_end = fit_parameter['x_limit'][1]
    #    min_index = np.searchsorted(fit_parameter['bins'], bins_start, side='left')
    #    max_index = np.searchsorted(fit_parameter['bins'], bins_end, side='right')
    #    bins_count = len(fit_parameter['bins'][min_index:max_index])
    #else:
    #    bins_start = fit_parameter['bins'][0]
    #    bins_end = fit_parameter['bins'][-1]       
    #    bins_count = len(fit_parameter['bins'])
    if bins2 == False:
        bins = get_bins_from_parameter_settings(*bins)
        temp_df = pd.DataFrame({'fit function': [fit_function], 'variable': [fit_funtion_mapping[fit_function][0]], 'condition': [fit_funtion_mapping[fit_function][1]], 'left bin edge': [bins[0]], 'right bin edge': [bins[-1]], 'count of bins': [len(bins)-1], 'bootstrap size': [bootstrap_size], 'amplitude from fit result': [fit_results['parameters'][0]], 'exponent from fit result': [fit_results['parameters'][1]], 'covariance c_11': [fit_results['covariance_matrix'][0,0]], 'covariance c_12': [fit_results['covariance_matrix'][0,1]], 'covariance c_22': [fit_results['covariance_matrix'][1,1]]})
        
        if not file_to_load:
            temp_df.to_csv(file_to_save, sep=';', encoding='utf8', index=False)
        else:
            df = pd.read_csv(file_to_load, sep=';', encoding='utf8')
            df = pd.concat([df, temp_df], ignore_index=True)
            df.to_csv(file_to_save, sep=';', encoding='utf8', index=False)
    else:
        bins = get_bins_from_parameter_settings(*bins)
        bins2 = get_bins_from_parameter_settings(*bins2)
        temp_df = pd.DataFrame({'fit function': [fit_function], 'left bin edge 1': [bins[0]], 'right bin edge 1': [bins[-1]], 'count of bins 1': [len(bins)-1], 'left bin edge 2': [bins2[0]], 'right bin edge 2': [bins2[-1]], 'count of bins 2': [len(bins2)-1], 'bootstrap size': [bootstrap_size], 'product from fit result': [fit_results['product']], 'number of valid fits': [fit_results['number_of_valid_fits']]})
        
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
     return np.linspace(start_bin, end_bin, int((end_bin-start_bin) / float(bin_width)))


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
    """ Function takes the fit function and automatically decided which exponent should get calculated

    Args:
        fit_function (str): name of fit function of exponent
        bootrstrap_size (int): size of bootstrap for error calculation of fit
        bins (list): [start bin, end bin, width of bin]
        df (pd.DataFrame): simulated data

    Returns:
        _type_: fit results
    """
    bins = get_bins_from_parameter_settings(*bins)
    if fit_funtion_mapping[fit_function][1] == '-':
        if fit_function == 'S_of_f':
            result = get_exponent_from_simulation_data_power_spectrum() # To Do
        else:
            result = get_exponent_from_simulation_data(fit_function, bins, df, fit_funtion_mapping[fit_function][0], bootrstrap_size, block_size=block_size)

    else:
        if fit_function == 'gamma1_gamma3_1':
            result = get_exponent_product_from_simulation_data_conditional_exp_value('gamma1_gamma3_1', bins, bins2, df, bootrstrap_size, block_size=block_size)
        elif fit_function == 'gamma1_gamma3_2':
            result = get_exponent_product_from_simulation_data_conditional_exp_value('gamma1_gamma3_2', bins, bins2, df, bootrstrap_size, block_size=block_size)
        else:
            result = get_exponent_from_simulation_data_conditional_exp_value(fit_function, bins, df, fit_funtion_mapping[fit_function][0], fit_funtion_mapping[fit_function][1], bootrstrap_size, block_size=block_size)
    return result

def calculate_products_of_exponents():
    pass


if __name__ == '__main__':
    keys_of_fit_functions = ['P_of_S', 'P_of_T', 'P_of_L', 'E_of_S_T', 'E_of_T_S', 'E_of_S_L', 'E_of_L_S', 'E_of_T_L', 'E_of_L_T']

    

    #### To Do #####
    ## exponent calculation for power spectrum
    ## calculate product of expontents and also take covariance in account here

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






