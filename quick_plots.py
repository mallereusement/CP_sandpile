import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import calc_exponents
from uncertainties import unumpy as unp

df = pd.read_csv('final_data2/3d-non_conservative-N20-abs_True_z4-closed/simulation_data/data_for_exponent_calculation.csv', sep=';')
fit_funtion_mapping = {
    'P_of_S': ['total dissipation', '-'],
    'P_of_T': ['lifetime', '-'],
    'P_of_L': ['spatial linear size', '-'],
    'E_of_S_T': ['total dissipation', 'lifetime'],
    'E_of_T_S': ['lifetime', 'total dissipation'],
    'E_of_S_L': ['total dissipation', 'spatial linear size'],
    'E_of_L_S': ['spatial linear size', 'total dissipation'],
    'E_of_T_L': ['lifetime', 'spatial linear size'],
    'E_of_L_T': ['spatial linear size', 'lifetime']
}

function_names = ['N(S=s)', 'N(T=$t_l$)', 'N(L=l)', 'E(S|T=$t_l$)', 'E(T|S=s)', 'E(S|L=l)', 'E(L|S=s)', 'E(T|L=l)', 'E(L|T=$t_l$)']
function_vars = ['s', '$t_l', 'l', '$t_l$', 's', 'l', 's', 'l', '$t_l$', 'f']

#time = df['time'].to_numpy()
#mean = df['mean'].to_numpy()

#plt.plot(time, mean)
#plt.show()

for i, function in enumerate(fit_funtion_mapping):

    variable = fit_funtion_mapping[function][0]
    condition = fit_funtion_mapping[function][1]
    
    if condition == '-':
        bin_list = np.arange(0,max(df[variable]))
        hist, _ = np.histogram(df[variable], bins=bin_list)
        plt.plot(bin_list[1:-1],hist[1:], color='blue')
    else:
        x, result, _ = calc_exponents.conditional_expectation_value(variable, condition, np.arange(0.5, max(df[condition])+0.5, 1), df)
        result = unp.nominal_values(result)
        plt.errorbar(x, result, fmt='o', color='blue', capsize=3, markersize=4, label='Simulation Data', zorder=1)
        
    #plt.title(function)
    plt.xscale('log')
    plt.yscale('log')    
    plt.ylabel(function_names[i], fontsize='14')
    plt.xlabel(function_vars[i], fontsize='14')
    plt.tick_params(axis='both', which='major', labelsize='14')
    plt.show()