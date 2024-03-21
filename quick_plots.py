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


#time = df['time'].to_numpy()
#mean = df['mean'].to_numpy()

#plt.plot(time, mean)
#plt.show()

for function in fit_funtion_mapping:

    variable = fit_funtion_mapping[function][0]
    condition = fit_funtion_mapping[function][1]
    
    if condition == '-':
        hist, _ = np.histogram(df[variable], bins=np.arange(0,400))
        plt.plot(np.arange(1,399),hist[1:])
        try:
            plt.axvline(np.min(np.where(hist[1:]==0)))
        except:
            pass
    else:
        x, result, _ = calc_exponents.conditional_expectation_value(variable, condition, np.arange(0.5, max(df[condition])+0.5, 1), df)
        result = unp.nominal_values(result)
        plt.scatter(x, result)
        try:
            plt.axvline(np.min(np.where(result==0)))
        except:
            pass
        
    plt.title(function)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
