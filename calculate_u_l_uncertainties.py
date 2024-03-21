import numpy as np
import pandas as pd
import run_analysis as ra


file_path = 'analysis_parameter.txt'

analysis_parameters = ra.read_analysis_parameters(file_path, ra.format_bool, ra.format_list, ra.format_int, ra.format_float, ra.format_list_str)

for parameter in analysis_parameters:
    name = analysis_parameters[parameter]['name']
    
    upper_bound = pd.read_csv(f'final_data2/results/{name}/upper_bound/results.csv', sep=';')['exponent from fit result'].apply(lambda x: float(x.split('+/-')[0])).to_numpy()
    lower_bound = pd.read_csv(f'final_data2/results/{name}/lower_bound/results.csv', sep=';')['exponent from fit result'].apply(lambda x: float(x.split('+/-')[0])).to_numpy()
    result_df = pd.read_csv(f'final_data2/results/{name}/ana2/results.csv', sep=';')
    
    result = result_df['exponent from fit result'].apply(lambda x: float(x.split('+/-')[0])).to_numpy()
    
    upper_bound_prod = pd.read_csv(f'final_data2/results/{name}/upper_bound/results_products.csv', sep=';')['product from fit result'].apply(lambda x: float(x.split('+/-')[0])).to_numpy()
    lower_bound_prod = pd.read_csv(f'final_data2/results/{name}/lower_bound/results_products.csv', sep=';')['product from fit result'].apply(lambda x: float(x.split('+/-')[0])).to_numpy()
    result_prod_df = pd.read_csv(f'final_data2/results/{name}/ana2/results_products.csv', sep=';')
       
    result_prod = result_prod_df['product from fit result'].apply(lambda x: float(x.split('+/-')[0])).to_numpy()
    
    unc_up = upper_bound - result
    unc_lo = result - lower_bound
    
    unc_up_prod= upper_bound_prod - result_prod
    unc_lo_prod = result_prod - lower_bound_prod
    
    df = pd.DataFrame()
    
    exp_name = result_df['fit function']
    prod_name = result_prod_df['fit function']
    
    df['name'] = np.append(exp_name, prod_name)
    df['unc_up'] = np.append(unc_up, unc_up_prod)
    df['unc_lo'] = np.append(unc_lo, unc_lo_prod)
    
    df.to_csv(f'final_data2/results/{name}/sys_uncertainties.csv', sep=',')
    