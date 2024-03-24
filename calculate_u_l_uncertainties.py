import numpy as np
import pandas as pd
import run_analysis as ra
import matplotlib.pyplot as plt


file_path = 'analysis_parameter.txt'
file_path_sim_data = 'final_data'

def calculate_uncertainties(up_list, low_list, results):
    unc_up = up_list - results
    unc_lo = results - low_list
    
    ## set all invalid uncertainties 0.1
    unc_up[np.where(unc_up < 0)] = 0.1
    unc_lo[np.where(unc_lo < 0)] = 0.1
    
    return unc_lo, unc_up

def calculate_tot_uncertainties(unc1, unc2):
    return np.sqrt(unc1**2 + unc2**2)


## set up figure for plot
plot, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(9,6), sharex=True)
plt.subplots_adjust(hspace=0, wspace=0)

analysis_parameters = ra.read_analysis_parameters(file_path, ra.format_bool, ra.format_list, ra.format_int, ra.format_float, ra.format_list_str)

params_names = ['$\\tau$', '$\\alpha$', '$\\lambda$', '$\\gamma_1$', '$\\gamma_2$',  '$\\gamma_3$', '$\\gamma_1\\gamma_3$']
positions2d = np.array([10,20,30,40,44,55,59,70,74,85,89])
positions = np.array([10,20,30,40,41,55,56,70,71,85,86])
markers = ['o', 'o', 'o', 'o', 'v', 'o', 'v', 'o', 'v', 'o', 'v']

colors2d = ['tab:blue', 'green', 'red', 'orange']

vlines = [15, 25, 35,  50, 65, 80]

i2d = 1

for i, parameter in enumerate(analysis_parameters):
    name = analysis_parameters[parameter]['name']
    
    upper_bound = pd.read_csv(f'{file_path_sim_data}/results/{name}/upper_bound/results.csv', sep=';')['exponent from fit result'].apply(lambda x: float(x.split('+/-')[0])).to_numpy()
    lower_bound = pd.read_csv(f'{file_path_sim_data}/results/{name}/lower_bound/results.csv', sep=';')['exponent from fit result'].apply(lambda x: float(x.split('+/-')[0])).to_numpy()
    result_df = pd.read_csv(f'{file_path_sim_data}/results/{name}/main/results.csv', sep=';')
    result = result_df['exponent from fit result'].apply(lambda x: float(x.split('+/-')[0])).to_numpy()
    result_err = result_df['exponent from fit result'].apply(lambda x: float(x.split('+/-')[1])).to_numpy()
    
    upper_bound_prod = pd.read_csv(f'{file_path_sim_data}/results/{name}/upper_bound/results_products.csv', sep=';')['product from fit result'].apply(lambda x: float(x.split('+/-')[0])).to_numpy()
    lower_bound_prod = pd.read_csv(f'{file_path_sim_data}/results/{name}/lower_bound/results_products.csv', sep=';')['product from fit result'].apply(lambda x: float(x.split('+/-')[0])).to_numpy()
    result_prod_df = pd.read_csv(f'{file_path_sim_data}/results/{name}/main/results_products.csv', sep=';')
    result_prod = result_prod_df['product from fit result'].apply(lambda x: float(x.split('+/-')[0])).to_numpy()
    result_prod_err = result_prod_df['product from fit result'].apply(lambda x: float(x.split('+/-')[1])).to_numpy()
    
    
    ## combine two arrays into one
    results_arr = np.append(result, result_prod)
    results_err_arr = np.append(result_err, result_prod_err)
    up_list = np.append(upper_bound, upper_bound_prod)
    low_list = np.append(lower_bound, lower_bound_prod)
    
    ## create dataframe to store data in
    df = pd.DataFrame()
    exp_name = np.append(result_df['fit function'], result_prod_df['fit function'])
   
    ## calculate lower and upper uncertainties
    unc_lo_arr, unc_up_arr = calculate_uncertainties(up_list, low_list, results_arr)
    
    unc_lo_arr = calculate_tot_uncertainties(results_err_arr, unc_lo_arr)
    unc_up_arr = calculate_tot_uncertainties(results_err_arr, unc_up_arr)
    
    
    ## save data in dataframe
    df['name'] = exp_name
    df['unc_up'] = unc_up_arr
    df['unc_lo'] = unc_lo_arr
    
    ## save df to csv-file
    df.to_csv(f'{file_path_sim_data}/results/{name}/sys_uncertainties.csv', sep=',')
    
    N = 100
    if 'N40' in name:
        N = 40
    if 'N20' in name:
        N = 20
    
    if '-non_conservative-' in name:
        perturbation = 'non conservative'
    else:
        perturbation = 'conservative'
    
    if 'open' in name:
        bounds = 'open'
    else:
        bounds = 'closed'
    
    z = 4
    if 'z3' in name:
        z = 3
    
    labels = [i2d, False, False, False, False, False, False, False, False, False, False]
    ## plot
    for n, point in enumerate(results_arr):
        if labels[n]:
            if '2d' in name:
                ax1.errorbar(positions2d[n], results_arr[n], yerr=np.array([[unc_lo_arr[n]], [unc_up_arr[n]]]), fmt=markers[n], capsize=3, markersize=4, label = labels[n], color=colors2d[i2d-1])
                ## move positions where data is plotted slightly for better visualization
                positions2d[n] = 1. + positions2d[n]
            elif '3d' in name:
                ax2.errorbar(positions[n], results_arr[n], yerr=np.array([[unc_lo_arr[n]], [unc_up_arr[n]]]), fmt=markers[n], capsize=3, markersize=4, label='5', color='purple')
            elif '4d' in name:
                ax3.errorbar(positions[n], results_arr[n], yerr=np.array([[unc_lo_arr[n]], [unc_up_arr[n]]]), fmt=markers[n], capsize=3, markersize=4, label='6', color='brown')
        else: 
            if '2d' in name:
                ax1.errorbar(positions2d[n], results_arr[n], yerr=np.array([[unc_lo_arr[n]], [unc_up_arr[n]]]), fmt=markers[n], capsize=3, markersize=4,  color=colors2d[i2d-1]) 
                ## move positions where data is plotted slightly for better visualization
                positions2d[n] = 1. + positions2d[n]
            elif '3d' in name:
                ax2.errorbar(positions[n], results_arr[n], yerr=np.array([[unc_lo_arr[n]], [unc_up_arr[n]]]), fmt=markers[n], capsize=3, markersize=4, color='purple')
            elif '4d' in name:
                ax3.errorbar(positions[n], results_arr[n], yerr=np.array([[unc_lo_arr[n]], [unc_up_arr[n]]]), fmt=markers[n], capsize=3, markersize=4, color='brown')
        
    if '2d' in name:
        i2d +=1
    
## include literature values
lit_2d = [2, np.nan, np.nan, 1.55, 2.08, 1.34, 2.08]
lit_3d = [2.37, np.nan, np.nan, 1.7, 2.72, 1.61, 2.74]
lit_4d = [2.5, np.nan, np.nan, 1.79, 3.09, 1.8, 3.22]
positions_lit = [9,19,29,39,54,69,84]


ax1.errorbar(positions_lit, lit_2d, yerr=0.1, fmt='o', capsize=3, markersize=4, label='lit', color='black')
ax2.errorbar(positions_lit, lit_3d, yerr=0.1, fmt='o', capsize=3, markersize=4, label='lit', color='black')
ax3.errorbar(positions_lit, lit_4d, yerr=0.1, fmt='o', capsize=3, markersize=4, label='lit', color='black')  

ax1.text(0.01,0.01, "2d", fontsize=14, transform=ax1.transAxes, horizontalalignment='left', verticalalignment='bottom')
ax2.text(0.01,0.01, "3d", fontsize=14, transform=ax2.transAxes, horizontalalignment='left', verticalalignment='bottom')
ax3.text(0.01,0.01, "4d", fontsize=14, transform=ax3.transAxes, horizontalalignment='left', verticalalignment='bottom')

ax1.set_xlim(0, 110)
ax2.set_xlim(0,110)
ax3.set_xlim(0,110)

ax1.legend(loc='lower right', fontsize=12)
ax2.legend(loc='lower right', fontsize=12)
ax3.legend(loc='lower right', fontsize=12)

for line in vlines:
    ax1.axvline(line, color='gray', linestyle='--')
    ax2.axvline(line, color='gray', linestyle='--')
    ax3.axvline(line, color='gray', linestyle='--')

plt.xticks(positions_lit,params_names, fontsize=14)


plt.savefig('report_plots/uncertainties_plot.jpg', dpi=300)
plt.show()