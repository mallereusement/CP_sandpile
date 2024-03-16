import calc_exponents
import run_sandpile
import argparse
import os
import pandas as pd
import power_spectrum
import numpy as np
import matplotlib.pyplot as plt
import plotting

def read_analysis_parameters(file_path, format_bool, format_list, format_int) -> dict:
    parameters = {}
    current_setting = None
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('- setting'):
                current_setting = line.split()[-1].rstrip(':')
                parameters[current_setting] = {}
            elif ':' in line:
                key = line.split(':', 1)[0][2:]
                value = line.split(':', 1)[1]
                if key in format_list_str:
                    parameters[current_setting][key.strip()] = value.strip().strip('][').split(', ')
                elif key in format_int:
                    parameters[current_setting][key.strip()] = int(float(value.strip()))
                elif key in format_list:
                    parameters[current_setting][key.strip()] = list(map(int,map(float, value.strip().strip('][').split(', ') )))
                elif key in format_bool:
                    if value.strip() == 'True':
                        parameters[current_setting][key.strip()] = True
                    elif value.strip() == 'False':
                        parameters[current_setting][key.strip()] = False
                else:
                    parameters[current_setting][key.strip()] = value.strip()
    return parameters



format_bool = ['save plots']
format_list = ['start bins', 'end bins', 'bin width']
format_int = ['power spectrum R', 'power spectrum T', 'power spectrum N', 'bootstrap size']
format_list_str = ['fit functions', 'xlabels', 'ylabels']

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="name of folder where the simulated data gets stored")
    parser.add_argument("analysis_parameter_file", type=str, help="name of file with analysis parameters")
    args = parser.parse_args()

    filepath_datastorage = args.path
    file_path = args.analysis_parameter_file
    analysis_parameters = read_analysis_parameters(file_path, format_bool, format_list, format_int)
    
    
    
    for parameter in analysis_parameters:            
            
        if analysis_parameters[parameter]['save plots']:
            os.mkdir('./' + f'{filepath_datastorage}/{analysis_parameters[parameter]["name"]}/plots')  ## create folder to store plots 
            
            ## load data for z-means:
            means_df = pd.read_csv(f'{filepath_datastorage}/{analysis_parameters[parameter]["name"]}/simulation_data/data_mean.csv', sep=';', encoding='utf8')
            means = means_df['mean'].to_numpy()
            times = means_df['time'].to_numpy()

            ## plot z-means and save plot
            plotting.nice_plot(times*1e-5, means, 't/$10^5$', '<z>', xmin=-0.01, xmax=2,  log=False)
            plt.savefig('./' + f'{filepath_datastorage}/{analysis_parameters[parameter]["name"]}/plots/z_means.jpg', dpi=300)


        run_sandpile.save_simulation_parameters('./' + f'{filepath_datastorage}/{analysis_parameters[parameter]["name"]}/analysis_parameter', analysis_parameters[parameter])
        os.mkdir('./' + f'{filepath_datastorage}/{analysis_parameters[parameter]["name"]}/results')

        


        if 'S_of_f' in analysis_parameters[parameter]['fit functions']:
            l, len_avalanche = power_spectrum.load_data(f'{filepath_datastorage}/{analysis_parameters[parameter]["name"]}/simulation_data/data_for_power_spectrum_calculation.txt')
            max_length = np.max(np.array(len_avalanche))
            R = analysis_parameters[parameter]['power spectrum R']
            N = analysis_parameters[parameter]['power spectrum N']
            T = analysis_parameters[parameter]['power spectrum T']
            power_spectrum_, freq = power_spectrum.calculate_power_spectrum(max_length, R, T, N, l)
            df_power_spectrum = pd.DataFrame({'frequency': freq, 'power spectrum': power_spectrum_})
            df_power_spectrum.to_csv(f'{filepath_datastorage}/{analysis_parameters[parameter]["name"]}/results/power_spectrum.csv', sep=';', encoding='utf8', index=False)
            idx = analysis_parameters[parameter]['fit functions'].index('S_of_f')
            # To Do: Exponent calculation
            if analysis_parameters[parameter]['save plots']:
                ## plot power spectrum and save plot
                plotting.nice_plot(freq, power_spectrum_, 'f', 'S(f)', -4, 0)
                plt.savefig('./' + f'{filepath_datastorage}/{analysis_parameters[parameter]["name"]}/plots/S_of_f.jpg', dpi=300)

                
            


        if (set(analysis_parameters[parameter]['fit functions']) & set(calc_exponents.keys_of_fit_functions)):
            df = pd.read_csv(f'{filepath_datastorage}/{analysis_parameters[parameter]["name"]}/simulation_data/data_for_exponent_calculation.csv', sep=';', encoding='utf8')
            file_count = False
            file_name_exponent_calculation = f'{filepath_datastorage}/{analysis_parameters[parameter]["name"]}/results/results.csv'
            for i in range(len(analysis_parameters[parameter]['fit functions'])):
                if i != idx:
                    bin_start = analysis_parameters[parameter]['start bins'][i]
                    bin_end = analysis_parameters[parameter]['end bins'][i]
                    bin_width = analysis_parameters[parameter]['bin width'][i]
                    bins = [bin_start, bin_end, bin_width]
                    result = calc_exponents.run_calculation(analysis_parameters[parameter]['fit functions'][i], analysis_parameters[parameter]['bootstrap size'], bins, df)
                    
                    if analysis_parameters[parameter]['save plots']:
                        ## plot conditional expectation values and fit and save plots
                        plotting.plot_conditional_exponents(result, analysis_parameters[parameter], i)
                        plt.savefig('./' + f'{filepath_datastorage}/{analysis_parameters[parameter]["name"]}/plots/{analysis_parameters[parameter]["fit functions"][i]}.jpg', dpi=300)
                    
                    
                    if not file_count:
                        calc_exponents.save_exponent_data(analysis_parameters[parameter]['fit functions'][i], bins, analysis_parameters[parameter]['bootstrap size'], result, file_name_exponent_calculation, file_to_load=False)
                        file_count = True
                    elif file_count:
                        calc_exponents.save_exponent_data(analysis_parameters[parameter]['fit functions'][i], bins, analysis_parameters[parameter]['bootstrap size'], result, file_name_exponent_calculation, file_to_load=file_name_exponent_calculation)
        
    