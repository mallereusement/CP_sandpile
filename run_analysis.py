import calc_exponents
import run_sandpile
import argparse
import os
import json
import pandas as pd
import power_spectrum
import numpy as np


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
                if key == 'fit functions':
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
format_int = ['power spectrum R', 'power spectrum T', 'power spectrum N']

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="name of folder where the simulated data gets stored")
    parser.add_argument("simulation_paramter_file", type=str, help="name of file with analysis parameters")
    args = parser.parse_args()

    filepath_datastorage = args.path
    file_path = args.simulation_paramter_file
    simulation_parameters = read_analysis_parameters(file_path, format_bool, format_list, format_int)
    #print(simulation_parameters)

    for parameter in simulation_parameters:
        if 'S_of_f' in simulation_parameters[parameter]['fit functions']:
            l, len_avalanche = power_spectrum.load_data(f'{filepath_datastorage}/{simulation_parameters[parameter]["name"]}/simulation_data/data_for_power_spectrum_calculation.txt')
            max_length = np.max(np.array(len_avalanche))
            R = simulation_parameters[parameter]['power spectrum R']
            N = simulation_parameters[parameter]['power spectrum N']
            T = simulation_parameters[parameter]['power spectrum T']
            power_spectrum_, freq = power_spectrum.calculate_power_spectrum(max_length, R, T, N, l)
            idx = simulation_parameters[parameter]['fit functions'].index('S_of_f')
            


        if (set(simulation_parameters[parameter]['fit functions']) & set(calc_exponents.keys_of_fit_functions)):
            df = pd.read_csv(f'{filepath_datastorage}/{simulation_parameters[parameter]["name"]}/simulation_data/data_for_exponent_calculation.csv', sep=';', encoding='utf8')

            for i in range(len(simulation_parameters[parameter]['fit functions'])):
            if i != idx:
                result = 

    
    