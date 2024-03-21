import simulation_algorithm as sim_alg # Importing custom simulation algorithm module
import argparse
import os
import json

def read_simulation_parameters(file_path, format_int, format_bool) -> dict:
    """Read simulation parameters from a file and parse them into a dictionary.

    Args:
        file_path (str): Path to the file containing simulation parameters.
        format_int (list): List of parameter keys that should be formatted as integers.
        format_bool (list): List of parameter keys that should be formatted as boolean values.

    Returns:
        dict: Dictionary containing simulation parameters.
    """
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
                if key in format_int:
                    parameters[current_setting][key.strip()] = int(float(value.strip()))
                elif key in format_bool:
                    if value.strip() == 'True':
                        parameters[current_setting][key.strip()] = True
                    elif value.strip() == 'False':
                        parameters[current_setting][key.strip()] = False
                else:
                    parameters[current_setting][key.strip()] = value.strip()
    return parameters

def save_simulation_parameters(filepath:str, parameters: dict) -> None:
    """Save simulation parameters to a JSON file.

    Args:
        filepath (str): Path to the JSON file where parameters will be saved.
        parameters (dict): Dictionary containing simulation parameters.
    """
    with open(filepath, 'w') as json_file:
        json.dump(parameters, json_file, indent=4)

# Lists with keys that should get casted to int or bool
format_int = ['dimension', 'size of grid', 'crititcal value of z', 'number of activated avalanches', 'maximum time steps', 'steady state']
format_bool = ['use absolute value', 'save file for power spectrum calculation', 'save file for exponent calculation', 'save mean value of grid', 'track avalanches after steady state']

if __name__ == '__main__':

    # Parsing command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="name of folder where the simulated data gets stored")
    parser.add_argument("simulation_paramter_file", type=str, help="name of file with simulation parameters")
    args = parser.parse_args()

    filepath_datastorage = args.path
    file_path = args.simulation_paramter_file
    # Reading simulation parameters from file
    simulation_parameters = read_simulation_parameters(file_path, format_int, format_bool)
    # Creating directories for storing simulation data and plots
    os.mkdir('./' + args.path)
    os.mkdir('./' + args.path + f'/plots')

    # Iterating over simulation parameters
    for parameter in simulation_parameters:
       
        os.mkdir('./' + args.path + f'/{simulation_parameters[parameter]["name"]}')   ## create path for specific simulation
        os.mkdir('./' + args.path + f'/{simulation_parameters[parameter]["name"]}/simulation_data')  ## create folder to store simulation data in
        
        os.mkdir('./' + args.path + f'/plots/{simulation_parameters[parameter]["name"]}')  ## create folder for the specific simulation
        # Saving simulation parameters to a JSON file
        save_simulation_parameters('./' + args.path + f'/{simulation_parameters[parameter]["name"]}/simulation_data/simulation_parameter', simulation_parameters[parameter])
        # Running simulation based on the parameters
        sim_alg.run_simulation(simulation_parameters[parameter], './' + args.path, simulation_parameters[parameter]["name"])