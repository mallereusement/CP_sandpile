import calc_exponents
import run_sandpile
import argparse
import os
import json


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="name of folder where the simulated data gets stored")
    parser.add_argument("simulation_paramter_file", type=str, help="name of file with simulation parameters")
    args = parser.parse_args()

    filepath_datastorage = args.path
    file_path = args.simulation_paramter_file
    simulation_parameters = run_sandpile.read_simulation_parameters(file_path, run_sandpile.format_int, run_sandpile.format_bool)
    os.mkdir('./' + args.path)
    
    