import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import simulation_algorithm as sim_alg # Importing custom simulation algorithm module
import argparse
import os
import matplotlib.animation as animation


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


# Lists with keys that should get casted to int or bool
format_int = ['dimension', 'size of grid', 'crititcal value of z', 'number of activated avalanches', 'maximum time steps', 'steady state', 'number of avalanches to safe', 'minimum time of avalanche']
format_bool = ['use absolute value', 'save file for power spectrum calculation', 'save file for exponent calculation', 'save mean value of grid', 'track avalanches after steady state', 'save animation of evolution to steady state', 'save animation of avalanches', 'take snapshot of avalanches', 'save multiple avalanches']


if __name__ == '__main__':

    # Parsing command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("animation_parameter_file", type=str, help="name of file with animation parameters")
    args = parser.parse_args()

    file_path = args.animation_parameter_file
    # Reading simulation parameters from file
    animation_parameters = read_simulation_parameters(file_path, format_int, format_bool)
    # Creating directories for storing simulation data and plots
    try:
        os.mkdir('.'+ f'/animations')
    except:
        pass

    # Iterating over simulation parameters
    for parameter in animation_parameters:
        try:
            os.mkdir('.' + f'/animations/{animation_parameters[parameter]["name"]}')   ## create path for specific animation
        except:
            pass

        # Running simulation to create animations based on the parameters
        sim_alg.run_simulation(animation_parameters[parameter], '.' + f'/animations', animation_parameters[parameter]["name"], safe_data_for_animations=True)
        if animation_parameters[parameter]['save animation of evolution to steady state']:
            loaded_array = np.load('.' + f'/animations/{animation_parameters[parameter]["name"]}/evo_to_steady_state.npy')
            os.remove('.' + f'/animations/{animation_parameters[parameter]["name"]}/evo_to_steady_state.npy')

            fig, ax = plt.subplots()

            # Create an empty heatmap plot
            heatmap = ax.imshow(loaded_array[0], cmap='hot', interpolation='nearest', vmax=5)

            # Function to update the heatmap for each frame
            def update(frame):
                heatmap.set_array(loaded_array[frame])
                return [heatmap]

            # Create the animation
            ani = animation.FuncAnimation(fig, update, frames=loaded_array.shape[0], interval=50, blit=True)

            # Save the animation as a video file (MP4 format)
            FPS = 240
            ani.save('.' + f'/animations/{animation_parameters[parameter]["name"]}/evolution_to_steady_state.mp4', writer='ffmpeg', fps=FPS)
        if animation_parameters[parameter]['save animation of avalanches']:
            loaded_array = np.load('.' + f'/animations/{animation_parameters[parameter]["name"]}/ava_tracker.npy')
            os.remove('.' + f'/animations/{animation_parameters[parameter]["name"]}/ava_tracker.npy')

            fig, ax = plt.subplots()

            # Create an empty heatmap plot
            heatmap = ax.imshow(loaded_array[0], cmap='hot', interpolation='nearest', vmax=np.max(loaded_array))

            # Function to update the heatmap for each frame
            def update(frame):
                heatmap.set_array(loaded_array[frame])
                return [heatmap]

            # Create the animation
            ani = animation.FuncAnimation(fig, update, frames=loaded_array.shape[0], interval=50, blit=True)

            # Save the animation as a video file (MP4 format)
            FPS = 30
            if animation_parameters[parameter]['save multiple avalanches']:
                ani.save('.' + f'/animations/{animation_parameters[parameter]["name"]}/multiple_ava_tracker.mp4', writer='ffmpeg', fps=FPS)
            else:
                ani.save('.' + f'/animations/{animation_parameters[parameter]["name"]}/one_ava_tracker.mp4', writer='ffmpeg', fps=FPS)