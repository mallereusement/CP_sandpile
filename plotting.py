import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_power_spectrum(freq, power_spectrum):
    """Plot power spectrum.

    Args:
        freq (array-like): Frequency data.
        power_spectrum (array-like): Power spectrum data.

    Returns:
        None
    """
    plt.plot(np.log(freq), np.log(power_spectrum), color='black')
    plt.xlabel('$log(f)$')
    plt.ylabel('$log(S(f))$')
    
    plt.tight_layout()
    return 

def nice_plot(xdata, ydata, xlabel:str, ylabel:str, xmin=None, xmax=None, log=True):
    """Create a nice plot.

    Args:
        xdata (array-like): X-axis data.
        ydata (array-like): Y-axis data.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        xmin (float, optional): Minimum value for the x-axis. Defaults to None.
        xmax (float, optional): Maximum value for the x-axis. Defaults to None.
        log (bool, optional): Whether to use a logarithmic scale. Defaults to True.

    Returns:
        None
    """
    fig = plt.figure()
    
    if log == True:
        #plt.xscale('log')
        #plt.yscale('log')
        plt.plot(np.log(xdata), np.log(ydata), color='black')
        plt.xlabel(f'log({xlabel})', fontsize='14')
        plt.ylabel(f'log({ylabel})', fontsize='14')
    else:
        plt.plot(xdata, ydata, color='black')
        plt.xlabel(xlabel, fontsize='14')
        plt.ylabel(ylabel, fontsize='14')
    plt.xlim(xmin, xmax)
    plt.tick_params(axis='both', which='major', labelsize='14')
    plt.tight_layout()


def plot_conditional_exponents(result: pd.DataFrame, parameter_key, index):  ### parameter_key is either simulation_parameters[parameter] (for conditional exponents) or 'S_of_f' (for power spectrum)
    """Plot conditional expectation values with fits.

    Args:
        result (pd.DataFrame): DataFrame containing simulation and fit results.
        parameter_key (dict): Dictionary containing parameter keys for labels.
        index (int): Index for selecting specific data.

    Returns:
        None
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8, 6), sharex=True)

     
    ax1.errorbar(result["x"], result["data"], yerr=result["errors"], fmt='o', color='blue', capsize=3, markersize=4, label='Simulation Data', zorder=1)
    ax1.plot(result["x"], result["model"], color="orange", linewidth=2, label="Model", zorder=2)
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.legend(fontsize='14')
    ax1.set_ylabel(parameter_key['ylabels'][index], fontsize='14')
    ax1.tick_params(axis='both', which='major', labelsize='14')
       
    error_ratio = (result["data"] - result["model"]) / result["errors"]

    
    ax2.scatter(result["x"], error_ratio, color='red')
    ax2.set_xscale('log')
    ax2.set_xlabel(parameter_key['xlabels'][index], fontsize='14')
    ax2.set_ylabel("(Data - Model) / Errors", fontsize='14')
    ax2.set_ylim(-5, 5)
    ax2.axhline(0, color='dimgray', ls='--')
    ax2.axhline(1, color='darkgray', ls='--')
    ax2.axhline(3, color='lightgray', ls='--')
    ax2.axhline(-1, color='darkgray', ls='--')
    ax2.axhline(-3, color='lightgray', ls='--')
    ax2.tick_params(axis='both', which='major', labelsize='14')
    plt.tight_layout()
    return  