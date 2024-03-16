import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_power_spectrum(freq, power_spectrum):
    plt.plot(np.log(freq), np.log(power_spectrum), color='black')
    plt.xlabel('$log(f)$')
    plt.ylabel('$log(S(f))$')
    
    plt.tight_layout()
    return 

def nice_plot(xdata, ydata, xlabel:str, ylabel:str, xmin=None, xmax=None, log=True):
    fig = plt.figure()
    
    if log == True:
        #plt.xscale('log')
        #plt.yscale('log')
        plt.plot(np.log(xdata), np.log(ydata), color='black')
        plt.xlabel(f'log({xlabel})')
        plt.ylabel(f'log({ylabel})')
    else:
        plt.plot(xdata, ydata, color='black')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    plt.xlim(xmin, xmax)
    plt.tight_layout()


def plot_conditional_exponents(result: pd.DataFrame, parameter_key, index):  ### parameter_key is either simulation_parameters[parameter] (for conditional exponents) or 'S_of_f' (for power spectrum)
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8, 6), sharex=True)

     
    ax1.errorbar(result["x"], result["data"], yerr=result["errors"], fmt='o', color='blue', capsize=3, markersize=4, label='Simulation Data', zorder=1)
    ax1.plot(result["x"], result["model"], color="orange", linewidth=2, label="Model", zorder=2)
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.set_ylabel(parameter_key['ylabels'][index])
       
    error_ratio = (result["data"] - result["model"]) / result["errors"]

    
    ax2.scatter(result["x"], error_ratio, color='red')
    ax2.set_xscale('log')
    ax2.set_xlabel(parameter_key['xlabels'][index])
    ax2.set_ylabel("(Data - Model) / Errors")
    ax2.set_ylim(-5, 5)

    plt.tight_layout()
    return  