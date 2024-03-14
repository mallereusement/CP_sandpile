import pandas as pd
import matplotlib.pyplot as plt


def plot_conditional_exponents(result: pd.DataFrame, parameter_key):  ### parameter_key is either simulation_parameters[parameter] (for conditional exponents) or 'S_of_f' (for power spectrum)
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8, 6), sharex=True)

     
    ax1.errorbar(result["x"], result["data"], yerr=result["errors"], fmt='o', color='blue', capsize=3, markersize=4, label='Simulation Data', zorder=1)
    ax1.plot(result["x"], result["model"], color="orange", linewidth=2, label="Model", zorder=2)
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.set_ylabel(parameter_key['ylabel'])
       
    error_ratio = (result["data"] - result["model"]) / result["errors"]

    
    ax2.scatter(result["x"], error_ratio, color='red')
    ax2.set_xscale('log')
    ax2.set_xlabel(parameter_key['xlabel'])
    ax2.set_ylabel("(Data - Model) / Errors")
    ax2.set_ylim(-5, 5)

    plt.tight_layout()
    plt.savefig(f"exponent_calculation/plots/{parameter_key['fit_function']}.jpg", dpi=300)
        