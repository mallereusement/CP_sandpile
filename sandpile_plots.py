import matplotlib.pyplot as plt
import pandas as pd
import numpy as np




def conditional_expectation_value(variable: str, condition: str, df: pd.DataFrame) -> np.ndarray:
    """calculates the conditional expectation value E(variable|condition)

    Args:
        variable (str): variable
        condition (str): condition
        df (pd.DataFrame): simulation data

    Returns:
        np.ndarray: array with the conditional expectation value
    """
    
    vmin = min(df[condition])
    vmax = max(df[condition])
    
    expectation_list = []
    xvals = []
    
    for value in range(int(vmin), int(vmax)):
        list = df[df[condition] == value][variable].to_numpy()
        
        expectation = np.sum(list) / len(list)
        expectation_list.append(expectation)
        xvals.append(value)
    return xvals, expectation_list


def plot_results(results_df: pd.DataFrame, variable: str, bins: np.ndarray, type_: str, boundary_condition_: str, x_label: str, y_label: str, file_for_plot: str, log_scale: bool=True) -> None:
    """ Plot results of given variable

    Args:
        results_df (pd.DataFrame): simulation data
        variable (str): variable to plot
        bins (np.ndarray): bins for histogram
        type_ (str): pertubation_mechanism
        boundary_condition_ (str): boundary condition
        x_label (str): x label of plot
        y_label (str): y label of plot
        file_for_plot (str): file where plot gets saved
        log_scale (bool, optional): Enable log scale for x and y. Defaults to True.
    """
    n, bins = np.histogram(results_df[variable], bins=bins)

    plt.plot(bins[:-1], n)
    if log_scale:
        plt.yscale('log')
        plt.xscale('log')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(f'{file_for_plot}/{variable}_{type_}_{boundary_condition_}.png')
    plt.show()

def plot_conditional_expectation_value(variable: str, condition: str, results_df: pd.DataFrame, type_: str, boundary_condition_: str, x_label: str, y_label: str, file_for_plot: str, log_scale=True) -> None:
    """Plot conditional expectation value for given variable and condition

    Args:
        variable (str): variable
        condition (str): condition
        results_df (pd.DataFrame): simulation data
        type_ (str): pertubation mechanism
        boundary_condition_ (str): boundary condition
        x_label (str): x label of plot
        y_label (str): y label of plot
        file_for_plot (str): file where plot gets saved
        log_scale (bool, optional): Enable log scale for x and y. Defaults to True.
    """
    xvals, E = conditional_expectation_value(variable, condition, results_df)
    plt.plot(xvals, E)
    if log_scale:
        plt.yscale('log')
        plt.xscale('log')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(f'{file_for_plot}/E_{variable}_{condition}_{type_}_{boundary_condition_}.png')
    plt.show()








if __name__ == '__main__':

    type = 'non_conservative'
    boundary_condition = 'closed'
    file_for_plots = 'plots'
    results = pd.read_csv(f'results_{type}_{boundary_condition}.csv', sep=';')
    
    ## lifetime
    plot_results(results, 'lifetime', np.arange(1,1000), type, boundary_condition, '$\\tau$', 'N($\\tau$)', file_for_plots)

    ## total dissipation
    plot_results(results, 'total dissipation', np.arange(1,1000), type, boundary_condition, 's', 'N(s)', file_for_plots)

    ## linear spatial distance
    plot_results(results, 'spatial linear size', np.arange(1,1000), type, boundary_condition, 'l', 'N(l)', file_for_plots)

    ## plot all conditional expectation values
    keys = ['lifetime', 'total dissipation', 'spatial linear size']
    short = ['t', 's', 'l']

    for i in range(3):
        for j in range(3):
            if i != j:

                plot_conditional_expectation_value(keys[i], keys[j], results, type, boundary_condition, short[j], f'E({short[i]}|{short[j]})', file_for_plots)