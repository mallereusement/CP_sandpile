import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

type = 'non_conservative'
boundary_condition = 'closed'

results = pd.read_csv(f'results_{type}_{boundary_condition}.csv', sep=';')

file_for_plots = 'plots'

## lifetime
n_t, t_bins = np.histogram(results["lifetime"], bins=np.arange(1,1000))

plt.plot(t_bins[:-1], n_t)
plt.yscale('log')
plt.xscale('log')
plt.xlabel('$\\tau$')
plt.ylabel('N($\\tau$)')
plt.savefig(f'{file_for_plots}/plotslifetime_{type}_{boundary_condition}.png')
plt.show()

## total dissipation
n_s, s_bins = np.histogram(results["total dissipation"], bins=np.arange(1,1000))

plt.plot(s_bins[:-1], n_s)
plt.yscale('log')
plt.xscale('log')
plt.xlabel('s')
plt.ylabel('N(s)')
plt.savefig(f'{file_for_plots}/tot_dissipation_{type}_{boundary_condition}.png')
plt.show()


## linear spatial distance
n_l, l_bins = np.histogram(results["spatial linear size"], bins=np.arange(1,1000))

plt.plot(l_bins[:-1], n_l)
plt.yscale('log')
plt.xscale('log')
plt.xlabel('l')
plt.ylabel('N(l)')
plt.savefig(f'{file_for_plots}/size_{type}_{boundary_condition}.png')
plt.show()


def conditional_expectation_value(variable: str, condition: str, df: pd.DataFrame) -> np.ndarray:
    """calculates the conditional expectation value E(variable|condition)

    Args:
        variable (str): _description_
        condition (str): _description_
        df (pd.DataFrame): dataframe

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


keys = ['lifetime', 'total dissipation', 'spatial linear size']
short = ['t', 's', 'l']

for i in range(3):
    for j in range(3):
        if i != j:
            xvals, E = conditional_expectation_value(keys[i], keys[j], results)

            plt.plot(xvals, E)
            plt.yscale('log')
            plt.xscale('log')
            plt.xlabel(short[j])
            plt.ylabel(f'E({short[i]}|{short[j]})')
            plt.savefig(f'{file_for_plots}/E_{short[i]}_{short[j]}_{type}_{boundary_condition}.png')
            plt.show()
