# Sandpile Project, Computational Physics, winter term 23/24, Uni Bonn

### Authors: Malena Held, Samuel Germer

The Sandpile Dynamics Simulator uses cellular automata to model the behavior of sandpiles. By randomly adding sand, the simulator demonstrates self-organized criticality, characterized by power-law distributions of avalanche lifetimes and sizes. It facilitates systematic exploration of scaling exponents for different dimensions.

## Table of Contents

**-** [**Overview**](#overview)

**-** [**Installation**](#installation)

**-** [**Usage**](#usage)

**-** [**Reproducing Data from Report**](#reproducing-data-from-report)

**-** [**Outlook**](#outlook)

## Overview

The Sandpile Dynamics Simulator is structured into two main components: simulation and analysis of simulated data.

#### Simulation:

- The simulator allows for the execution of multiple simulations consecutively.
- Custom simulation parameters such as dimensionality and boundary conditions can be specified.

#### Analysis:

- The analysis component enables the determination of critical exponents through fitting of simulated data.
- Statistical uncertainties of both simulated data and exponents are determined using bootstrapping.
- Multiple analyses with varying parameters can be conducted on the same dataset.
- The tool automatically generates plots to visualize the results of the analysis

## Installation

The Sandpile Dynamics Simulator is written in Python and requires the following packages:

- `numpy` for storing and manipulating the lattice
- `iminuit` for fitting
- `pandas` for data analysis and safe of simulated data to .csv tables
- `uncertainties` for handling values with assigned statistical uncertatinties
- `tqdm` for progressbar of code
- `matplotlib` for plotting

To install the required packages, you can use `pip`:

```bash
pip install numpy iminuit pandas uncertainties tqdm matplotlib
```

Once the dependencies are installed, you can clone the repository and start using the simulator.

## Usage

### How to execute code

To use the Sandpile Dynamics Simulator, follow these steps:

1. **Running Simulations**: Use the following command to run simulations:

   ```bash
   python run_sandpile.py {foldername_where_data_gets_stored} {path_of_file_with_simulation_parameters}
   ```

   Replace `{foldername_where_data_gets_stored}` with the name of the folder where the simulated data will be stored, and `{path_of_file_with_simulation_parameters}` with the path to the file containing simulation parameters.

2. **Running Analysis**: After simulating the data, use the following command to perform analysis:

   ```bash
   python run_analysis.py {foldername_where_simulated_data_is_stored} {path_of_file_with_analysis_parameters}
   ```

Replace `{foldername_where_simulated_data_is_stored}` with the name of the folder where the simulated data is stored, and `{path_of_file_with_analysis_parameters}` with the path to the file containing analysis parameters.

### Form of parameter files

#### Simulation

You can specify the simulation parameters in a file, where each setting represents a simulation. You can include as many simulations in the file as you want. Here is an example of such a file for two specific simulations.

```yaml
- setting01
  - name: 2d-non_conservative-N40-abs_True_z4-closed
  - dimension: 2
  - size of grid: 40
  - critical value of z: 4
  - perturbation mechanism: non_conservative
  - boundary condition: closed
  - use absolute value: True
  - number of activated avalanches: 50000
  - maximum time steps: 20000000
  - track avalanches after steady state: False
  - steady state: 10000
  - save file for power spectrum calculation: True
  - save file for exponent calculation: True
  - save mean value of grid: True

- setting02
  - name: 3d-non_conservative-N20-abs_True_z4-closed
  - dimension: 3
  - size of grid: 20
  - critical value of z: 4
  - perturbation mechanism: non_conservative
  - boundary condition: closed
  - use absolute value: True
  - number of activated avalanches: 50000
  - maximum time steps: 20000000
  - track avalanches after steady state: False
  - steady state: 18000
  - save file for power spectrum calculation: True
  - save file for exponent calculation: True
  - save mean value of grid: True
```

##### Simulation Parameters Explanation

- `name`: name of simulation
- `dimension`: The dimensionality of the grid, >=2
- `size of grid`: The size of the grid, >=3
- `crititcal value of z`: The critical value z_crit of z when an avalanche is triggered
- `pertubation mechanism`: The perturbation mechanism used in the simulation. Either `conservative` or `non_conservative`
- `boundary condition`: The boundary condition applied in the simulation. Either `open` or `closed`
- `use absolute value`: Whether to use the condtion `|z(r)|>z_crit` or `z(r)>z_crit`
- `number of activated avalanches`: The number of activated avalanches in the simulation. Simulation stops if this number is reached.
- `maximum time steps`: The maximum number of time steps in the simulation. The script stops if this number is reached before the `number of activated avalanches`
- `track avalanches after steady state`: Feature is not yet available. Idea ist to automatically detect point of steady state
- `steady state`: The time step at which steady state is reached. The avalanches get recorded after this time step.
- `save file for power spectrum calculation`: Whether to save a file for power spectrum calculation. (Not used in our report)
- `save file for exponent calculation`: Whether to save a file for exponent calculation.
- `save mean value of grid`: Whether to save the mean value of the grid as a function of time

#### Analysis

You can include multiple analyses, even for the same data, in the file. An example file looks like this:

```yaml
- setting01
    - name: 2d-non_conservative-N40-abs_True_z4-closed
    - name for save: ana1_binw1
    - fit functions: [P_of_S, P_of_T, P_of_L, E_of_S_T, E_of_T_S, E_of_S_L, E_of_L_S, E_of_T_L, E_of_L_T, S_of_f, gamma1_gamma3_1, gamma1_gamma3_2]
    - start bins: [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
    - end bins: [200.5, 80.5, 20.5, 100.5, 300.5, 30.5, 300.5, 20.5, 100.5, 10.5]
    - bin width: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    - save plots: True
    - power spectrum R: 10
    - power spectrum T: 100
    - power spectrum N: 1000
    - bootstrap size: 200
    - xlabels: [s, $\tau$, l, $\tau$, s, l, s, l, $\tau$, f]
    - ylabels: [P(S), P($\tau$), P(L), E(S|T), E(T|S), E(S|L), E(L|S), E(T|L), E(L|T), S(f)]
    - block size: 0

- setting02
    - name: 3d-non_conservative-N20-abs_True_z4-closed
    - name for save: ana1_binw1
    - fit functions: [P_of_S, P_of_T, P_of_L, E_of_S_T, E_of_T_S, E_of_S_L, E_of_L_S, E_of_T_L, E_of_L_T, S_of_f, gamma1_gamma3_1, gamma1_gamma3_2]
    - start bins: [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
    - end bins: [200.5, 80.5, 20.5, 100.5, 300.5, 30.5, 300.5, 20.5, 100.5, 10.5]
    - bin width: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    - save plots: True
    - power spectrum R: 10
    - power spectrum T: 100
    - power spectrum N: 1000
    - bootstrap size: 200
    - xlabels: [s, $\tau$, l, $\tau$, s, l, s, l, $\tau$, f]
    - ylabels: [P(S), P($\tau$), P(L), E(S|T), E(T|S), E(S|L), E(L|S), E(T|L), E(L|T), S(f)]
    - block size: 0
```

##### Analysis Parameters Explanation

- `name`: name of simulation, this needs to be the same name that was specified in the simulation parameters
- `name for save`: name under which the analysis files (plots and fit parameters) get stored
- `fit functions`: fit functions that should be exectuded on the data. `E_X_of_Y` are the conditional expectation values, `gamma1_gamma3_1/2` are the products of the two exponents, this is neceressary because here, the fits need to run with the same bootstrap samples to include the correlation between the two
- `start bins`: left bin edge of fit range, can be specified for each fit function, `gamma1_gamma3_1/2` automatically use the fit ranges from the corresponding fit functions of the two exponents.
- `end bins`: right bin edge of fit range, can be specified for each fit function, `gamma1_gamma3_1/2` automatically use the fit ranges from the corresponding fit functions of the two exponents.
- `bin width`: bin width for fit, standard is 1 (no coarse graining), can be specified for each fit function, `gamma1_gamma3_1/2` automatically use the fit ranges from the corresponding fit functions of the two exponents.
- `save plots`: save plots with `True` or not with `False`
- `power spectrum R`: parameters for generation of power spectrum, not included in our report
- `power spectrum T`: parameters for generation of power spectrum, not included in our report
- `power spectrum N`: parameters for generation of power spectrum, not included in our report
- `bootstrap size`: number of bootstrap samples that get used for statistical error estimation of fit and data
- `xlabels`: x labels of plots
- `ylabels`: y labels of plots
- `block size`: block size for bootstrapping with blocking. Standard is zero, no blocking is used. This applys for all our data because of no autocorrelation

### Files

The following files can be generated

**simulation data:**

- `data_for_exponent_calculation.csv` stores lifetime, total dissipation, spatial linear size and timestep of all record avalanches
- `data_for_power_spectrum_calculation.txt.csv` stores dissipation rate of all record avalanches, not used in our report
- `data_mean.csv` stores `<z>(t)`

**Analysis files:**

- `power_spectrum.csv`: powerspectrum, generated from specified power spectrum parameters, not used in our report
- `results.csv`: fit results and exponents with errors
- `results_products.csv`: Fit results for product of exponents
- `plots` of `<z>(t)`, `N(s)`, `N(t)`, `N(l)` and conditional expectation values with fits

### Folder structure

The folder structure of the saved simulation and analyis data looks the following:

```yaml
{foldername_where_data_gets_stored}/
│
├── simulation1/
│   ├──simulation_data/
│       ├──data_for_exponent_calculation.csv
│       ├──data_for_power_spectrum_calculation.txt
│       ├──data_mean.csv
│       ├──simulation_parameter
├── simulation2/
│   ├──simulation_data/
│       ├──data_for_exponent_calculation.csv
│       ├──data_for_power_spectrum_calculation.txt
│       ├──data_mean.csv
│       ├──simulation_parameter
...
├── plots/
│   ├──simulation1/
│       ├──analysis1
│           ├──{all plots}
│       ├──analysis2
│           ├──{all plots}
│       ├──...
│   ├──simulation2/
│       ├──analysis1
│           ├──{all plots}
│       ├──analysis2
│           ├──{all plots}
│       ├──...
│   ├──...
├── results/
│   ├──simulation1/
│       ├──analysis1
│           ├──power_spectrum.csv
│           ├──results.csv
│           ├──results_products.csv
│       ├──analysis2
│           ├──power_spectrum.csv
│           ├──results.csv
│           ├──results_products.csv
│       ├──...
│   ├──simulation2/
│       ├──analysis1
│           ├──power_spectrum.csv
│           ├──results.csv
│           ├──results_products.csv
│       ├──analysis2
│           ├──power_spectrum.csv
│           ├──results.csv
│           ├──results_products.csv
│       ├──...
│   ├──...
```

## Reproducing Data from Report

The data that was used for the report can be found in the file `final_data`. The data can also be reproduced by running first:

```bash
python run_sandpile.py final_data_reproduced simulation_parameters.txt
```

After the simulation finished, run:

```bash
python run_analysis.py final_data_reproduced analysis_parameters.txt
```

After the analysis finished, run the analysis to get uper and lower bounds for exponents (systematic uncertainties):

```bash
python run_analysis.py final_data_reproduced analysis_parameters_systematics.txt
```
## Outlook
Future plans could include analyzing the power spectrum data that can allready generated by the code and compare it with and without avalanche interference.
