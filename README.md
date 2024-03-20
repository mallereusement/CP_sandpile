# Sandpile Project, Computational Physics, winter term 23/24, Uni Bonn

### Authors: Malena Held, Samuel Germer

The Sandpile Dynamics Simulator uses cellular automata to model the behavior of sandpiles. By randomly adding sand, the simulator demonstrates self-organized criticality, characterized by power-law distributions of avalanche lifetimes and sizes. It facilitates systematic exploration of scaling exponents for different dimensions.

## Table of Contents

**-** [**Overview**](#overview)

**-** [**Installation**](#installation)

**-** [**Usage**](#usage)

**-** [**Examples**](#examples)

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
- `uncertainties` for handling values with asigned statistical uncertatinties
- `tqdm` for progressbar of code
- `matplotlib` for plotting

To install the required packages, you can use `pip`:

```bash
pip install numpy iminuit pandas uncertainties tqdm matplotlib
```

Once the dependencies are installed, you can clone the repository and start using the simulator.

## Usage

## Example usage commands
