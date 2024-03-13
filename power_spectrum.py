import numpy as np
import pandas as pd
import static_definitions
from iminuit.cost import LeastSquares
from iminuit import Minuit
import uncertainties as unc
import matplotlib.pyplot as plt
from uncertainties import unumpy as unp
from tqdm import tqdm
from scipy.fft import fft, ifft

type = 'non_conservative'
boundary_condition = 'closed'
d = 2
d_list = [2]

def load_data(filename: str):
    file  = open(f'{filename}', 'r')
    lines = file.readlines()
    l = []
    len_avalanche = []
    for i, line in enumerate(lines):
        l.append(list(map(int,map(float, line.strip().strip('][').split(', ') ))))
        
        len_avalanche.append(len(l[i]))
    return l, len_avalanche



def calculate_j(l: list, R: int, T:int, R_start:int, max_length):
    j_tau = np.zeros(T + max_length)

    for i in range(R):
        tau_0 = np.random.randint(0,T)
            
            #data = df[str(i + R_start)].to_numpy()
        data = np.array(l[int(i+R_start)])
            
        dissipation = np.zeros(T + max_length)
        dissipation[:len(data)] = data
        dissipation = np.roll(dissipation, tau_0)
            
        j_tau += dissipation
    return j_tau


def calculate_power_spectrum(max_length, R, T, N, l):
    abs_fft_j = np.zeros(T + max_length)
    for n in range(N):
        j_tau = calculate_j(l, R, T, n*R, max_length)
        abs_fft_j += abs(fft(j_tau))**2

            
    abs_fft_j = abs_fft_j / N
    freq = 2*np.pi*np.arange(len(abs_fft_j)) / T
    return abs_fft_j, freq




if __name__ == '__main__':       

    
    for d in d_list:


        file  = open(f'simulation_avalanche_dissipation/dissipation_sim_{type}_{boundary_condition}_{d}.txt', 'r')
        lines = file.readlines()
        l = []
        len_avalanche = []
        for i, line in enumerate(lines):
            l.append(list(map(int,map(float, line.strip().strip('][').split(', ') ))))
            
            len_avalanche.append(len(l[i]))
            
        print(l[:100])

        R = 10
        T = 1000
        N = 100
        max_length = np.max(np.array(len_avalanche))

        if len(l) < N*R:
            print('ERROR: too few avalanches in data')
            exit()




        #def calculate_power_spectrum()

        abs_fft_j = np.zeros(T + max_length)
        pbar = tqdm(total = N, desc ="Running Simulation")
        for n in range(N):
            j_tau = calculate_j(l, R, T, n*R, max_length)
            abs_fft_j += abs(fft(j_tau))**2
            pbar.update(1)
            
        abs_fft_j = abs_fft_j / N
        pbar.close()
            

        freq = 2*np.pi*np.arange(len(abs_fft_j)) / T
        plt.plot(np.log(freq), np.log(abs_fft_j))
    plt.show()