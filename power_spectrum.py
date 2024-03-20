import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.fft import fft, ifft



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


