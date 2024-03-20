import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics import tsaplots


path_to_file = 'data_for_exponent_calculation.csv'

df = pd.read_csv(path_to_file, sep=';', encoding='utf8')
data = df['lifetime'].to_numpy(dtype=float)

fig = tsaplots.plot_acf(data, lags=10000)
plt.xlim(-1, 10)
plt.ylim(-0.1, 1.1)
plt.ylabel('Normalized ACF')
plt.xlabel('time')
plt.title('')
plt.savefig('acf.png', dpi=300)

