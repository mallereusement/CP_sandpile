import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm



df = pd.read_csv('data_for_exponent_calculation.csv', sep=';', encoding='utf8')
min_timestep = df['timestep'].min()
max_timestep = df['timestep'].max()
all_timesteps = pd.DataFrame({'timestep': range(min_timestep, max_timestep + 1)})

# Merge the original DataFrame with the DataFrame containing all possible timesteps
df_all_timesteps = all_timesteps.merge(df, on='timestep', how='left')

# Fill missing values with zeros
df_all_timesteps['lifetime'] = df_all_timesteps['lifetime'].fillna(0)
df = df_all_timesteps
df = df.head(1000)


data = df['lifetime'].to_numpy()

# Adding plot title.
plt.title("Autocorrelation Plot") 
 
# Providing x-axis name.
plt.xlabel("Lags") 
 
# Plotting the Autocorrelation plot.
plt.acorr(data, maxlags = 10) 
plt.grid(True)
 
plt.show() 