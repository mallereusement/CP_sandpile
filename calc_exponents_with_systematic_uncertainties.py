import numpy as np
import pandas as pd
import os


def read_dataframes(path, t):

    dfs = []

    # Iterate over each folder in the directory
    for folder_name in os.listdir(path):
        # Check if the folder starts with "results_"
        if folder_name.startswith("results_"):
            if t == 1:
                csv_file_path = f'{path}/{folder_name}/results.csv'
            if t == 2:
                csv_file_path = f'{path}/{folder_name}/results_products.csv'
            df = pd.read_csv(csv_file_path, sep=';', encoding='utf8')
            dfs.append(df)
    return dfs

def get_exponents(dfs, t):
    for df in dfs:
        if t == 1:
            df['exp'] = df['exponent from fit result'].apply(lambda x: x.split('+/-')[0])
            df['exp_unc'] = df['exponent from fit result'].apply(lambda x: x.split('+/-')[1])
        if t == 2:
            df['exp'] = df['product from fit result'].apply(lambda x: x.split('+/-')[0])
            df['exp_unc'] = df['product from fit result'].apply(lambda x: x.split('+/-')[1])
            
    return dfs


fit_functions1 = ['P_of_S', 'P_of_T', 'P_of_L', 'E_of_S_T', 'E_of_T_S', 'E_of_S_L', 'E_of_L_S', 'E_of_T_L', 'E_of_L_T']
fit_functions2 = ['gamma1_gamma3_1', 'gamma1_gamma3_2']

if __name__ == '__main__':

    path = '20240319_final_results/2d-non_conservative-N100-abs_True_z4-closed'

    dfs = read_dataframes(path, 1)
    dfs = get_exponents(dfs, 1)

    df_for_save = {'fit function': [], 'exponent': [], 'uncertainty': []}

    for f in fit_functions1:
        try:
            exp_mean = []
            exp_unc = []
            unc_stat = []
            exps = []
            for df in dfs:
                try:
                    exp = float(df[df['fit function'] == f]['exp'].tolist()[0])
                    unc = float(df[df['fit function'] == f]['exp_unc'].tolist()[0])
                    unc_stat.append(unc)
                    exps.append(exp)
                except:
                    pass


            exps = np.array(exps)


            exp_mean = np.mean(exps)
            unc_stat = np.mean(np.array(unc_stat))
            exp_unc = unc_stat + np.sum([(exps[i] - exp_mean)**2 / len(exps) for i in range(len(exps))])
            df_for_save['fit function'].append(f)
            df_for_save['exponent'].append(exp_mean)
            df_for_save['uncertainty'].append(exp_unc)
            print(f'exp of {f}: {exp_mean} +/- {exp_unc}')
        except:
            pass
    dfs = read_dataframes(path, 2)
    dfs = get_exponents(dfs, 2)

    for f in fit_functions2:
        try:
            exp_mean = []
            exp_unc = []
            unc_stat = []
            exps = []
            for df in dfs:
                exp = float(df[df['fit function'] == f]['exp'].tolist()[0])
                unc = float(df[df['fit function'] == f]['exp_unc'].tolist()[0])
                unc_stat.append(unc)
                exps.append(exp)
            exps = np.array(exps)

            exp_mean = np.mean(exps)
            unc_stat = np.mean(np.array(unc_stat)**2)
            exp_unc = unc_stat + np.sum([(exps[i] - exp_mean)**2 for i in range(len(exps))]) / len(exps)
            exp_unc = np.sqrt(exp_unc)
            df_for_save['fit function'].append(f)
            df_for_save['exponent'].append(exp_mean)
            df_for_save['uncertainty'].append(exp_unc)
            print(f'exp of {f}: {exp_mean} +/- {exp_unc}')
        except:
            pass       

    df_for_save = pd.DataFrame(df_for_save)
    df_for_save.to_csv(f'{path}/exponents_averaged_over_all_fits.csv', sep=';', encoding='utf8')
