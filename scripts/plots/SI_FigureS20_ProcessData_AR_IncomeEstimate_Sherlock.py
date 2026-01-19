# import packages
print('starting to import packages')
import os
import numpy as np
import pandas as pd
import itertools
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
import processing_functions_April2025 as pf
import psutil
import gc
import matplotlib.cbook as cbook
print('packages imported')


#%% aggregate data for households in long form df
def aggregate_hh_data_long_array(filepath, combinations, name_add):
    selected_cols = ['real', 'dT', 'dP', 'dCV', 'AR'] # Bill
    i = 0
    list_arrays = []
    for combo in combinations:
        print('i: {}, {}'.format(i, combo))
        if (combo[1] == 0 or combo[1] == 5 or combo[2] == 60 or combo[2] == 120 or combo[3] == 1.0 or combo[3] == 1.2):
            # load hh level data
            #income_class = 'random' + name_add  # can change this to random_NoInf for no inf scenarios (I think)
            om = ''

            # get dates from cashflow data
            df_cashflow, max_rates, df_dates = pf.get_max_rate_dates(filepath, combo, name_add)

            # get household data
            df_long = pf.load_combinations_single_dates_filter_IE(combo, om, filepath, name_add, df_dates)

            # get selected columns in numpy array
            arr = df_long[selected_cols].to_numpy()
            arr = arr.astype(np.float32)

            # concatenate to larger array
            list_arrays.append(arr)
        else:
            print('do not need this realization')
        i += 1

    arr_combined = np.vstack(list_arrays)
    return arr_combined

def print_memory_usage():
    process = psutil.Process()
    print(f"Memory Usage: {process.memory_info().rss / 1e9:.2f} GB")

print('functions defined')

#%% process data for figure
# get combinations to load
rof = 0.654
filepath = '../../../../../scratch/users/jskerker/AffordPaper1/Figure4/data_IE/'
inf_order = ['desal', 'dpr', 'mcasr', 'transfer_soquel', 'transfer_sv']
# 1. desal, 2. dpr, 3. mcasr, 4. transfer soquel, 5. transfer sv
name_add = 'IE_'
# import and aggregate all climate combinations
df_combinations = pd.read_csv(filepath + 'climate_scenarios_500_2025-12-12.csv')
combinations = df_combinations.values.tolist()
print(combinations)

# process data for figure 4- all data
filepath = '../../../../../scratch/users/jskerker/AffordPaper1/Figure4/data_IE/'
arr_data = aggregate_hh_data_long_array(filepath, combinations, name_add)
print(arr_data.shape)
print(arr_data[0:10])
np.save(filepath + 'array_data_AR_IE.npy', arr_data)
print('finished saving random sims data')

# process current conditions data for figure 4
# import and aggregate historical climate data for ROF=0.61, first inf=desal
filepath = '../../../../../scratch/users/jskerker/AffordPaper1/Figure2/'
real_All = [1270, 1956, 1987, 2770, 3449, 3515, 3574, 4211, 4373, 4937]
dT_All = [0, 1]
dP_All = [100]
dCV_All = ['1.0']
demand_All = ['Baseline']
combinations_modcool = list(itertools.product(real_All, dT_All, dP_All, dCV_All, demand_All))
name_add = 'NoInf_IE_'
print('Starting no inf data')
arr_data_current = aggregate_hh_data_long_array(filepath, combinations_modcool, name_add)
print(arr_data_current.shape)
print(arr_data_current[0:10])
np.save(filepath + 'array_data_AR_IE_NoInf.npy', arr_data_current)
print('finished saving no inf data')

print_memory_usage()

print('STATISTICS')
print('Quantiles: 0.1, 0.25, 0.5, 0.75, 0.9: {}'.format(np.quantile(arr_data[:,4], [0.1, 0.25, 0.5, 0.75, 0.9])))