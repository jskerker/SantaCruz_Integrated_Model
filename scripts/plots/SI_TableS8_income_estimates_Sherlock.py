# import functions
import numpy as np
import pandas as pd
import itertools
import processing_functions_April2025 as pf

# loop through income estimates and get AR50 and AR80
filepath = '../../../../../scratch/users/jskerker/AffordPaper1/SA/Climate/'
filepath_save = '../outputs/'

# income estimates
income_estimates = np.arange(1, 11)
print(income_estimates)

# climate scenarios- modcool
real_All = [1270, 1956, 1987, 2770, 3449, 3515, 3574, 4211, 4373, 4937]
dT_All = [0, 1]
dP_All = [100]
dCV_All = ['1.0']
demand_All = ['Baseline']
combinations_modcool = list(itertools.product(real_All, dT_All, dP_All, dCV_All, demand_All))

# climate scenarios- dry, hot
dT_All = [4, 5]
dP_All = [80, 90]
dCV_All = ['1.2']
combinations_dryhot = list(itertools.product(real_All, dT_All, dP_All, dCV_All, demand_All))

# other params
name_add = ''
rof = 0.65
inf_order = ['dpr', 'desal', 'transfer_soquel', 'mcasr', 'transfer_sv']
#sample_size = 100000
df_long = pd.DataFrame()
list_statistics = []

# loop through income estimates- modcool
for income in income_estimates:
    mapped_income = 'map_inc_{}'.format(income)
    print(mapped_income)

    # loop through scenarios
    df_long = pf.aggregate_sows_for_hh_data_long_IE_SI_All(filepath, rof, inf_order, combinations_modcool, name_add, mapped_income)

    # get statistics
    AR_50 = df_long['AR'].median()
    AR_80 = df_long['AR'].quantile(0.8)
    del df_long
    list_statistics.append([mapped_income, AR_50, AR_80])

df_stats_modcool = pd.DataFrame(list_statistics, columns=['mapped_income', 'AR_50', 'AR_80'])
df_stats_modcool.to_csv(filepath_save + 'df_stats_modcool.csv', index=False)

# loop through income estimates- dryhot
list_statistics = []
for income in income_estimates:
    mapped_income = 'map_inc_{}'.format(income)
    print(mapped_income)

    # loop through scenarios
    df_long = pf.aggregate_sows_for_hh_data_long_IE_SI_All(filepath, rof, inf_order, combinations_dryhot, name_add, mapped_income)

    # get statistics
    AR_50 = df_long['AR'].median()
    AR_80 = df_long['AR'].quantile(0.8)
    del df_long
    list_statistics.append([mapped_income, AR_50, AR_80])

df_stats_dryhot = pd.DataFrame(list_statistics, columns=['mapped_income', 'AR_50', 'AR_80'])
df_stats_dryhot.to_csv(filepath_save + 'df_stats_dryhot.csv', index=False)