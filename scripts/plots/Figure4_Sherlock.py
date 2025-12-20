# import packages
print('starting to import packages')
import os
import numpy as np
import pandas as pd
import itertools
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import processing_functions as pf
import psutil
import json
print('packages imported')

#%% function that merges household and income data and returns the merged df
def merge_hh_income_data(df_hh, climate_scenario_name):
    # import income data
    # not included in Github repo
    df_income = pd.read_csv('../../../../../scratch/users/jskerker/AffordPaper1/Figure4/data_01Dec2025/resampled_income_data_30Nov2024.csv')
    df_income = df_income.rename(columns={'account': 'Account'})

    # merge household and income data
    df_hh_merge = pd.merge(df_hh, df_income, how='left', on='Account')

    # 3. add column for low income or not and climate change or historical
    df_hh_merge['is_low_inc'] = df_hh_merge['map_inc_1'].apply(lambda x: 'Low' if x < 40000 else 'Not')
    df_hh_merge['middle_income'] = df_hh_merge['map_inc_1'].apply(lambda x: 'Middle' if 40000 < x < 150000 else 'Not')

    # 4. add climate scenario name
    df_hh_merge['climate_scen'] = climate_scenario_name

    # 5. create category based on climate scenario and low income
    df_hh_merge['category'] = df_hh_merge['climate_scen'] + ', ' + df_hh_merge['is_low_inc']
    return df_hh_merge

# aggregate data for households in long form df
def aggregate_hh_data_long_array(filepath, combinations, name_add):
    selected_cols = ['real', 'dT', 'dP', 'dCV', 'demand_scenario', 'Bill', 'AR']
    i = 0
    list_arrays = []
    for combo in combinations:
        print('i: {}, {}'.format(i, combo))
        if (combo[1] == 0 or combo[1] == 5 or combo[2] == 60 or combo[2] == 120 or combo[3] == 1.0 or combo[3] == 1.2):
            # load hh level data
            om = ''

            # get dates from cashflow data
            df_cashflow, max_rates, df_dates = pf.get_max_rate_dates(filepath, combo, name_add)

            # get household data
            df_long = pf.load_combinations_single_dates_filter(combo, om, filepath, name_add, df_dates)

            # get selected columns in numpy array
            arr = df_long[selected_cols].to_numpy()

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
filepath = '../../../../../scratch/users/jskerker/AffordPaper1/Figure4/data_01Dec2025/'
inf_order = ['desal', 'dpr', 'mcasr', 'transfer_soquel', 'transfer_sv']
# 1. desal, 2. dpr, 3. mcasr, 4. transfer soquel, 5. transfer sv
name_add = ''
# import and aggregate all climate combinations
df_combinations = pd.read_csv(filepath + 'climate_scenarios_500_2025-12-05.csv')
combinations = df_combinations.values.tolist()
print(combinations)

# process data for figure 4- all data
arr_data = aggregate_hh_data_long_array(filepath, combinations, name_add)
np.save(filepath + 'array_data_HH_data.npy', arr_data)
print('finished saving random sims data')
del arr_data

# process current conditions data for figure 4
# import and aggregate historical climate data for ROF=0.61, first inf=desal
filepath_NoInf = '../../../../../scratch/users/jskerker/AffordPaper1/Figure2/'
real_All = [1270, 1956, 1987, 2770, 3449, 3515, 3574, 4211, 4373, 4937]
dT_All = [0, 1]
dP_All = [100]
dCV_All = ['1.0']
demand_All = ['Baseline']
combinations_modcool = list(itertools.product(real_All, dT_All, dP_All, dCV_All, demand_All))
name_add = 'NoInf_'
print('Starting no inf data')
arr_data_current = aggregate_hh_data_long_array(filepath_NoInf, combinations_modcool, name_add)
np.save(filepath + 'array_data_HH_data_current_NoInf.npy', arr_data_current)
print('finished saving no inf data')
del arr_data_current

print_memory_usage()
#%% import data for figure
# import arr_data
arr_data_current = np.load(os.path.join(filepath, 'array_data_HH_data_current_NoInf.npy'), allow_pickle=True)
print('imported no inf data')
arr_data = np.load(os.path.join(filepath, 'array_data_HH_data.npy'), allow_pickle=True)
print('imported all hh data')

print(f"Array data current size: {arr_data_current.nbytes / 1024**2:.2f} MB")
print(f"Array data size: {arr_data.nbytes / 1024**2:.2f} MB")

# create a version of the bar plots (or boxplots) looking at affordability outcomes
col = 'AR'
coln = 6
data_values = arr_data[:, coln]
list_arrs = []
scenario_list = []
scenario_list_labels = ['Current', 'x0.6', 'x1.2', 'x1.0', 'x1.2', '+0.0', '+5.0']

# current conditions
list_arrs.append(arr_data_current[:, coln])
scenario_list.append('Current')
#del arr_data_current

# precipitation scenarios
selected_precip = [60, 120]
for dP_group in selected_precip:
    mask = arr_data[:, 2] == dP_group  # boolean mask for demand type
    dP_vals = data_values[mask]  # Select values for the group
    list_arrs.append(dP_vals)  # Append as a NumPy array
    scenario_list.append(dP_group)  # Store the demand type

# CV scenarios
selected_CV = [1.0, 1.2]
for dCV_group in selected_CV:
    mask = arr_data[:, 3] == dCV_group  # boolean mask for demand type
    dCV_vals = data_values[mask]  # Select values for the group
    list_arrs.append(dCV_vals)  # Append as a NumPy array
    scenario_list.append(dCV_group)  # Store the demand type

# Temperature scenarios
selected_T = [0, 5]
for dT_group in selected_T:
    mask = arr_data[:, 1] == dT_group  # boolean mask for demand type
    dT_vals = data_values[mask]  # Select values for the group
    list_arrs.append(dT_vals)  # Append as a NumPy array
    scenario_list.append(dT_group)  # Store the demand type

#del arr_data
print_memory_usage()
#%% create AR figure
# update matplotlib params
# Load rcParams
with open("rcparams.json", "r") as f:
    loaded_rcparams = json.load(f)
# Remove the backend setting if it exists
loaded_rcparams.pop("backend", None)
mpl.use("Agg")  # Use a non-interactive backend suitable for Slurm
# Convert back if necessary (e.g., Cycler objects won't be restored properly)
mpl.rcParams.update(loaded_rcparams)

# create a version of the bar plots (or boxplots) looking at affordability outcomes
fig = plt.figure(figsize=(10, 4))
x_axis = np.array([-1.1, 0, 1, 2.5, 3.5, 5, 6])  # , 7.5, 8.5
# boxplot
box = plt.boxplot(list_arrs, positions=x_axis, widths=0.6, patch_artist=True, showfliers=False)

# Customize colors
colors = ['silver', 'teal', 'teal', 'cadetblue', 'cadetblue', 'mediumturquoise', 'mediumturquoise', 'paleturquoise',
          'paleturquoise']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

# add plot labels/limits
plt.plot([-1.75, 9.25], [2.5, 2.5], color='k', linewidth=1.5, linestyle='--')
plt.xticks(x_axis, scenario_list_labels, fontsize=12)
plt.xlabel('Climate Uncertainty', fontsize=12)
plt.ylabel('AR (% of bill / income)', fontsize=12)
plt.xlim(-1.75, 6.75)
y_max = 10
plt.ylim(-0.5, y_max)
plt.yticks(np.arange(0, y_max + 1, 2), fontsize=12)

# add lines
lw = 1
line_color = 'k'
plt.plot([-0.55, -0.55], [-1, y_max], color=line_color, linewidth=lw, linestyle=':')
plt.plot([1.75, 1.75], [-1, y_max], color=line_color, linewidth=lw, linestyle=':')
plt.plot([4.25, 4.25], [-1, y_max], color=line_color, linewidth=lw, linestyle=':')

# add labels
diff = 0.75

# save figure
plt.savefig('../outputs/Figure4_Boxplots_climate_sensitivities_AR.png', dpi=300, bbox_inches='tight')

#%% extract AR data for text
# Extract data for ARs
medians = [median.get_ydata()[0] for median in box['medians']]
whiskers = [whisker.get_ydata() for whisker in box['whiskers']]
fliers = [flier.get_ydata() for flier in box['fliers']]

# Print extracted data
print("Medians:", medians)
print("Whiskers:", whiskers)
print("Outliers:", fliers)

#%% create boxplots for water bills for SI
# WATER BILL VERSION FOR SI
# create a version of the bar plots (or boxplots) looking at affordability outcomes
col = 'Bill'
coln = 5
data_values = arr_data[:, coln]
list_arrs = []
scenario_list = []

# current conditions
list_arrs.append(arr_data_current[:, coln])
scenario_list.append('Current')

# precipitation scenarios
selected_precip = [60, 120]
for dP_group in selected_precip:
    mask = arr_data[:, 2] == dP_group  # boolean mask for demand type
    dP_vals = data_values[mask]  # Select values for the group
    list_arrs.append(dP_vals)  # Append as a NumPy array
    scenario_list.append(dP_group)  # Store the demand type

# CV scenarios
selected_CV = [1.0, 1.2]
for dCV_group in selected_CV:
    mask = arr_data[:, 3] == dCV_group  # boolean mask for demand type
    dCV_vals = data_values[mask]  # Select values for the group
    list_arrs.append(dCV_vals)  # Append as a NumPy array
    scenario_list.append(dCV_group)  # Store the demand type

# Temperature scenarios
selected_T = [0, 5]
for dT_group in selected_T:
    mask = arr_data[:, 1] == dT_group  # boolean mask for demand type
    dT_vals = data_values[mask]  # Select values for the group
    list_arrs.append(dT_vals)  # Append as a NumPy array
    scenario_list.append(dT_group)  # Store the demand type
print_memory_usage()

#%%
fig = plt.figure(figsize=(10, 3.5))
x_axis = np.array([-1.1, 0, 1, 2.5, 3.5, 5, 6])
# boxplot
box = plt.boxplot(list_arrs, positions=x_axis, widths=0.6, patch_artist=True, showfliers=False)

# Customize colors
colors = ['silver', 'teal', 'teal', 'cadetblue', 'cadetblue', 'mediumturquoise', 'mediumturquoise', 'paleturquoise',
          'paleturquoise']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

# add plot labels/limits
plt.xticks(x_axis, scenario_list_labels, fontsize=12)
plt.xlabel('Climate Uncertainty', fontsize=12)
plt.ylabel('Bill ($/month)', fontsize=12)
plt.xlim(-1.75, 6.75)
y_max = 450
plt.ylim(-0.5, y_max)
plt.yticks(np.arange(0, y_max + 1, 50), labels=['0', '', '100', '', '200', '', '300', '', '400', ''], fontsize=12)

# add lines
lw = 1
line_color = 'k'
plt.plot([-0.55, -0.55], [-1, y_max], color=line_color, linewidth=lw, linestyle=':')
plt.plot([1.75, 1.75], [-1, y_max], color=line_color, linewidth=lw, linestyle=':')
plt.plot([4.25, 4.25], [-1, y_max], color=line_color, linewidth=lw, linestyle=':')

# add labels
diff = 30

# save figure
plt.savefig('../outputs/Figure4_SI_Boxplots_climate_sensitivities_Bill.png', dpi=300, bbox_inches='tight')

#%% print out values
for i in range(len(list_arrs)):
    print(i)
    arr = list_arrs[i]
    print('median value: {}, 80th percentile: {}'.format(np.median(arr), np.quantile(arr, 0.8)))