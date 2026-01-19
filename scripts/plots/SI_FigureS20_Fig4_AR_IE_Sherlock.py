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
import processing_functions_March2025 as pf
import psutil
import gc
import matplotlib.cbook as cbook
print('packages imported')

def print_memory_usage():
    process = psutil.Process()
    print(f"Memory Usage: {process.memory_info().rss / 1e9:.2f} GB")

print('functions defined')
print_memory_usage()
#%% import data for figure
# import arr_data
filepath = '../../../../../scratch/users/jskerker/AffordPaper1/Figure2/'
arr_data_current = np.load(os.path.join(filepath, 'array_data_AR_IE_NoInf.npy'), allow_pickle=True)
arr_data_current = arr_data_current.astype(np.float32)
n_rows = arr_data_current.shape[0]
print('rows of data (no inf): {}'.format(n_rows))
print('imported no inf data')

filepath = '../../../../../scratch/users/jskerker/AffordPaper1/Figure4/data_IE/'
arr_data = np.load(os.path.join(filepath, 'array_data_AR_IE.npy'), allow_pickle=True)
arr_data = arr_data.astype(np.float32)
n_rows = arr_data.shape[0]
print('rows of data (500 sims): {}'.format(n_rows))
print('imported all hh data')
print_memory_usage()

# create a version of the bar plots (or boxplots) looking at affordability outcomes
coln = 4 # AR
#data_values = arr_data[:, coln]
list_arrs = []
scenario_list = []
scenario_list_labels = ['Current', 'x0.6', 'x1.2', 'x1.0', 'x1.2', '+0.0', '+5.0']

# current conditions
list_arrs.append(arr_data_current[:, coln])
scenario_list.append('Current')

# precipitation scenarios
selected_precip = [60, 120]
for dP_group in selected_precip:
    filtered = arr_data[arr_data[:, 2] == dP_group]
    list_arrs.append(filtered[:, coln]) # append as numpy array
    scenario_list.append(dP_group) # store demand type

# CV scenarios
selected_CV = [1.0, 1.2]
for dCV_group in selected_CV:
    filtered = arr_data[arr_data[:, 3] == dCV_group]
    list_arrs.append(filtered[:, coln])
    scenario_list.append(dCV_group)

# Temperature scenarios
selected_T = [0, 5]
for dT_group in selected_T:
    filtered = arr_data[arr_data[:, 1] == dT_group]
    list_arrs.append(filtered[:, coln])
    scenario_list.append(dT_group)
print_memory_usage()

del arr_data
del arr_data_current
del filtered
gc.collect()
print('deleted arr_data, data_values')
print_memory_usage()

# precompute statistics to save memory
print('length of one array: {}'.format(len(list_arrs[2])))
stats = cbook.boxplot_stats(list_arrs, labels=None)
#del list_arrs
print('precomputed statistics and deleted list_arrs')
print_memory_usage()

#%% create AR figure
# create a version of the bar plots (or boxplots) looking at affordability outcomes
fig, ax = plt.subplots(figsize=(10, 4))
x_axis = np.array([-1.1, 0, 1, 2.5, 3.5, 5, 6])  # , 7.5, 8.5
# set gray background
ax.set_facecolor('#EAEAF2') # light gray
ax.grid(True, color='white', zorder=0) # white gridlines
# Remove the spines (outline)
for spine in ax.spines.values():
    spine.set_visible(False)
# Remove tick marks
ax.tick_params(axis='both', length=0)
# boxplot
#box = plt.boxplot(list_arrs, positions=x_axis, widths=0.6, patch_artist=True, showfliers=False)
box = ax.bxp(stats, positions=x_axis, widths=0.6, patch_artist=True, showfliers=False) # , manage_ticks=False
print_memory_usage()
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
# plt.plot([6.75, 6.75], [-1, y_max], color=line_color, linewidth=lw, linestyle=':')

# add labels
diff = 0.75
# plt.text(-0.1, y_max-diff, 'Precipitation (%)', fontsize=12, fontweight='bold')
# plt.text(2.45, y_max-diff-0.5, 'Precipitation \nVariability', fontsize=12, fontweight='bold', ha='center', va='center')
# plt.text(5.3, y_max-diff, 'Temperature (°C)', fontsize=12, fontweight='bold')
# plt.text(7.0, y_max-diff, '', fontsize=12, fontweight='bold')

# save figure
plt.savefig('../outputs/Figure4_Boxplots_SI_AR_IE_14Dec.png', dpi=300, bbox_inches='tight')
#plt.show()

#%% extract AR data for text
# Extract data for ARs
medians = [median.get_ydata()[0] for median in box['medians']]
whiskers = [whisker.get_ydata() for whisker in box['whiskers']]
fliers = [flier.get_ydata() for flier in box['fliers']]

# Print extracted data
print("Medians:", medians)
print("Whiskers:", whiskers)
print("Outliers:", fliers)


#%% print out values
for i in np.arange(len(list_arrs)):
    print(i)
    arr = list_arrs[i]
    print('median value: {}, 80th percentile: {}, 90th percentile: {}'.format(np.median(arr), np.quantile(arr, 0.8), np.quantile(arr, 0.9)))