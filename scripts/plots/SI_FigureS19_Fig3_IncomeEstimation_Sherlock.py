# import packages
import numpy as np
import pandas as pd
import itertools
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.patches import Patch
import matplotlib as mpl
import psutil
import warnings
import json
from datetime import datetime
warnings.filterwarnings("ignore", category=DeprecationWarning)

print("start: ", datetime.now())
def optimize_df(df):
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

def print_memory_usage():
    process = psutil.Process()
    print(f"Memory Usage: {process.memory_info().rss / 1e9:.2f} GB")

# function that merges household and income data and returns the merged df
def merge_hh_income_data(df_hh, climate_scenario_name):
    # import income data
    df_income = pd.read_csv('../../../../../scratch/users/jskerker/AffordPaper1/Figure2/resampled_income_data_30Nov2024.csv')
    df_income = df_income.rename(columns={'account': 'Account'})

    # merge household and income data
    df_hh_merge = pd.merge(df_hh, df_income, how='left', on='Account')

    # 2b. for testing income estimation, recalculate AR using a different income estimate
    df_hh_merge['AR_v2'] = df_hh_merge['Bill'] / (df_hh_merge['map_inc_2']/12) * 100 # % of annual income

    # 3. add column for low income or not and climate change or historical
    print(df_hh_merge.columns)
    df_hh_merge['is_low_inc'] = df_hh_merge['map_inc_2'].apply(lambda x: 'Low' if x < 40000 else 'Not')
    # df_hh_merge['middle_income'] = df_hh_merge['map_inc_1'].apply(lambda x: 'Middle' if 40000 < x < 150000 else 'Not')

    # 4. add climate scenario name
    df_hh_merge['climate_scen'] = climate_scenario_name

    # 5. create category based on climate scenario and low income
    df_hh_merge['category'] = df_hh_merge['climate_scen'] + ', ' + df_hh_merge['is_low_inc']
    return df_hh_merge

print('packages imported')

# import household-level data
filepath = '../../../../../scratch/users/jskerker/AffordPaper1/Figure2/'
filepath_SA = '../../../../../scratch/users/jskerker/AffordPaper1/SA/Climate/'

# moderate climate scenarios
df_hh_hist = pd.read_parquet(filepath_SA + "df_hh_modcool_long.parquet")
df_hh_hist = optimize_df(df_hh_hist)
df_hh_hist = merge_hh_income_data(df_hh_hist, 'baseline')
print('moderate scenarios imported: ', datetime.now())

# baseline, no inf
df_hh_hist_NoInf = pd.read_parquet(filepath + "df_hh_modcool_NoInf_long.parquet")
df_hh_hist_NoInf = optimize_df(df_hh_hist_NoInf)
df_hh_hist_NoInf = merge_hh_income_data(df_hh_hist_NoInf, 'baseline/no inf.')
print('no inf scenarios imported: ', datetime.now())

# dry climate scenarios
df_hh_dryhot = pd.read_parquet(filepath_SA + "df_hh_dryhot_long.parquet")
df_hh_dryhot = optimize_df(df_hh_dryhot)
df_hh_dryhot = merge_hh_income_data(df_hh_dryhot, 'dry/hot')
df_hh = pd.concat([df_hh_hist, df_hh_hist_NoInf, df_hh_dryhot])
df_hh = df_hh.reset_index(drop=True)
print('imported household data: ', datetime.now())

# update matplotlib params
# Load rcParams
with open("rcparams.json", "r") as f:
    loaded_rcparams = json.load(f)
# Remove the backend setting if it exists
loaded_rcparams.pop("backend", None)
mpl.use("Agg")  # Use a non-interactive backend suitable for Slurm
# Convert back if necessary (e.g., Cycler objects won't be restored properly)
mpl.rcParams.update(loaded_rcparams)

# Figure 3- v2- boxplots of avg (a) water demands, (b) water bills, (c) affordability ratios


# set color palette
palette_dict = {'baseline/no inf., Low': 'salmon', 'baseline/no inf., Not': 'mediumturquoise', 'baseline, Low': 'salmon', 'baseline, Not': 'mediumturquoise', 'dry/hot, Low': 'salmon', 'dry/hot, Not': 'mediumturquoise', 'Spacer1': 'white', 'Spacer2': 'white'}
order_list = ['baseline/no inf., Low', 'baseline/no inf., Not', 'Spacer1', 'baseline, Low', 'baseline, Not', 'Spacer2', 'dry/hot, Low', 'dry/hot, Not']
#custom_x_positions = [0, 0.6, 1, 1.4, 2, 2.4, 2.8, 3.2]
custom_x_positions = [0, 0.6, 0.7, 0.8, 1.6, 1.8, 2.0, 2.6]
# set up figure
fig = plt.figure(figsize = (10, 3))
# define the grid layout
gs = gridspec.GridSpec(1, 3, width_ratios = [1, 1, 1], wspace=0.35)

# subplot 1: demand
y_max = 16
ax00 = fig.add_subplot(gs[0, 0])
sns.boxplot(data=df_hh, x='category', y='Demand', hue='category', ax=ax00, showfliers=False, width=0.8, order=order_list, palette=palette_dict, legend=False) # , hue='category', hue='is_low_inc'
ax00.set_xlabel('')
#ax00.plot([2, 2], [-0.5, y_max], color='black', linestyle='--', linewidth=1)
#ax00.plot([5, 5], [-0.5, y_max], color='black', linestyle='--', linewidth=1)
ax00.set_xlim(-0.75, 7.75)
ax00.set_ylim(-0.5, y_max)
ax00.set_yticks(np.arange(0, y_max+1, 2))
ax00.set_yticklabels(['0', '', '4', '', '8', '', '12', '', '16'], fontsize=11)
ax00.set_xticks(custom_x_positions)
ax00.set_xticklabels(['', '', '', '', '', '', '', ''])
ax00.set_ylabel('Demand (ccf/month)', fontsize=11)
ax00.set_title('Water Demands', fontsize=11, fontweight='bold')

# subplot 2: bills
y_max = 400
ax01 = fig.add_subplot(gs[0, 1])
sns.boxplot(data=df_hh, x='category', hue='category', y='Bill', ax=ax01, showfliers=False, width=0.8, order=order_list, palette=palette_dict, legend=False)
ax01.set_xlabel('')
#ax01.plot([2, 2], [-5, y_max], color='black', linestyle='--', linewidth=1.)
#ax01.plot([5, 5], [-5, y_max], color='black', linestyle='--', linewidth=1.)
ax01.set_xlim(-0.75, 7.75)
ax01.set_ylim(0, y_max)
ax01.set_yticks(np.arange(0, y_max+1, 50))
ax01.set_yticklabels(['0', '', '100', '', '200', '', '300', '', '400'], fontsize=11)
ax01.set_xticks(custom_x_positions)
ax01.set_xticklabels(['', '', '', '', '', '', '', ''])
ax01.set_ylabel('Bill ($/month)', fontsize=11)
ax01.set_title('Water Bills', fontsize=11, fontweight='bold')

# subplot 3: AR
y_max = 40
ax02 = fig.add_subplot(gs[0, 2])
box = sns.boxplot(data=df_hh, x='category', hue='category', y='AR_v2', ax=ax02, showfliers=False, width=0.8, order=order_list, palette=palette_dict, legend=False)
ax02.plot([-0.75, 7.75], [2.5, 2.5], color='navy', linestyle=':', linewidth=1.2)
ax02.set_xlabel('')
#ax02.plot([2, 2], [-1, y_max], color='black', linestyle='--', linewidth=1.)
#ax02.plot([5, 5], [-1, y_max], color='black', linestyle='--', linewidth=1.)
ax02.set_xlim(-0.75, 7.75)
ax02.set_ylim(-1, y_max)
ax02.set_yticks(np.arange(0, y_max+1, 2.5))
ax02.set_yticklabels(['0', '', '', '', '10', '', '', '', '20', '', '', '', '30', '', '', '', 40], fontsize=11)
ax02.set_xticks(custom_x_positions)
ax02.set_xticklabels(['', '', '', '', '', '', '', ''])
ax02.set_ylabel('AR (% of bill / income)', fontsize=11)
ax02.set_title('Affordability Ratios', fontsize=11, fontweight='bold')

# create custom legend- middle plot
legend_handles = [
    Patch(color='salmon', label='Low Income'),
    Patch(color='mediumturquoise', label='All Others'),
]
legend = ax00.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.0, 1.02), fontsize=9, handletextpad=0.5)
legend.set_frame_on(False)

# add text labels
level = -1.6
sz = 8.5

# Extract Whisker Locations
whisker_lines = [line.get_ydata() for line in ax02.lines if len(line.get_ydata()) == 2]
whisker_positions = [whisker[1] for whisker in whisker_lines]  # Upper and lower whisker values
print("Whisker Locations:", whisker_positions)

# add titles
# ax00.text(-0.7, level, 'Baseline', fontweight='bold', fontsize=sz, fontstyle='italic')
# ax00.text(3.5, level, ' Moderate', fontweight='bold', fontsize=sz, fontstyle='italic')
# ax00.text(6.5, level, 'Dry', fontweight='bold', fontsize=sz, fontstyle='italic')
# # plot 2
# ax00.text(10.8, level, 'Baseline', fontweight='bold', fontsize=sz, fontstyle='italic')
# ax00.text(15, level, ' Moderate', fontweight='bold', fontsize=sz, fontstyle='italic')
# ax00.text(18, level, 'Dry', fontweight='bold', fontsize=sz, fontstyle='italic')
# # plot 3
# ax00.text(22.3, level, 'Baseline', fontweight='bold', fontsize=sz, fontstyle='italic')
# ax00.text(26.5, level, ' Moderate', fontweight='bold', fontsize=sz, fontstyle='italic')
# ax00.text(29.5, level, 'Dry', fontweight='bold', fontsize=sz, fontstyle='italic')

# add a-c labels
ax02.text(-23.5,  36.75, 'a', fontweight='bold', fontsize=16)
ax02.text(-12, 36, 'b', fontweight='bold', fontsize=16)
ax02.text(-0.6, 36.75, 'c', fontweight='bold', fontsize=16)

plt.savefig('../outputs/Figure3-Boxplots-SI_map_inc_2_14Dec.png', dpi=300, bbox_inches='tight')
#plt.show()
print('plot made: ', datetime.now())
# print memory usage
print_memory_usage()  # After loading data

# get statistics for analysis
# water demands
col = 'Demand'
mean_val = df_hh.groupby('category')[col].mean()
print('Demand averages: ', mean_val)

quant_val = df_hh.groupby('category')[col].quantile(0.8)
print('Demand 80th percentiles: ', quant_val)

# bills
col = 'Bill'
mean_val = df_hh.groupby('category')[col].mean()
print('Bill averages: ', mean_val)

quant_val = df_hh.groupby('category')[col].quantile(0.5)
print('Bill medians: ', quant_val)

# ARs
col = 'AR_v2'
mean_val = df_hh.groupby('category')[col].mean()
print('AR averages: ', mean_val)

quant_val = df_hh.groupby('category')[col].quantile(0.5)
print('AR 50th percentiles: ', quant_val)

quant_val = df_hh.groupby('category')[col].quantile(0.8)
print('AR 80th percentiles: ', quant_val)