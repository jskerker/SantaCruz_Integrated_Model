# Import packages
import numpy as np
import pandas as pd
import itertools
from matplotlib import pyplot as plt
import matplotlib as mpl
from cycler import Cycler
import matplotlib.gridspec as gridspec
import processing_functions_March2025 as pf
import json
import warnings
warnings.filterwarnings("ignore")
from statsmodels.distributions.empirical_distribution import ECDF
from matplotlib.ticker import MultipleLocator
import psutil

def optimize_df(df):
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

def print_memory_usage():
    process = psutil.Process()
    print(f"Memory Usage: {process.memory_info().rss / 1e9:.2f} GB")

print('imported packages')


# import data
filepath_NoInf = '../../../../../scratch/users/jskerker/AffordPaper1/Figure2/'
# get monthly data- no inf
df_modcool_NoInf = pd.read_csv(filepath_NoInf + 'df_modcool_NoInf.csv')

# get monthly data- modcoo, dryhot
filepath = '../../../../../scratch/users/jskerker/AffordPaper1/SA/Climate/'
df_modcool = pd.read_csv(filepath + 'df_modcool.csv')
df_dryhot = pd.read_csv(filepath + 'df_dryhot.csv')
print('imported monthly data')

# import income estimation data
df_income = pd.read_csv(
    '../../../../../scratch/users/jskerker/AffordPaper1/Figure2/resampled_income_data_30Nov2024.csv')
df_income = df_income.rename(columns={'account': 'Account'})
df_income = df_income[['Account', 'map_inc_2']]

# import household data
columns = ['Account', 'Bill'] # , 'AR'
print('columns to import: ', columns)
df_hh_modcool = pd.read_parquet(filepath + "df_hh_modcool_long.parquet", columns=columns)
df_hh_modcool = pd.merge(df_hh_modcool, df_income, how='left', on='Account') # merge with income data
df_hh_modcool['AR'] = df_hh_modcool['Bill'] / (df_hh_modcool['map_inc_2']/12) * 100 # calculate AR
print('imported modcool data')

df_hh_modcool_NoInf = pd.read_parquet(filepath_NoInf + "df_hh_modcool_NoInf_long.parquet", columns=columns)
df_hh_modcool_NoInf = pd.merge(df_hh_modcool_NoInf, df_income, how='left', on='Account') # merge with income data
df_hh_modcool_NoInf['AR'] = df_hh_modcool_NoInf['Bill'] / (df_hh_modcool_NoInf['map_inc_2']/12) * 100 # calculate AR
print('imported modcool no inf data')

df_hh_dryhot = pd.read_parquet(filepath + "df_hh_dryhot_long.parquet", columns=columns)
df_hh_dryhot = pd.merge(df_hh_dryhot, df_income, how='left', on='Account') # merge with income data
df_hh_dryhot['AR'] = df_hh_dryhot['Bill'] / (df_hh_dryhot['map_inc_2']/12) * 100 # calculate AR
print('imported dry hot data')

# use downcasting for numerical data to reduce memory usage
df_hh_modcool = optimize_df(df_hh_modcool)
df_hh_modcool_NoInf = optimize_df(df_hh_modcool_NoInf)
df_hh_dryhot = optimize_df(df_hh_dryhot)
print('used downcasting to reduce memory usage')

# update matplotlib params
# Load rcParams
with open("rcparams.json", "r") as f:
    loaded_rcparams = json.load(f)
# Remove the backend setting if it exists
loaded_rcparams.pop("backend", None)
mpl.use("Agg")  # Use a non-interactive backend suitable for Slurm
# Convert back if necessary (e.g., Cycler objects won't be restored properly)
mpl.rcParams.update(loaded_rcparams)

# NEW VERSION OF FIGURE 2
# Figure 2- v2- cdfs of (a) water availability, (b) utility revenue, (c) water bills, (d) affordability ratios

# set up figure
fig = plt.figure(figsize = (8, 6))
# set up subplots
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], wspace=0.3, hspace=0.45)
print('set up figure')
### subplot 1: Water Availability ###
ax00 = fig.add_subplot(gs[0, 0])
col =  'LL_Reservoir_MG' # 'waterAvail' # 'SLR_BigTrees' #
ecdf_hist = ECDF(df_modcool_NoInf[col])
ecdf_base = ECDF(df_modcool[col])
ecdf_cc = ECDF(df_dryhot[col])
ax00.step(ecdf_hist.x, ecdf_hist.y, color='gray', lw=1.8, where='post')
ax00.step(ecdf_base.x, ecdf_base.y, color='teal', lw=1.8, where='post')
ax00.step(ecdf_cc.x, ecdf_cc.y, color='salmon', lw=1.8, where='post')
ax00.set_ylabel('CDF', fontsize=11)
ax00.set_yticks(np.arange(0, 1.01, 0.1))
ax00.set_yticklabels(['0.0', '', '0.2', '', '0.4', '', '0.6', '', '0.8', '', '1.0'], fontsize=11)
#ax00.set_xlim(750, 1100)
ax00.set_ylim(-0.02, 1.02) #(-0.02, 1.02)
#ax00.set_ylim(-0.01, 0.1)
#ax00.set_facecolor('gainsboro')  # Background of the plotting area
#ax00.grid(True, color='white', linewidth=0.5)
ax00.invert_xaxis()

#ax00.legend(loc='upper left')

# water availability code
# ax00.set_xlabel('Water availability (MG/yr)', fontsize=11)
#ax00.set_xticks(np.arange(800, 2901, 200))
#ax00.set_xticklabels(['800', '', '1200', '', '1600', '', '2000', '', '2400', '', '2800'], fontsize=11)
# ax00.set_title('Water Supply Availability', fontsize=11, fontweight='bold')

# reservoir storage code
#ax00.plot([2860, 2860], [-0.02, 1.02], color='k', linestyle='--')
ax00.set_xlabel('Storage (MG)', fontsize=11)
ax00.set_title('Reservoir Storage', fontsize=12, fontweight='bold')
#ax00.text(2815, 0.02, 'Max. \nStorage', fontstyle='italic', fontsize=10, rotation=90)

### subplot 2: new supply costs ($M) ###
ax01 = fig.add_subplot(gs[0, 1])
col = 'rev_mo_M'
ecdf_hist = ECDF(df_modcool_NoInf[col])
ecdf_base = ECDF(df_modcool[col])
ecdf_cc = ECDF(df_dryhot[col])
ax01.step(ecdf_hist.x, ecdf_hist.y, color='gray', lw=1.8, where='post')
ax01.step(ecdf_base.x, ecdf_base.y, color='teal', lw=1.8, where='post')
ax01.step(ecdf_cc.x, ecdf_cc.y, color='salmon', lw=1.8, where='post')
ax01.set_xlabel('Costs ($M/month)', fontsize=11)
ax01.set_ylabel('CDF', fontsize=11)
ax01.set_xticks(np.arange(0, 2.1, 0.5))
ax01.set_xlim(-0.1, 2)
#ax01.set_xticklabels([0, 3, 6, 9, 12, 15, 18], fontsize=11)
ax01.set_yticks(np.arange(0, 1.01, 0.1))
ax01.set_yticklabels(['0.0', '', '0.2', '', '0.4', '', '0.6', '', '0.8', '', '1.0'], fontsize=11)
ax01.set_ylim(-0.02, 1.02)
#ax01.grid(True, color='gray', linewidth=0.5)
ax01.set_title('New Supply Costs', fontsize=11, fontweight='bold')

### subplot 3: monthly bills- median for every HH ###
ax10 = fig.add_subplot(gs[1, 0])
col = 'Bill'
ecdf_hist = ECDF(df_hh_modcool_NoInf[col])
ecdf_base = ECDF(df_hh_modcool[col])
ecdf_cc = ECDF(df_hh_dryhot[col])
ax10.step(ecdf_hist.x, ecdf_hist.y, color='gray', lw=1.8, label='Baseline, No Inf.', where='post')
ax10.step(ecdf_base.x, ecdf_base.y, color='teal', lw=1.8, label='Baseline Climate.', where='post')
ax10.step(ecdf_cc.x, ecdf_cc.y, color='salmon', lw=1.8, label='Dry, Hot Change', where='post')
ax10.set_xlabel('Bill ($/month)', fontsize=11)
ax10.set_ylabel('CDF', fontsize=11)
ax10.set_xticks(np.arange(0, 405, 50))
ax10.set_xticklabels(['0', '', '100', '', '200', '', '300', '', '400'], fontsize=11)
ax10.set_yticks(np.arange(0, 1.01, 0.1))
ax10.set_yticklabels(['0.0', '', '0.2', '', '0.4', '', '0.6', '', '0.8', '', '1.0'], fontsize=11)
ax10.set_ylim(-0.02, 1.02)
ax10.set_xlim(-0, 400)
#ax10.grid(True, color='gray', linewidth=0.5)
ax10.set_title('Water Bills', fontsize=11, fontweight='bold')

### subplot 4: monthly ARs- median for every HH ###
ax11 = fig.add_subplot(gs[1, 1])
col = 'AR'
ecdf_hist = ECDF(df_hh_modcool_NoInf[col])
ecdf_base = ECDF(df_hh_modcool[col])
ecdf_cc = ECDF(df_hh_dryhot[col])
ax11.step(ecdf_hist.x, ecdf_hist.y, color='gray', lw=1.8, label='Baseline', where='post')
ax11.step(ecdf_base.x, ecdf_base.y, color='teal', lw=1.8, label='Moderate Climate \nwith Adaptation', where='post')
ax11.step(ecdf_cc.x, ecdf_cc.y, color='salmon', lw=1.8, label='Dry Climate \nwith Adaptation', where='post')
ax11.plot([2.5, 2.5], [-0.05, 1.05], color='k', linestyle='--')
ax11.set_xlabel('AR (% of bill / income)', fontsize=11)
ax11.set_ylabel('CDF', fontsize=11)
ax11.set_xticks(np.arange(0, 21, 2))
ax11.set_xticklabels(np.arange(0, 21, 2), fontsize=11)
ax11.set_yticks(np.arange(0, 1.01, 0.1))
ax11.set_yticklabels(['0.0', '', '0.2', '', '0.4', '', '0.6', '', '0.8', '', '1.0'], fontsize=11)
ax11.set_ylim(-0.02, 1.02)
ax11.set_xlim(-0.2, 10)
#ax11.grid(True, color='gray', linewidth=0.5)
#ax11.tick_params(axis='y', which='minor', length=0.1, width=1)     # Minor ticks
ax11.text(1.95, 0.04, 'EPA threshold', fontstyle='italic', fontsize=10, rotation=90)
# legend
legend = ax11.legend(loc='lower right', fontsize=9, bbox_to_anchor=(1.03, -0.035))
legend.set_frame_on(False)
ax11.set_title('Affordability Ratios', fontsize=11, fontweight='bold')

# add text labels a-d
sz = 18
ax11.text(-13.4, 2.56, 'a', fontsize=sz, fontweight='bold')
ax11.text(-0.2, 2.56, 'b', fontsize=sz, fontweight='bold')
ax11.text(-13.4, 1.05, 'c', fontsize=sz, fontweight='bold')
ax11.text(-0.2, 1.05, 'd', fontsize=sz, fontweight='bold')

# save figure
plt.savefig('../outputs/Figure2_CDFs-SI_map_inc_2_14Dec.png', dpi=300, bbox_inches='tight')
print('saved figure')

print('data needed for analysis:')

# 50th and 80th percentile bills
print('50th and 80th percentile bills for baseline: {} and {}'.format(np.quantile(df_hh_modcool_NoInf['Bill'], 0.5),
                                                                      np.quantile(df_hh_modcool_NoInf['Bill'], 0.8)))
print('50th and 80th percentile bills for moderate climate: {} and {}'.format(np.quantile(df_hh_modcool['Bill'], 0.5),
                                                                      np.quantile(df_hh_modcool['Bill'], 0.8)))
print('50th and 80th percentile bills for dry climate: {} and {}'.format(np.quantile(df_hh_dryhot['Bill'], 0.5),
                                                                      np.quantile(df_hh_dryhot['Bill'], 0.8)))

# find percentage of ARs > 2.5%
col = 'AR'
threshold = 2.5

# baseline scenario
percentage = (df_hh_modcool_NoInf[col] > threshold).mean() * 100
print(f"Baseline Scenario: Percentage of values greater than {threshold}: {percentage:.2f}%")
# baseline scenario- greater than or equal to
percentage = (df_hh_modcool_NoInf[col] >= threshold).mean() * 100
print(f"Percentage of values greater than or equal to {threshold}: {percentage:.2f}%")

# mod, cool
percentage = (df_hh_modcool[col] > threshold).mean() * 100
print(f"Moderate Climate: Percentage of values greater than {threshold}: {percentage:.2f}%")
# mod, cool scenario- greater than or equal to
percentage = (df_hh_modcool[col] >= threshold).mean() * 100
print(f"Percentage of values greater than or equal to {threshold}: {percentage:.2f}%")

# dry, hot
percentage = (df_hh_dryhot[col] > threshold).mean() * 100
print(f"Dry Climate: Percentage of values greater than {threshold}: {percentage:.2f}%")
# mod, cool scenario- greater than or equal to
percentage = (df_hh_dryhot[col] >= threshold).mean() * 100
print(f"Percentage of values greater than or equal to {threshold}: {percentage:.2f}%")

# print length of dfs
print('Baseline scenario: length of hh dataframe: {}'.format(len(df_hh_modcool_NoInf['Bill'])))
print('Moderate climate scenario: length of hh dataframe: {}'.format(len(df_hh_modcool['Bill'])))
print('Dry climate scenario: length of hh dataframe: {}'.format(len(df_hh_dryhot['Bill'])))

# print memory usage
print_memory_usage()  # After loading data