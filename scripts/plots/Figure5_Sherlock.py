#%% import packages
import processing_functions as pf
import numpy as np
import pandas as pd
import ast
# plotting packages
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D  # For custom legend handles
import matplotlib.transforms as mtransforms
print('packages imported')

#%% import functions
# function to separate decision variables into numpy array for rof thresholds and df for inf order
def separate_decision_vars(decision_variables):
    rof_thresholds = decision_variables[:, 0]
    rank_order = np.argsort(np.argsort(decision_variables[:, 1:])) + 1
    df_rank_order = pd.DataFrame(rank_order, columns=['transfer_soquel', 'transfer_sv', 'mcasr', 'desal', 'dpr'])

    # reorder columns
    new_order = ['desal', 'dpr', 'mcasr', 'transfer_soquel', 'transfer_sv']
    df_rank_order = df_rank_order[new_order]
    return rof_thresholds, df_rank_order

#%% process household-level data- baseline policy (Policy A)
filepath_save = '../../../../../scratch/users/jskerker/AffordPaper1/Figure5/'
# reprocess hh level data- Policy A
filepath = '../../../../../scratch/users/jskerker/AffordPaper1/Figure4/data_01Dec2025/'
name_add = ''
# get combinations to loop through
df_combinations = pd.read_csv(filepath + 'climate_scenarios_500.csv')
combinations = df_combinations.values.tolist()

# uncomment these lines to reprocess data
df_hh_data = pf.compile_hh_data_ARquants(filepath, combinations, name_add)
df_hh_data.to_csv(filepath_save + 'df_hh_data_ARquants_BaselineROF.csv')
print('processed data for Policy A')

#%% process household-level data-
# reprocess hh level data- low rof policy: rof=0.05, first inf=desal
filepath = '../../../../../scratch/users/jskerker/AffordPaper1/Figure5/LowROF_RandomSims/'
name_add = ''
# get combinations to loop through
df_combinations = pd.read_csv(filepath + 'climate_scenarios_500.csv')
combinations = df_combinations.values.tolist()

df_hh_data = pf.compile_hh_data_ARquants(filepath, combinations, name_add)
df_hh_data.to_csv(filepath_save + 'df_hh_data_ARquants_LowROF.csv')
print('processed data for Policy B')

#%% process household-level data-
# reprocess hh level data- hifh rof policy: rof=0.939, first inf=mcasr
filepath = '../../../../../scratch/users/jskerker/AffordPaper1/Figure5/HighROF_RandomSims/'
name_add = ''
# get combinations to loop through
df_combinations = pd.read_csv(filepath + 'climate_scenarios_500.csv')
combinations = df_combinations.values.tolist()

df_hh_data = pf.compile_hh_data_ARquants(filepath, combinations, name_add)
df_hh_data.to_csv(filepath_save + 'df_hh_data_ARquants_HighROF.csv')
print('processed data for Policy C')

#%% reimport data- baseline policy
# baseline policy: rof=0.65, first inf=desal
# PROCESS DATA
filepath = '../../../../../scratch/users/jskerker/AffordPaper1/Figure4/data_01Dec2025/'

# 1. Import df hh data
#df_hh = pd.read_csv(filepath_save + 'df_hh_data_ARquants_BaselineROF.csv')

# 2. Get SOW metrics
# import and aggregate data for ROF=0.65, first inf=desal
count = 500
rof = 0.65
inf_order = ['desal', 'dpr', 'mcasr', 'transfer_soquel', 'transfer_sv']
# 1. desal, 2. dpr, 3. mcasr, 4. transfer soquel, 5. transfer sv
df_combinations = pd.read_csv(filepath + 'climate_scenarios_500.csv')
climate_combinations = df_combinations.values.tolist()
df_policyA_desal = pf.aggregate_sows_for_policy(filepath, rof, inf_order, climate_combinations, 2020, 2071)

# 3. Merge SOW & AR data
df_hh_policyA = df_policyA_desal.merge(df_hh, on=['real', 'dT', 'dP', 'dCV', 'demand'])
#print('policy A: ', df_hh_policyA)
df_hh_policyA.to_csv(filepath_save + 'df_hh_policyA_Figure5.csv', index=False)
# read in data
df_hh_policyA = pd.read_csv(filepath_save + 'df_hh_policyA_Figure5.csv')

#%% reimport data- low ROF policy
# PROCESS DATA
filepath = '../../../../../scratch/users/jskerker/AffordPaper1/Figure5/LowROF_RandomSims/'

# 1. Import df hh data
df_hh = pd.read_csv(filepath_save + 'df_hh_data_ARquants_LowROF.csv')

# 2. Get SOW metrics
# import and aggregate data for ROF=0.05, first inf=MCASR
count = 500
rof = 0.05
inf_order = ['desal', 'transfer_soquel', 'dpr', 'mcasr', 'transfer_sv']
# 1. desal, 2. transfer soquel, 3. dpr, 4. mcasr, 5. transfer sv
df_combinations = pd.read_csv(filepath + 'climate_scenarios_500_2025-12-06.csv')
climate_combinations = df_combinations.values.tolist()
df_policyB_desal = pf.aggregate_sows_for_policy(filepath, rof, inf_order, climate_combinations, 2020, 2071)

# 4. Merge SOW & AR data
df_hh_policyB = df_policyB_desal.merge(df_hh, on=['real', 'dT', 'dP', 'dCV', 'demand'])
#print('policy B: ', df_hh_policyB)
df_hh_policyB.to_csv(filepath_save + 'df_hh_policyB_Figure5.csv', index=False)
df_hh_policyB = pd.read_csv(filepath_save + 'df_hh_policyB_Figure5.csv')

#%% reimport data- high ROF policy
filepath = '../../../../../scratch/users/jskerker/AffordPaper1/Figure5/HighROF_RandomSims/'

# 1. Import df hh data
df_hh = pd.read_csv(filepath_save + 'df_hh_data_ARquants_HighROF.csv')

# 2. Get SOW metrics
# import and aggregate data for ROF=0.91, first inf=MCASR
count = 500
rof = 0.93
inf_order = ['mcasr', 'transfer_sv', 'dpr', 'desal', 'transfer_soquel']
# 1. mcasr, 2. transfer sv, 3. dpr, 4. desal, 5.transfer soquel
df_combinations = pd.read_csv(filepath + 'climate_scenarios_500.csv')
climate_combinations = df_combinations.values.tolist()
df_policyC_mcasr = pf.aggregate_sows_for_policy(filepath, rof, inf_order, climate_combinations, 2020, 2071)

# 3. Merge SOW & AR data
df_hh_policyC = df_policyC_mcasr.merge(df_hh, on=['real', 'dT', 'dP', 'dCV', 'demand'])
print('policy C: ', df_hh_policyC)
df_hh_policyC.to_csv(filepath_save + 'df_hh_policyC_Figure5.csv', index=False)
df_hh_policyC = pd.read_csv(filepath_save + 'df_hh_policyC_Figure5.csv')

#%% Get pareto frontier values from optimization
filename = 'matched_decision_variables.txt'

# initialize lists
decision_variables = []
objective_values = []

# Initialize lists
decision_variables = []
objective_values = []

# Open and process file
filename = filepath_save + filename
with open(filename, 'r') as file:
    for line in file:
        line = line.strip()
        if not line:
            continue

        # Split at '#' to separate decision variables and objective tuple
        if '#' in line:
            parts = line.split('#')
            decision_str = parts[0].strip()
            comment_str = parts[1].strip()
        else:
            decision_str = line
            comment_str = ""

        # Convert decision variables
        decision_vals = list(map(float, decision_str.split()))

        # Parse objective values from the comment
        if 'Objectives:' in comment_str:
            obj_str = comment_str.split('Objectives:')[1].strip()
            obj_vals = ast.literal_eval(obj_str)  # Safe way to parse a tuple from string
        else:
            obj_vals = []

        # Append if valid
        if len(decision_vals) > 0 and len(obj_vals) == 2:
            decision_variables.append(decision_vals)
            objective_values.append(obj_vals)

# Now you have lists of decision_variables and objective_values
print("Parsed", len(decision_variables), "entries.")
print(decision_variables)

# Convert to numpy arrays for easier handling (optional)
decision_variables = np.array(decision_variables)
objective_values = np.array(objective_values)
rof_thresholds, df_rank_order = separate_decision_vars(decision_variables)

# process objective values data
objective_values = objective_values * -1

# Divide the second column (costs) by 1e7- 10s of millions of dollars
objective_values[:, 1]  #/= 1000000 # 0

# get dataframe with name of first value for each row
df_inf_first = pd.DataFrame(
    {'FirstInf': df_rank_order.apply(lambda row: row.idxmin() if 1 in row.values else None, axis=1)})
df_inf = df_inf_first
print('imported data for panel a- optimal policies')

#%% create plot
fig = plt.figure(figsize=(12, 8))  # Adjust figure size
gs = gridspec.GridSpec(3, 4, width_ratios=[3, 1, 3, 1], height_ratios=[2, 1, 2], wspace=0.15, hspace=0.15)  # Varying column widths

marker_map = { "transfer_soquel": "*", "desal": "s", "dpr": "X", "mcasr": "o"} # , "transfer_sv": ">"
label_map = {
    "transfer_soquel": "Transfer option 1",
    "desal": "Desalination",
    "dpr": "Direct potable reuse",
    "mcasr": "Aquifer storage & recharge"
} #     "transfer_sv": "T2",
norm = mcolors.Normalize(vmin=0, vmax=1)

# Define colors for the segments
colors = ['lightcyan', 'skyblue', 'darkturquoise', 'teal', 'darkslategrey'] # 'dodgerblue'- for #4
# Create a custom colormap
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

### First plot: pareto frontier ###
ax0 = fig.add_subplot(gs[0:2, 1:3])
# set gray background
ax0.set_facecolor('#EAEAF2') # light gray
ax0.grid(True, color='white', zorder=0) # white gridlines
# Remove the spines (outline)
for spine in ax0.spines.values():
    spine.set_visible(False)

cmap = 'GnBu'
# for loop to go through each marker_map item
for typ, marker in marker_map.items():
    idx = df_inf['FirstInf'] == typ
    subset_obj_vals =  objective_values[idx]
    subset_rof = rof_thresholds[idx]
    scatter = ax0.scatter(subset_obj_vals[:, 0], subset_obj_vals[:, 1]/50, c=subset_rof, norm=norm, cmap=custom_cmap, s=33, marker=marker, label=label_map[typ],  # Use the label_map
     edgecolors='gray', linewidth=0.5, zorder=3)

# colorbar and labels
sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
sm.set_array([])  # To prevent a warning when adding the colorbar
colorbar = plt.colorbar(sm, ax=ax0)
# Move the colorbar
colorbar.ax.set_position([0.67, 0.555, 0.04, 0.33])
# Set the colorbar label and ticks font size
colorbar.set_label('ROF Threshold', fontsize=11)  # Set the label and its font size
colorbar.ax.tick_params(labelsize=11)

# legend
legend = ax0.legend(title="First infrastructure option", loc='upper right', fontsize=8.5, title_fontsize=8.5)
legend.get_title().set_fontweight("bold")
legend.set_frame_on(False)

# add line for 98.5% reliability
#ax00.set_ylim(-50, 870)
ax0.set_xlim(0, 80)
ax0.set_ylim(-0.5, 18)
ax0.set_xlabel('Average unmet demand (MG/yr)', fontsize=11)
ax0.set_ylabel('Added supply cost ($M/yr)', fontsize=11)
ax0.set_xticks(ticks=[0, 20, 40, 60, 80], labels=[0, 20, 40, 60, 80], fontsize=11)
ax0.set_yticks(ticks=[0, 3, 6, 9, 12, 15, 18], labels=[0, 3, 6, 9, 12, 15, 18], fontsize=11)
ax0.tick_params(axis='both', which='both', length=0)
ax0.set_title('Pareto frontier', fontsize=11, fontweight='bold')

# move subplot
left, bottom, width, height = ax0.get_position().bounds
ax0.set_position([left-0.15, bottom+0.12, width+0.1, height-0.1])

# Add a label with a pointer to a specific point- A (obj values: 25.46, 375.98/7.52)
ax0.annotate(
    'Strategy A',        # Text for the label
    xy=(25.5, 7.5),                # Location of the point
    xytext=(28.5, 9.1),           # Location of the text
    arrowprops=dict(
        arrowstyle='-',      # Style of the arrow
        color='black',        # Color of the arrow
        lw=1.5                # Line width of the arrow
    ),
    fontsize=10.5,              # Font size of the label text
    color='salmon',          # Color of the label text
    fontweight='bold'
)

# Add a label with a pointer to a specific point- B (obj values: 16.6, 666.6/13.33)
ax0.annotate(
    'Strategy B',        # Text for the label
    xy=(16.6, 13.3),                # Location of the point
    xytext=(21, 14.2),           # Location of the text
    arrowprops=dict(
        arrowstyle='-',      # Style of the arrow
        color='black',        # Color of the arrow
        lw=1.5               # Line width of the arrow
    ),
    fontsize=10.5,              # Font size of the label text
    color='olivedrab', # Color of the label text
    fontweight='bold'
)

# Add a label with a pointer to a specific point- B (obj values: 63.6, 0)
ax0.annotate(
    'Strategy C',        # Text for the label
    xy=(53.6, 0.45),                # Location of the point
    xytext=(54.5, 2),           # Location of the text
    arrowprops=dict(
        arrowstyle='-',      # Style of the arrow
        color='black',        # Color of the arrow
        lw=1.5               # Line width of the arrow
    ),
    fontsize=10.5,              # Font size of the label text
    color='mediumturquoise',              # Color of the label text
    fontweight='bold'
)

### subplot 2: kde plot of unmet demands ###
ax2 = fig.add_subplot(gs[1, 0])
col = 'unmetDemandMG'
bw = 1
sns.kdeplot(data=df_hh_policyA[col], ax=ax2, color='salmon', alpha=0.8, linewidth=1.8, label='Policy A', bw_adjust=bw)
sns.kdeplot(data=df_hh_policyB[col], ax=ax2, color='olivedrab', alpha=0.8, linewidth=1.8, label='Policy B', bw_adjust=bw)
sns.kdeplot(data=df_hh_policyC[col], ax=ax2, color='mediumturquoise', alpha=0.8, linewidth=1.8, label='Policy C', bw_adjust=bw)
# x limits and ticks
ax2.set_xlim(-50, 450)
ax2.set_xticks(np.arange(0, 401, 100))
ax2.set_xticklabels(['', '', '', '', ''])
# y limits and ticks
ax2.set_yticks(np.arange(0, 0.021, 0.01))
ax2.set_yticklabels(['', '', ''])
ax2.tick_params(axis='both', which='both', length=0)
# labels and background
ax2.set_xlabel('')
ax2.set_ylabel('')
ax2.set_facecolor('white')  # Sets background to white
ax2.patch.set_visible(False)  # Hides the background completely
ax2.grid(False) # remove grid lines
# Remove all spines (borders)
for spine in ax2.spines.values(): # get rid of spines (borders)
    spine.set_visible(False)
# move subplot
left, bottom, width, height = ax2.get_position().bounds
ax2.set_position([left, bottom-0.02, width, height])

### subplot 3: kde plot of reliability ###
ax3 = fig.add_subplot(gs[1, 2])
col = 'percReliability'
bw = 1
sns.kdeplot(data=df_hh_policyA[col], ax=ax3, color='salmon', alpha=0.8, linewidth=1.8, label='Policy A', bw_adjust=bw)
sns.kdeplot(data=df_hh_policyB[col], ax=ax3, color='olivedrab', alpha=0.8, linewidth=1.8, label='Policy B', bw_adjust=bw)
sns.kdeplot(data=df_hh_policyC[col], ax=ax3, color='mediumturquoise', alpha=0.8, linewidth=1.8, label='Policy C', bw_adjust=bw)
# x limits and ticks
ax3.set_xlim(85, 101)
ax3.set_xticks(np.arange(85, 101, 5))
ax3.set_xticklabels(['', '', '', ''])
# y limits and ticks
ax3.set_ylim(0, 0.5)
ax3.set_yticks(np.arange(0, 0.51, 0.1))
ax3.set_yticklabels(['', '', '', '', '', ''])
ax3.tick_params(axis='both', which='both', length=0)
# labels
ax3.set_xlabel('')
ax3.set_ylabel('')
# set background to be white
ax3.set_facecolor('white')  # Sets background to white
ax3.patch.set_visible(False)  # Hides the background completely
ax3.grid(False) # remove grid lines
# Remove all spines (borders)
for spine in ax3.spines.values(): # get rid of spines (borders)
    spine.set_visible(False)
# move subplot
left, bottom, width, height = ax3.get_position().bounds
ax3.set_position([left, bottom-0.02, width, height])

### subplot 4: scatter plot of avg supply costs vs unmet demands ###
ax4 = fig.add_subplot(gs[2, 0])
# set gray background
ax4.set_facecolor('#EAEAF2') # light gray
ax4.grid(True, color='white', zorder=0) # white gridlines
# Remove the spines (outline)
for spine in ax4.spines.values():
    spine.set_visible(False)
col = 'cashflow_cost'
ax4.scatter(df_hh_policyA['unmetDemandMG'], df_hh_policyA['cashflow_cost'], c='salmon', marker='o', s=8, label='Policy A', alpha=0.3, zorder=3)
ax4.scatter(df_hh_policyB['unmetDemandMG'], df_hh_policyB['cashflow_cost'], c='olivedrab', marker='o', s=8, label='Policy B', alpha=0.2, zorder=3)
ax4.scatter(df_hh_policyC['unmetDemandMG'], df_hh_policyC['cashflow_cost'], c='mediumturquoise',  s=8, marker='o', label='Policy C', alpha=0.3, zorder=3)
ax4.set_xlim(-50, 450)
ax4.set_xticks(np.arange(0, 401, 100))
ax4.set_xticklabels(np.arange(0, 401, 100), fontsize=11)
#ax4.set_ylim(0, 20)
ax4.set_xlabel('Average unmet demand (MG/yr)', fontsize=11)
ax4.set_ylabel('Added supply cost ($M/yr)', fontsize=11)
ax4.set_ylim(-1, 18)
ax4.set_yticks(np.arange(0, 18, 5))
ax4.set_yticklabels(np.arange(0, 18, 5), fontsize=11)
ax4.tick_params(axis='both', which='both', length=0)

### subplot 5: kde plot of cashflow costs ###
ax5 = fig.add_subplot(gs[2, 1])
col = 'cashflow_cost'
bw = 1.4
sns.kdeplot(data=df_hh_policyA, y=col, ax=ax5, color='salmon', alpha=0.8, linewidth=1.8, label='Policy A', bw_adjust=bw)
sns.kdeplot(data=df_hh_policyB, y=col, ax=ax5, color='olivedrab', alpha=0.8, linewidth=1.8, label='Policy B', bw_adjust=bw)
sns.kdeplot(data=df_hh_policyC, y=col, ax=ax5, color='mediumturquoise', alpha=0.8, linewidth=1.8, label='Policy C', bw_adjust=bw)
# y axes
ax5.set_ylim(-1, 18)
ax5.set_yticks(np.arange(0, 18, 5))
ax5.set_yticklabels(['', '', '', ''])
# x axes
ax5.set_xlim(-0.05, 0.8)
ax5.set_xticks(np.arange(0, 1.01, 0.5))
ax5.set_xticklabels(['', '', ''])
ax5.set_ylabel('')
ax5.set_xlabel('')
ax5.tick_params(axis='both', which='both', length=0)
# set background to be white
ax5.set_facecolor('white')  # Sets background to white
ax5.patch.set_visible(False)  # Hides the background completely
ax5.grid(False) # remove grid lines
for spine in ax5.spines.values():
    spine.set_visible(False)
# move subplot
left, bottom, width, height = ax5.get_position().bounds
ax5.set_position([left-0.02, bottom, width, height])

### subplot 6: scatter plot of median AR vs reliability ###
ax6 = fig.add_subplot(gs[2, 2])
# set gray background
ax6.set_facecolor('#EAEAF2') # light gray
ax6.grid(True, color='white', zorder=0) # white gridlines
# Remove the spines (outline)
for spine in ax6.spines.values():
    spine.set_visible(False)
ax6.scatter(df_hh_policyA['percReliability'], df_hh_policyA['AR_80'], c='salmon', marker='o', s=8, label='Strategy A: Baseline', alpha=0.3, zorder=3)
ax6.scatter(df_hh_policyB['percReliability'], df_hh_policyB['AR_80'], c='olivedrab', marker='o', s=8, label='Strategy B: Low ROF, Desal. First', alpha=0.2, zorder=3)
ax6.scatter(df_hh_policyC['percReliability'], df_hh_policyC['AR_80'], c='mediumturquoise',  s=8, marker='o', label='Strategy C: High ROF, ASR First', alpha=0.3, zorder=3)
ax6.plot([84, 101], [2.5, 2.5], linestyle='--', color='k', linewidth=1.5, zorder=4)
ax6.set_xlim(84, 101)
ax6.set_xticks(np.arange(85, 101, 5))
ax6.set_xticklabels(np.arange(85, 101, 5), fontsize=11)
ax6.set_xlabel('Reliability (%)', fontsize=11)
ax6.set_ylabel('80th perc. AR (% of bill / income)', fontsize=11)
ax6.text(84.2, 2.6, 'EPA affordability threshold', fontsize=10, fontstyle='italic')
ax6.set_ylim(0, 7)
ax6.set_yticks(np.arange(0, 7.01, 1))
ax6.set_yticklabels(np.arange(0, 7.01, 1), fontsize=11)
# update yticklabels location
offset = mtransforms.ScaledTranslation(-6/72, 0, fig.dpi_scale_trans)
for label in ax6.get_yticklabels():
    label.set_transform(label.get_transform() + offset)
ax6.tick_params(axis='both', which='both', length=0)
# custom legend
sz = 5
custom_legend = [
    Line2D([0], [0], marker='o', color='none', linestyle='None', markerfacecolor='salmon', markersize=sz, markeredgecolor='none', label='Strategy A: Baseline'),
    Line2D([0], [0], marker='o', color='none', linestyle='None', markerfacecolor='olivedrab', markersize=sz, markeredgecolor='none', label='Strategy B: Low ROF, Desal. First'),
    Line2D([0], [0], marker='o', color='none', linestyle='None', markerfacecolor='mediumturquoise', markersize=sz, markeredgecolor='none', label='Strategy C: High ROF, ASR First')
]
legend = ax6.legend(handles=custom_legend, loc='lower left', fontsize=8.5, handletextpad=0.4, bbox_to_anchor=(-0.04, -0.03))
legend.set_frame_on(False)


### subplot 5: kde plot of median ARs ###
ax7 = fig.add_subplot(gs[2, 3])
col = 'AR_80'
bw = 1.4
sns.kdeplot(data=df_hh_policyA, y=col, ax=ax7, color='salmon', alpha=0.8, linewidth=1.8, label='Policy A', bw_adjust=bw)
sns.kdeplot(data=df_hh_policyB, y=col, ax=ax7, color='olivedrab', alpha=0.8, linewidth=1.8, label='Policy B', bw_adjust=bw)
sns.kdeplot(data=df_hh_policyC, y=col, ax=ax7, color='mediumturquoise', alpha=0.8, linewidth=1.8, label='Policy C', bw_adjust=bw)
ax7.set_ylim(0, 7)
ax7.set_yticks(np.arange(0, 7.01, 1))
ax7.set_yticklabels(['', '', '', '', '', '', '', ''])
ax7.set_xlim(-0.2, 6)
ax7.set_xticks(np.arange(0, 6.1, 2))
ax7.set_xticklabels(['', '', '', ''])
ax7.tick_params(axis='both', which='both', length=0)
ax7.set_ylabel('')
ax7.set_xlabel('')
# set background to be white
ax7.set_facecolor('white')  # Sets background to white
ax7.patch.set_visible(False)  # Hides the background completely
ax7.grid(False) # remove grid lines
for spine in ax7.spines.values():
    spine.set_visible(False)
# move subplot
left, bottom, width, height = ax7.get_position().bounds
ax7.set_position([left-0.02, bottom, width, height])

# add text for a-c labels
ax6.text(67, 19.1, 'a', fontsize=18, fontweight='bold')
ax6.text(58.0, 6.3, 'b', fontsize=18, fontweight='bold')
ax6.text(84.15, 6.3, 'c', fontsize=18, fontweight='bold')

# save figure
plt.savefig('../outputs/Figure5_Policy_comparison.png', dpi=300, bbox_inches='tight')

#%% print out statistics
print('POLICY B')
print('average reliability: {}, min: {}, max: {}'.format(df_hh_policyB['percReliability'].mean(), df_hh_policyB['percReliability'].min(), df_hh_policyB['percReliability'].max()))
print('average cost: {}, min: {}, max: {}'.format(df_hh_policyB['cashflow_cost'].mean(), df_hh_policyB['cashflow_cost'].min(), df_hh_policyB['cashflow_cost'].max()))
print('average AR80: {}, min: {}, max: {}'.format(df_hh_policyB['AR_80'].mean(), df_hh_policyB['AR_80'].min(), df_hh_policyB['AR_80'].max()))

print('POLICY A')
print('average reliability: {}, min: {}, max: {}'.format(df_hh_policyA['percReliability'].mean(), df_hh_policyA['percReliability'].min(), df_hh_policyA['percReliability'].max()))
print('average cost: {}, min: {}, max: {}'.format(df_hh_policyA['cashflow_cost'].mean(), df_hh_policyA['cashflow_cost'].min(), df_hh_policyA['cashflow_cost'].max()))
print('average AR80: {}, min: {}, max: {}'.format(df_hh_policyA['AR_80'].mean(), df_hh_policyA['AR_80'].min(), df_hh_policyA['AR_80'].max()))

print('POLICY C')
print('average reliability: {}, min: {}, max: {}'.format(df_hh_policyC['percReliability'].mean(), df_hh_policyC['percReliability'].min(), df_hh_policyC['percReliability'].max()))
print('average cost: {}, min: {}, max: {}'.format(df_hh_policyC['cashflow_cost'].mean(), df_hh_policyC['cashflow_cost'].min(), df_hh_policyC['cashflow_cost'].max()))
print('average AR80: {}, min: {}, max: {}'.format(df_hh_policyC['AR_80'].mean(), df_hh_policyC['AR_80'].min(), df_hh_policyC['AR_80'].max()))