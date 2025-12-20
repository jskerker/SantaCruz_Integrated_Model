# low_memory_figure2.py
# Replace the ECDF/loading/plotting parts of your script with this file.
# Assumes you've already imported modules like numpy as np, pandas as pd,
# matplotlib, pyarrow.parquet as pq, gc, psutil, etc.

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import gc
import psutil
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import json
import processing_functions as pf

def print_memory_usage(prefix=""):
    process = psutil.Process()
    print(f"{prefix}Memory Usage: {process.memory_info().rss / 1e9:.2f} GB")

def optimize_df(df):
    # keep this - it helps when you do need to load small CSVs
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

def ecdf_hist_parquet(path, col, bins=2000):
    """
    Compute an ECDF for a single numeric column in a parquet file using histograms.
    Two-pass method:
      1) Get global min/max and total count across row groups,
      2) Compute histogram counts for a fixed set of bin edges,
      3) Return bin centers (x) and cumulative fraction (y), plus quantile helper.

    Returns:
      x: np.ndarray (bin centers, length=bins)
      y: np.ndarray (cumulative fraction, length=bins)
      quantile_fn: function(q) -> approximate quantile value
    """
    pqf = pq.ParquetFile(path)
    total_n = 0
    gmin = None
    gmax = None

    # First pass: compute min, max, total count
    for i in range(pqf.num_row_groups):
        rg = pqf.read_row_group(i, columns=[col])
        # read as pandas column for simplicity (row group chunks are small relative to whole file)
        s = rg.to_pandas()[col].dropna()
        if len(s) == 0:
            continue
        arr_min = s.min()
        arr_max = s.max()
        total_n += len(s)
        if gmin is None or arr_min < gmin:
            gmin = arr_min
        if gmax is None or arr_max > gmax:
            gmax = arr_max
        # free memory
        del s, rg
        gc.collect()

    if total_n == 0:
        # empty column - return empty arrays
        return np.array([]), np.array([]), lambda q: np.nan

    # handle degenerate case: all values equal
    if gmin == gmax:
        x = np.array([gmin])
        y = np.array([1.0])
        def quantile_fn(q):
            return gmin
        return x, y, quantile_fn

    # Build bin edges (fixed)
    bin_edges = np.linspace(gmin, gmax, bins + 1)
    counts = np.zeros(bins, dtype=np.int64)

    # Second pass: accumulate histogram counts
    for i in range(pqf.num_row_groups):
        rg = pqf.read_row_group(i, columns=[col])
        s = rg.to_pandas()[col].dropna().values  # numpy array for histogram
        if s.size:
            c, _ = np.histogram(s, bins=bin_edges)
            counts += c
        del s, rg, c
        gc.collect()

    # bin centers and cdf
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    cum_counts = np.cumsum(counts)
    y = cum_counts / float(total_n)

    # create quantile function (approximate via linear interpolation of histogram)
    def quantile_fn(q):
        if q <= 0:
            return gmin
        if q >= 1:
            return gmax
        idx = np.searchsorted(y, q)
        if idx == 0:
            # interpolate between left edge and first center
            left_count = 0
            left_edge = bin_edges[0]
            right_edge = bin_edges[1]
            frac_in_bin = (q * total_n - left_count) / max(counts[0], 1)
            return left_edge + (right_edge - left_edge) * min(max(frac_in_bin, 0), 1)
        # interpolate inside bin idx
        c_before = cum_counts[idx - 1] if idx - 1 >= 0 else 0
        if counts[idx] == 0:
            return bin_centers[idx]
        frac = (q * total_n - c_before) / counts[idx]
        # linear interpolation within bin
        left_edge = bin_edges[idx]
        right_edge = bin_edges[idx + 1]
        return left_edge + (right_edge - left_edge) * min(max(frac, 0.0), 1.0)

    return bin_centers, y, quantile_fn

# -----------------------
# Process data
# -----------------------
rof = 0.65
inf_order = ['dpr', 'desal', 'transfer_soquel', 'mcasr', 'transfer_sv']
# 1. dpr, 2. desal, 3. transfer soquel, 4. mcasr, 5. transfer sv
# import and aggregate historical climate data for ROF=0.65, first inf=desal
real_All = [1270, 1956, 1987, 2770, 3449, 3515, 3574, 4211, 4373, 4937]
dT_All = [0, 1]
dP_All = [100]
CV_All = ['1.0']
demand_All = ['Baseline']
combinations_modcool = list(itertools.product(real_All, dT_All, dP_All, dCV_All, demand_All))

# modcool
name_add = ''
filepath_clim = filepath + '../SA/Climate/'
df_modcool= pf.aggregate_sows_for_policy_monthly(filepath_clim, rof, inf_order, combinations_modcool, name_add) # this works
df_modcool.to_csv(filepath_clim + 'df_modcool.csv')
df_hh_modcool = pf.aggregate_sows_for_hh_data_long(filepath_clim, rof, inf_order, combinations_modcool, name_add)
df_hh_modcool.to_parquet(filepath_clim + "df_hh_modcool_long.parquet", engine="pyarrow", compression="snappy")
print('modcool: done with aggregating data')
del df_hh_modcool

# No inf
name_add = 'NoInf_'
df_modcool_NoInf = pf.aggregate_sows_for_policy_monthly(filepath, rof, inf_order, combinations_modcool, name_add) # this works
df_modcool_NoInf.to_csv(filepath + 'df_modcool_NoInf.csv')
df_hh_modcool_NoInf = pf.aggregate_sows_for_hh_data_long(filepath, rof, inf_order, combinations_modcool, name_add)
df_hh_modcool_NoInf.to_parquet(filepath + "df_hh_modcool_NoInf_long.parquet", engine="pyarrow", compression="snappy")
print('modcool, no inf: done with aggregating data')
del df_hh_modcool_NoInf

# dry, hot
dT_All = [4, 5]
dP_All = [80, 90]
dCV_All = ['1.2']
combinations_dryhot = list(itertools.product(real_All, dT_All, dP_All, dCV_All, demand_All))
name_add = ''
df_dryhot= pf.aggregate_sows_for_policy_monthly(filepath_clim, rof, inf_order, combinations_dryhot, name_add) # this works
df_dryhot.to_csv(filepath_clim + 'df_dryhot.csv')
df_hh_dryhot = pf.aggregate_sows_for_hh_data_long(filepath_clim, rof, inf_order, combinations_dryhot, name_add)
df_hh_dryhot.to_parquet(filepath + "df_hh_dryhot_long.parquet", engine="pyarrow", compression="snappy")
print('dryhot: done with aggregating data')
del df_hh_dryhot

# all climate sims
dT_All = [0, 1, 2, 3, 4, 5]
dP_All = [80, 90, 100, 110, 120]
dCV_All = ['1.0', '1.1', '1.2']
combinations_all = list(itertools.product(real_All, dT_All, dP_All, dCV_All, demand_All))
name_add = ''
df_all = pf.aggregate_sows_for_policy_monthly(filepath_clim, rof, inf_order, combinations_all, name_add) # this works
df_all.to_csv(filepath_clim + 'df_all_climate.csv')
df_hh_all = pf.aggregate_sows_for_hh_data_long(filepath_clim, rof, inf_order, combinations_all, name_add)
df_hh_all.to_parquet(filepath_clim + "df_hh_all_long.parquet", engine="pyarrow", compression="snappy")
df_hh_all = optimize_df(df_hh_all)
print('all climate: done with aggregating data')
del df_hh_all
gc.collect()


# -----------------------
# Begin figure construction
# -----------------------

# filepaths
filepath = '../../../../../scratch/users/jskerker/AffordPaper1/Figure2/'
filepath_clim = filepath + '../SA/Climate/'

# load small monthly CSVs (these are small-ish in your original script)
print("Loading monthly CSVs (expected small)...")
df_modcool = optimize_df(pd.read_csv(filepath_clim + 'df_modcool.csv'))
df_modcool_NoInf = optimize_df(pd.read_csv(filepath + 'df_modcool_NoInf.csv'))
df_dryhot = optimize_df(pd.read_csv(filepath_clim + 'df_dryhot.csv'))
df_all = optimize_df(pd.read_csv(filepath_clim + 'df_all_climate.csv'))
print_memory_usage("After loading monthly CSVs: ")

# Prepare matplotlib rcParams as you had it
with open("rcparams.json", "r") as f:
    loaded_rcparams = json.load(f)
loaded_rcparams.pop("backend", None)
mpl.use("Agg")
mpl.rcParams.update(loaded_rcparams)

# create figure
fig = plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], wspace=0.3, hspace=0.45)

# ------------
# Subplot 1: Reservoir Storage (monthly CSVs)
# ------------
ax00 = fig.add_subplot(gs[0, 0])
col = 'LL_Reservoir_MG'
# use pandas ECDF for small CSVs (these are small)
from statsmodels.distributions.empirical_distribution import ECDF
ecdf_hist = ECDF(df_modcool_NoInf[col])
ecdf_base = ECDF(df_modcool[col])
ecdf_cc = ECDF(df_dryhot[col])
ecdf_all = ECDF(df_all[col])
ax00.step(ecdf_all.x, ecdf_all.y, color='lightskyblue', lw=1.8, where='post')
ax00.step(ecdf_hist.x, ecdf_hist.y, color='gray', lw=1.8, where='post')
ax00.step(ecdf_base.x, ecdf_base.y, color='teal', lw=1.8, where='post')
ax00.step(ecdf_cc.x, ecdf_cc.y, color='salmon', lw=1.8, where='post')
ax00.set_ylabel('CDF', fontsize=11)
ax00.set_yticks(np.arange(0, 1.01, 0.1))
ax00.set_yticklabels(['0.0', '', '0.2', '', '0.4', '', '0.6', '', '0.8', '', '1.0'], fontsize=11)
ax00.set_ylim(-0.02, 1.02)
ax00.invert_xaxis()
ax00.set_xlabel('Storage (MG)', fontsize=11)
ax00.set_title('Reservoir Storage', fontsize=12, fontweight='bold')

# ------------
# Subplot 2: New Supply Costs (monthly CSVs)
# ------------
ax01 = fig.add_subplot(gs[0, 1])
col = 'rev_mo_M'
ecdf_hist = ECDF(df_modcool_NoInf[col])
ecdf_base = ECDF(df_modcool[col])
ecdf_cc = ECDF(df_dryhot[col])
ecdf_all = ECDF(df_all[col])
ax01.step(ecdf_all.x, ecdf_all.y, color='lightskyblue', lw=1.8, where='post')
ax01.step(ecdf_hist.x, ecdf_hist.y, color='gray', lw=1.8, where='post')
ax01.step(ecdf_base.x, ecdf_base.y, color='teal', lw=1.8, where='post')
ax01.step(ecdf_cc.x, ecdf_cc.y, color='salmon', lw=1.8, where='post')
ax01.set_xlabel('Costs ($M/month)', fontsize=11)
ax01.set_ylabel('CDF', fontsize=11)
ax01.set_xticks(np.arange(0, 2.1, 0.5))
ax01.set_xlim(-0.1, 2.0)
ax01.set_yticks(np.arange(0, 1.01, 0.1))
ax01.set_yticklabels(['0.0', '', '0.2', '', '0.4', '', '0.6', '', '0.8', '', '1.0'], fontsize=11)
ax01.set_ylim(-0.02, 1.02)
ax01.set_title('New Supply Costs', fontsize=11, fontweight='bold')

# Delete monthly CSV dfs to free memory (we already extracted what we need)
del df_modcool, df_modcool_NoInf, df_dryhot, df_all
gc.collect()
print_memory_usage("After deleting monthly CSVs: ")

# ------------------------------
# Subplots 3 & 4: Household ECDFs
# ------------------------------
# We'll compute histogram ECDFs from parquet files.
bins = 20000  # adjust: smaller -> less memory & less resolution; larger -> more resolution

# helper to compute and plot a scenario and return percentiles
def compute_plot_parquet_ecdf(ax, parquet_path, col, plot_color, label=None):
    x, y, qfn = ecdf_hist_parquet(parquet_path, col, bins=bins)
    if x.size == 0:
        print(f"No data found in {parquet_path} for column {col}")
        return None
    # plot
    ax.step(x, y, color=plot_color, lw=1.8 if label else 1.2, where='post', label=label)
    # compute percentiles of interest
    p50 = qfn(0.5)
    p80 = qfn(0.8)
    # free things we can
    del x, y
    gc.collect()
    return p50, p80

# subplot 3: Bills
ax10 = fig.add_subplot(gs[1, 0])
ax10.set_xlabel('Bill ($/month)', fontsize=11)
ax10.set_ylabel('CDF', fontsize=11)
ax10.set_xticks(np.arange(0, 405, 50))
ax10.set_xticklabels(['0', '', '100', '', '200', '', '300', '', '400'], fontsize=11)
ax10.set_yticks(np.arange(0, 1.01, 0.1))
ax10.set_yticklabels(['0.0', '', '0.2', '', '0.4', '', '0.6', '', '0.8', '', '1.0'], fontsize=11)
ax10.set_ylim(-0.02, 1.02)
ax10.set_xlim(-0, 400)
ax10.set_title('Water Bills', fontsize=11, fontweight='bold')

print("Computing household ECDFs (Bills)...")
p50_all, p80_all = compute_plot_parquet_ecdf(ax10, filepath_clim + "df_hh_all_long.parquet", "Bill", "lightskyblue", label="All Climate Sims")
print("50th & 80th percentiles (All):", p50_all, p80_all)
p50_hist, p80_hist = compute_plot_parquet_ecdf(ax10, filepath + "df_hh_modcool_NoInf_long.parquet", "Bill", "gray", label="Baseline, No Inf.")
print("50th & 80th percentiles (Baseline):", p50_hist, p80_hist)
p50_mod, p80_mod = compute_plot_parquet_ecdf(ax10, filepath_clim + "df_hh_modcool_long.parquet", "Bill", "teal", label="Baseline Climate.")
print("50th & 80th percentiles (Modcool):", p50_mod, p80_mod)
p50_dry, p80_dry = compute_plot_parquet_ecdf(ax10, filepath_clim + "df_hh_dryhot_long.parquet", "Bill", "salmon", label="Dry, Hot Change")
print("50th & 80th percentiles (Dryhot):", p50_dry, p80_dry)

#ax10.legend(fontsize=9, frameon=False, bbox_to_anchor=(1.03, -0.035))

# subplot 4: Affordability Ratio (AR)
ax11 = fig.add_subplot(gs[1, 1])
ax11.set_xlabel('AR (% of bill / income)', fontsize=11)
ax11.set_ylabel('CDF', fontsize=11)
ax11.set_xticks(np.arange(0, 21, 2))
ax11.set_xticklabels(np.arange(0, 21, 2), fontsize=11)
ax11.set_yticks(np.arange(0, 1.01, 0.1))
ax11.set_yticklabels(['0.0', '', '0.2', '', '0.4', '', '0.6', '', '0.8', '', '1.0'], fontsize=11)
ax11.set_ylim(-0.02, 1.02)
ax11.set_xlim(-0.2, 10)
ax11.set_title('Affordability Ratios', fontsize=11, fontweight='bold')

print("Computing household ECDFs (Affordability Ratios)...")
# We recompute for AR - or reuse if you cached qfn earlier. Here we reuse compute_plot_parquet_ecdf for AR.
compute_plot_parquet_ecdf(ax11, filepath_clim + "df_hh_all_long.parquet", "AR", "lightskyblue", label="All Climate Sims")
compute_plot_parquet_ecdf(ax11, filepath + "df_hh_modcool_NoInf_long.parquet", "AR", "gray", label="Baseline")
compute_plot_parquet_ecdf(ax11, filepath_clim + "df_hh_modcool_long.parquet", "AR", "teal", label="Moderate Climate \nwith Adaptation")
compute_plot_parquet_ecdf(ax11, filepath_clim + "df_hh_dryhot_long.parquet", "AR", "salmon", label="Dry Climate \nwith Adaptation")

ax11.plot([2.5, 2.5], [-0.05, 1.05], color='k', linestyle='--')
ax11.text(1.95, 0.04, 'EPA threshold', fontstyle='italic', fontsize=10, rotation=90)
legend = ax11.legend(loc='lower right', fontsize=8.5, bbox_to_anchor=(1.03, -0.035))
legend.set_frame_on(False)

# add a-d labels
sz = 18
ax11.text(-13.4, 2.56, 'a', fontsize=sz, fontweight='bold')
ax11.text(-0.2, 2.56, 'b', fontsize=sz, fontweight='bold')
ax11.text(-13.4, 1.05, 'c', fontsize=sz, fontweight='bold')
ax11.text(-0.2, 1.05, 'd', fontsize=sz, fontweight='bold')

# save and close
plt.savefig('../outputs/Figure2_CDFs_lowmem.png', dpi=300, bbox_inches='tight')
plt.close('all')
print("Saved figure to ../outputs/Figure2_CDFs_lowmem.png")

print_memory_usage("Final: ")
