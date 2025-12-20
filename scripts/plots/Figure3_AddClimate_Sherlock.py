#%% Memory-efficient full pipeline: compute boxplot stats from parquet (two-pass histogram) and plot
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import gc
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Patch
import json
import psutil
from datetime import datetime

print("Start low-mem Figure3 pipeline:", datetime.now())

#%% ---------- parameters ----------
filepath = '../../../../../scratch/users/jskerker/AffordPaper1/Figure2/'
filepath_clim = filepath + '../SA/Climate/'

income_csv = '../../../../../scratch/users/jskerker/AffordPaper1/Figure2/resampled_income_data_30Nov2024.csv'
income_cols = ['account', 'map_inc_1']
income_key_col = 'Account'  # after we standardize

parquet_scenarios = [
    (filepath_clim + "df_hh_all_long.parquet",           "all"),
    (filepath + "df_hh_modcool_NoInf_long.parquet",      "baseline/no inf."),
    (filepath_clim + "df_hh_modcool_long.parquet",       "baseline"),
    (filepath_clim + "df_hh_dryhot_long.parquet",        "dry/hot"),
]

# keep in same order used in your plotting (order_list)
order_list = ['all, Low', 'all, Not', 'Spacer1',
              'baseline/no inf., Low', 'baseline/no inf., Not', 'Spacer2',
              'baseline, Low', 'baseline, Not', 'Spacer3',
              'dry/hot, Low', 'dry/hot, Not']

custom_x_positions = [0, 0.8, 1.45, 2.1, 2.9, 3.55, 4.2, 5.0, 5.65, 6.3, 7.1]

palette_dict = {'all, Low': 'salmon', 'all, Not': 'mediumturquoise',
                'baseline/no inf., Low': 'salmon', 'baseline/no inf., Not': 'mediumturquoise',
                'baseline, Low': 'salmon', 'baseline, Not': 'mediumturquoise',
                'dry/hot, Low': 'salmon', 'dry/hot, Not': 'mediumturquoise',
                'Spacer1': 'white', 'Spacer2': 'white', 'Spacer3': 'white'}

# ensure mapping from correct keys
color_for = {}
for k in order_list:
    if 'Low' in k:
        color_for[k] = 'salmon'
    elif 'Not' in k:
        color_for[k] = 'mediumturquoise'
    else:
        color_for[k] = 'white'

# histogram resolution (higher -> more accurate quantiles; increases mem slightly)
BINS = 2000

# metrics we need for boxplots
metrics = ['Demand', 'Bill', 'AR']

# ---------- helper utilities ----------
def print_memory_usage(prefix=""):
    p = psutil.Process()
    print(f"{prefix} Mem: {p.memory_info().rss/1e9:.2f} GB")

def load_income_map(income_csv_path):
    # small file - load once
    df_income = pd.read_csv(income_csv_path, usecols=income_cols)
    df_income = df_income.rename(columns={'account': income_key_col})
    income_map = pd.Series(df_income['map_inc_1'].values, index=df_income[income_key_col]).to_dict()
    return income_map

def classify_income(account, income_map, threshold=40000):
    # returns 'Low' or 'Not'
    try:
        val = income_map.get(account, np.nan)
        if pd.isna(val):
            return 'Not'  # default to Not if missing (matches your previous approach which would create NaN)
        return 'Low' if val < threshold else 'Not'
    except Exception:
        return 'Not'

# Two-pass histogram per scenario per group
def compute_hist_stats_for_parquet(parquet_path, columns, income_map, bins=BINS):
    """
    columns: list of columns to read from parquet. Must include 'Account' and metrics.
    Returns:
      group_stats: dict keyed by category string (e.g., 'baseline, Low') ->
                   dict of metric->(counts array, bin_edges, total_n)
    """
    pqf = pq.ParquetFile(parquet_path)
    # prepare group keys for this scenario will be defined externally
    # We'll compute per-group min/max/count first
    group_info = {}  # group -> metric -> {'min':..., 'max':..., 'count':...}

    # First pass: mins, maxs, counts per metric per group
    for rg_i in range(pqf.num_row_groups):
        rg = pqf.read_row_group(rg_i, columns=columns)
        pdf = rg.to_pandas()  # only small chunk
        # standardize Account column name if needed (some files might call 'account' vs 'Account')
        if 'account' in pdf.columns and 'Account' not in pdf.columns:
            pdf = pdf.rename(columns={'account': 'Account'})

        # map income group
        pdf['is_low_inc'] = pdf['Account'].map(lambda a: 'Low' if income_map.get(a, np.nan) < 40000 else 'Not')

        # For each row, build category name based on scenario label which we'll attach later
        # But here we only need per-scenario per-group stats; caller will know scenario label

        for grp in pdf['is_low_inc'].unique():
            grp_mask = pdf['is_low_inc'] == grp
            for m in metrics:
                if m not in pdf.columns:
                    continue
                arr = pdf.loc[grp_mask, m].dropna()
                if arr.empty:
                    continue
                gkey = grp  # 'Low' or 'Not'
                group_info.setdefault(gkey, {}).setdefault(m, {'min': None, 'max': None, 'count': 0})
                cur = group_info[gkey][m]
                cur['count'] += int(arr.shape[0])
                mmin = arr.min()
                mmax = arr.max()
                if cur['min'] is None or mmin < cur['min']:
                    cur['min'] = mmin
                if cur['max'] is None or mmax > cur['max']:
                    cur['max'] = mmax

        del pdf, rg
        gc.collect()

    # If no data at all:
    if not group_info:
        return {}

    # Now create histograms containers per group & metric
    group_hist = {}  # group -> metric -> {'counts': np.array, 'edges': np.array, 'total': n}
    for gkey, mm in group_info.items():
        group_hist[gkey] = {}
        for m, info in mm.items():
            if info['count'] == 0:
                continue
            gmin = info['min']
            gmax = info['max']
            if gmin == gmax:
                # tiny degenerate bin
                edges = np.array([gmin, gmax + 1.0])  # avoid zero-width
                counts = np.zeros(1, dtype=np.int64)
            else:
                edges = np.linspace(gmin, gmax, bins + 1)
                counts = np.zeros(bins, dtype=np.int64)
            group_hist[gkey][m] = {'counts': counts, 'edges': edges, 'total': info['count']}

    # Second pass: fill histogram counts
    for rg_i in range(pqf.num_row_groups):
        rg = pqf.read_row_group(rg_i, columns=columns)
        pdf = rg.to_pandas()
        if 'account' in pdf.columns and 'Account' not in pdf.columns:
            pdf = pdf.rename(columns={'account': 'Account'})

        pdf['is_low_inc'] = pdf['Account'].map(lambda a: 'Low' if income_map.get(a, np.nan) < 40000 else 'Not')

        for gkey in list(group_hist.keys()):
            for m in list(group_hist[gkey].keys()):
                if m not in pdf.columns:
                    continue
                mask = pdf['is_low_inc'] == gkey
                vals = pdf.loc[mask, m].dropna().values
                if vals.size == 0:
                    continue
                edges = group_hist[gkey][m]['edges']
                # numpy histogram; returns counts per bin
                c, _ = np.histogram(vals, bins=edges)
                group_hist[gkey][m]['counts'] += c
        del pdf, rg
        gc.collect()

    return group_hist

def hist_counts_to_boxstats(counts, edges, total_n):
    """
    Given histogram counts and edges compute Q1, median, Q3 and whisker low/high (1.5*IQR rule),
    returning numeric values.
    """
    if total_n == 0 or counts.sum() == 0:
        return None

    cum = np.cumsum(counts)
    # quantile helper
    def quantile(q):
        target = q * total_n
        idx = np.searchsorted(cum, target)
        if idx <= 0:
            return edges[0]
        if idx >= len(counts):
            return edges[-1]
        # linear interpolation inside bin
        c_before = cum[idx - 1]
        inside = counts[idx]
        if inside == 0:
            return (edges[idx] + edges[idx+1]) / 2.0
        frac = (target - c_before) / inside
        left_edge = edges[idx]
        right_edge = edges[idx+1]
        return left_edge + frac * (right_edge - left_edge)

    q1 = quantile(0.25)
    q2 = quantile(0.5)
    q3 = quantile(0.75)
    q80 = quantile(0.8)
    iqr = q3 - q1
    whisk_low = q1 - 1.5 * iqr
    whisk_high = q3 + 1.5 * iqr

    # convert whiskers to nearest observed within histogram (approx)
    # find smallest bin center >= whisk_low but >= data min
    bin_centers = (edges[:-1] + edges[1:]) / 2.0
    # cumulative counts give positions; find first bin with center >= whisk_low that has non-zero counts
    # to get approximate actual whisker as nearest observed value within 1.5 IQR
    valid_idxs = np.where(counts > 0)[0]
    if valid_idxs.size == 0:
        return q1, q2, q3, q1, q3

    # approximate min within whisker
    low_idx = valid_idxs[0]
    # find last index with center < whisk_low (we want lowest observed >= whisk_low)
    idxs_below = valid_idxs[bin_centers[valid_idxs] < whisk_low]
    if idxs_below.size == 0:
        whisker_low_val = bin_centers[valid_idxs[0]]
    else:
        # whisk low is next observed bin after the last below target (i.e., the first observed bin >= whisk_low)
        next_idxs = valid_idxs[bin_centers[valid_idxs] >= whisk_low]
        whisker_low_val = bin_centers[next_idxs[0]] if next_idxs.size else bin_centers[valid_idxs[0]]

    # approximate high
    idxs_above = valid_idxs[bin_centers[valid_idxs] > whisk_high]
    if idxs_above.size == 0:
        whisker_high_val = bin_centers[valid_idxs[-1]]
    else:
        prev_idxs = valid_idxs[bin_centers[valid_idxs] <= whisk_high]
        whisker_high_val = bin_centers[prev_idxs[-1]] if prev_idxs.size else bin_centers[valid_idxs[-1]]
    print('median: {}, 80th percentile: {}'.format(q2, q80))
    return q1, q2, q3, whisker_low_val, whisker_high_val

#%% ---------- main loop: compute stats across scenarios ----------
print("Loading income mapping...")
income_map = load_income_map(income_csv)
print_memory_usage("After income map:")

# We'll store box stats keyed by category string (e.g., 'baseline, Low') -> metric -> box stats tuple
box_stats = {cat: {} for cat in order_list if 'Spacer' not in cat}  # only real categories

for pq_path, scen_label in parquet_scenarios:
    print(f"\nProcessing scenario: {scen_label} from {pq_path} ...", datetime.now())
    # columns to read - Account + metrics
    # some files may have 'account' lowercase; we'll handle rename in function
    cols = ['Account'] + metrics
    group_hist = compute_hist_stats_for_parquet(pq_path, cols, income_map, bins=BINS)

    # group_hist keys are 'Low' and/or 'Not' (if they exist)
    for gkey in ['Low', 'Not']:
        print(f"\nProcessing group: {gkey}")
        category_name = f"{scen_label}, {gkey}"
        if gkey not in group_hist:
            # mark as missing
            for m in metrics:
                box_stats[category_name][m] = None
            continue
        for m in metrics:
            print('metric: {}'.format(m))
            if m not in group_hist[gkey]:
                box_stats[category_name][m] = None
                continue
            gh = group_hist[gkey][m]
            counts = gh['counts']
            edges = gh['edges']
            total = int(gh['total'])
            # derive box stats
            res = hist_counts_to_boxstats(counts, edges, total)
            box_stats[category_name][m] = res  # tuple (q1,q2,q3,whisk_low,whisk_high) or None

    # free memory between scenarios
    del group_hist
    gc.collect()
    print_memory_usage(f"After scenario {scen_label}:")

#%% ---------- plotting (matplotlib polygon boxes to match seaborn style) ----------
# load rcparams as you did
with open("rcparams.json", "r") as f:
    loaded_rcparams = json.load(f)
loaded_rcparams.pop("backend", None)
mpl.use("Agg")
mpl.rcParams.update(loaded_rcparams)

# plotting helpers
def draw_box_ax(ax, xpos, q1, q2, q3, low, high, color, width=0.35):
    # draw rectangle for IQR
    rect = Rectangle((xpos - width/2, q1), width, q3 - q1, facecolor=color, edgecolor='black', linewidth=1)
    ax.add_patch(rect)
    # median line
    ax.plot([xpos - width/2, xpos-0.01 + width/2], [q2, q2], color='black', linewidth=1.2)
    # whiskers
    ax.plot([xpos, xpos], [q3, high], color='black', linewidth=1)
    ax.plot([xpos, xpos], [low, q1], color='black', linewidth=1)
    # caps
    cap_w = width * 0.3
    ax.plot([xpos - cap_w/2, xpos + cap_w/2], [high, high], color='black', linewidth=1)
    ax.plot([xpos - cap_w/2, xpos + cap_w/2], [low, low], color='black', linewidth=1)

# build figure identical layout
fig = plt.figure(figsize=(13, 3))
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(1, 3, width_ratios=[1,1,1], wspace=0.35)

# subplot 1: Demand
ax00 = fig.add_subplot(gs[0,0])
y_max = 16
ax00.set_xlim(-0.75, 7.75)
ax00.set_ylim(-0.5, y_max)
ax00.set_yticks(np.arange(0, y_max+1, 2))
ax00.set_yticklabels(['0', '', '4', '', '8', '', '12', '', '16'], fontsize=11)
ax00.set_xticks(custom_x_positions)
ax00.set_xticklabels([''] * len(custom_x_positions))
ax00.set_xlabel('Climate Scenarios', fontsize=11, labelpad=10)
ax00.set_ylabel('Demand (ccf/month)', fontsize=11)
ax00.grid(axis="y", linewidth=0.8)
ax00.grid(axis="x", visible=False)
ax00.set_title('Water Demands', fontsize=11, fontweight='bold')

# subplot 2: Bill
ax01 = fig.add_subplot(gs[0,1])
y_max = 450
ax01.set_xlim(-0.75, 7.75)
ax01.set_ylim(0, y_max)
ax01.set_yticks(np.arange(0, y_max+1, 50))
ax01.set_yticklabels(['0', '', '100', '', '200', '', '300', '', '400', ''], fontsize=11)
ax01.set_xticks(custom_x_positions)
ax01.set_xticklabels([''] * len(custom_x_positions))
ax01.set_xlabel('Climate Scenarios', fontsize=11, labelpad=10)
ax01.set_ylabel('Bill ($/month)', fontsize=11)
ax01.grid(axis="y", linewidth=0.8)
ax01.grid(axis="x", visible=False)
ax01.set_title('Water Bills', fontsize=11, fontweight='bold')

# subplot 3: AR
ax02 = fig.add_subplot(gs[0,2])
y_max = 40
ax02.set_xlim(-0.75, 7.75)
ax02.set_ylim(-1, y_max)
ax02.set_yticks(np.arange(0, y_max+1, 2.5))
ax02.set_yticklabels(['0', '', '', '', '10', '', '', '', '20', '', '', '', '30', '', '', '', 40], fontsize=11)
ax02.set_xticks(custom_x_positions)
ax02.set_xticklabels([''] * len(custom_x_positions))
ax02.set_xlabel('Climate Scenarios', fontsize=11, labelpad=10)
ax02.set_ylabel('AR (% of bill / income)', fontsize=11)
ax02.grid(axis="y", linewidth=0.8)
ax02.grid(axis="x", visible=False)
ax02.set_title('Affordability Ratios', fontsize=11, fontweight='bold')
ax02.plot([-0.75, 7.75], [2.5, 2.5], color='navy', linestyle=':', linewidth=1.2)

# iterate positions and draw boxes for each category in order_list
for xpos, cat in zip(custom_x_positions, order_list):
    if 'Spacer' in cat:
        # leave blank / spacer
        continue
    stats_for_cat = box_stats.get(cat)
    if stats_for_cat is None:
        continue
    # Demand
    ds = stats_for_cat.get('Demand')
    if ds is not None:
        q1,q2,q3,low,high = ds
        draw_box_ax(ax00, xpos, q1,q2,q3, low, high, color_for[cat], width=0.6)
    # Bill
    bs = stats_for_cat.get('Bill')
    if bs is not None:
        q1,q2,q3,low,high = bs
        draw_box_ax(ax01, xpos, q1,q2,q3, low, high, color_for[cat], width=0.6)
    # AR
    ars = stats_for_cat.get('AR')
    if ars is not None:
        q1,q2,q3,low,high = ars
        draw_box_ax(ax02, xpos, q1,q2,q3, low, high, color_for[cat], width=0.6)

# legend
legend_handles = [Patch(color='salmon', label='Low Income'), Patch(color='mediumturquoise', label='All Others')]
legend = ax00.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.02, 1.02), fontsize=9, handletextpad=0.5)
legend.set_frame_on(False)

# add text labels (exact placements preserved from your original)
level = -1.6
sz = 8.5
ax00.text(0.2, level, 'All', fontsize=sz, fontstyle='italic')
ax00.text(1.9, level, 'Baseline', fontsize=sz, fontstyle='italic')
ax00.text(3.85, level, ' Moderate', fontsize=sz, fontstyle='italic')
ax00.text(6.5, level, 'Dry', fontsize=sz, fontstyle='italic')
ax00.text(11.6, level, 'All', fontsize=sz, fontstyle='italic')
ax00.text(13.3, level, 'Baseline', fontsize=sz, fontstyle='italic')
ax00.text(15.25, level, ' Moderate', fontsize=sz, fontstyle='italic')
ax00.text(17.9, level, 'Dry', fontsize=sz, fontstyle='italic')
ax00.text(23.15, level, 'All', fontsize=sz, fontstyle='italic')
ax00.text(24.85, level, 'Baseline', fontsize=sz, fontstyle='italic')
ax00.text(26.75, level, ' Moderate', fontsize=sz, fontstyle='italic')
ax00.text(29.35, level, 'Dry', fontsize=sz, fontstyle='italic')

# add a-c labels
ax02.text(-23.6,  37.2, 'a', fontweight='bold', fontsize=16)
ax02.text(-12.1, 36.2, 'b', fontweight='bold', fontsize=16)
ax02.text(-0.6, 37.2, 'c', fontweight='bold', fontsize=16)

# save
outpath = '../outputs/Figure3-Boxplots_09Dec2025_lowmem.png'
plt.savefig(outpath, dpi=300, bbox_inches='tight')
plt.close('all')
print("Saved figure to:", outpath)
print_memory_usage("Final:")
