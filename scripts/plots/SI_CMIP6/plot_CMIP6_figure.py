import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.legend import Legend


# Set working directory - fill in
#wdir =
#os.chdir(wdir)

# read plot data
cmip6_temp = pd.read_csv('cmip6_temp.csv')
cmip6_prec = pd.read_csv('cmip6_prec.csv')
cal_adapt_temp = pd.read_csv('cal_adapt_temp.csv')
cal_adapt_prec = pd.read_csv('cal_adapt_prec.csv')

# define plot colors and shapes
colors = ['#76120b', '#02007f']
shapes = ['o', '^']

# Define a new figure
fig = plt.figure()
ax = fig.add_subplot() # add subplot

# plot points
ax.scatter(x=cmip6_prec['pc_1'], y=cmip6_temp['delta_1'], s=50, c=colors[0], marker=shapes[0], alpha=0.5, linewidths=1, edgecolors=colors[0], zorder=4)
ax.scatter(x=cmip6_prec['pc_2'], y=cmip6_temp['delta_2'], s=50, c=colors[1], marker=shapes[0], alpha=0.5, linewidths=1, edgecolors=colors[1], zorder=4)
ax.scatter(x=cal_adapt_prec['pc_1'], y=cal_adapt_temp['delta_1'], s=50, c=colors[0], marker=shapes[1], alpha=0.5, linewidths=1, edgecolors=colors[0], zorder=4)
ax.scatter(x=cal_adapt_prec['pc_2'], y=cal_adapt_temp['delta_2'], s=50, c=colors[1], marker=shapes[1], alpha=0.5, linewidths=1, edgecolors=colors[1], zorder=4)

# Plot formatting
ax.set_xlabel('changes in annual precipitation (%)', fontsize=12)
ax.set_ylabel('changes in average temperature (C)', fontsize=12)
ax.set_title('Climate Anomaly', fontsize=14)
ax.set_xlim([-80, 80])
ax.set_xticks(np.arange(-80, 81, 20))
ax.set_ylim([0, 8])
ax.set_yticks(np.arange(0, 9, 1))
ax.grid(zorder=0)
ax.grid(visible=True, which="major", color='#D5D8DC', linestyle="-")
ax.grid(visible=False, which="minor")

# add custom legends
legend_1 = [Line2D([0], [0], linewidth=0, color='w', marker=shapes[0], markersize=7, markeredgecolor='k', markeredgewidth=1, label='CMIP6 GCMs'),
            Line2D([0], [0], linewidth=0, color='w', marker=shapes[1], markersize=7, markeredgecolor='k', markeredgewidth=1, label='Cal-Adapt GCMs')]
ax.legend(handles=legend_1, loc='lower center', bbox_to_anchor=(0.5, -0.2), fontsize=12, ncols=2)
legend_2 = Legend(ax, [Patch(facecolor=colors[0], edgecolor=colors[0], alpha=0.5, label='2026-2055'),
                               Patch(facecolor=colors[1], edgecolor=colors[1], alpha=0.5, label='2056-2085')],
            ['2026-2055', '2056-2085'], loc='upper right', fontsize=12)
ax.add_artist(legend_2)

# Save figure
figure = plt.gcf()
figure.set_size_inches(8, 6)
name = "climate_anomaly_figure.png"
plt.savefig(name, dpi=600, bbox_inches='tight')
plt.close(fig)
