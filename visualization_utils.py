import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ATTACK_TYPES = [
    'brute_force', 'port_scan', 'sql_injection', 'ddos',
    'malware', 'ip_blocked', 'unauthorized_access', 'intrusion'
]

COLOR_MAP = {
    'brute_force': '#4C78A8',
    'port_scan': '#F58518',
    'sql_injection': '#54A24B',
    'ddos': '#E45756',
    'malware': '#B279A2',
    'ip_blocked': '#FF9DA6',
    'unauthorized_access': '#9D755D',
    'intrusion': '#72B7B2',
}

def generate_attack_distribution_chart(df, next_day_attack=None, next_day_label="Next day", save_path=None, show=True):
    cols = [c for c in ATTACK_TYPES if c in df.columns]
    dates = df['date'].dt.strftime('%Y-%m-%d').tolist()

    counts = df[cols].fillna(0).clip(lower=0)
    row_sum = counts.sum(axis=1).replace(0, 1)
    pct = counts.div(row_sum, axis=0)

    if next_day_attack is not None and next_day_attack in cols:
        dates = dates + [next_day_label]
        extra = {c: 0.0 for c in cols}
        extra[next_day_attack] = 1.0
        pct = pct._append(extra, ignore_index=True)

    n = len(dates)

    fig_h = max(9, min(40, n * 0.28))
    fig, ax = plt.subplots(figsize=(18, fig_h))

    bar_height = 0.8

    left = np.zeros(n)
    for c in cols:
        vals = pct[c].values
        ax.barh(dates, vals, left=left, height=bar_height,
                label=c, color=COLOR_MAP.get(c))
        left += vals

    for i in range(n):
        cum = 0.0
        for c in cols:
            v = float(pct.iloc[i][c])
            if v > 0.01:
                x = cum + v / 2
                ax.text(x, i, f"{int(round(v*100))}%", va='center', ha='center',
                        fontsize=7, color='white')
            cum += v

    dominant = pct[cols].idxmax(axis=1).tolist()
    if   n <= 60: right_fs = 8
    elif n <= 90: right_fs = 7
    else:         right_fs = 5 
    for i, label in enumerate(dominant):
        ax.text(1.002, i, label,
                va='center', ha='left', fontsize=right_fs,
                transform=ax.get_yaxis_transform()) 

    ax.set_title("Attack distribution per day (stacked %)", fontsize=14)
    ax.set_xlabel("")
    ax.set_xlim(0, 1.0)
    ax.get_xaxis().set_visible(False)

    if n > 90:
        yticks = np.arange(n)
        ax.set_yticks(yticks)
        shown_labels = []
        for i, d in enumerate(dates):
            if i == n - 1 and dates[i] == next_day_label:
                shown_labels.append(d)  
            elif i % 2 == 0:
                shown_labels.append(d) 
            else:
                shown_labels.append("")  
        ax.set_yticklabels(shown_labels)
        ax.tick_params(axis='y', labelsize=6, pad=2)
    else:
        ax.tick_params(axis='y', labelsize=7, pad=2)

    y0, y1 = ax.get_ylim()
    ax.add_patch(Rectangle((0, y0), 1.0, y1 - y0, fill=False, linewidth=1.1,
                           transform=ax.get_xaxis_transform()))

    handles, labels = ax.get_legend_handles_labels()
    ncols = min(len(labels), 8)
    fig.legend(
        handles, labels,
        loc='lower center',
        ncol=ncols,
        frameon=False,
        title="Attack type",
        fontsize=9,
        bbox_to_anchor=(0.5, 0.02), 
        borderaxespad=0.0
    )

    fig.tight_layout(rect=[0.08, 0.12, 0.98, 0.95])

    fig.canvas.draw()                  
    pos = ax.get_position()            
    left_pad = pos.x0                  
    fig.subplots_adjust(
        left=left_pad,
        right=1.0 - left_pad,          
        bottom=0.12,
        top=0.95
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)