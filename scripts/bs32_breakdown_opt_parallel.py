import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


rows = [














    ("Wiki10", 8,  3.15,        337.65625,    2.325,        669.8125,     1.85,        1.475,   681.3125,  10.85),
    ("Wiki10", 22, 1.5,         168.53125,    1.075,        330.5875,     1.85,        0.7,     338.8125,  12.5),
    ("Wiki10", 44, 1.45,        122.9375,     0.725,        240.6,        1.925,       4.5,     248.275,   12.55),

    ("Amz-Sub", 8,  11.125,     1047.21875,   4.875,        2059.0625,    3.55,        2.175,   2085.4875, 60.875),
    ("Amz-Sub", 22, 4.05,       513.53125,    2.0,          988.7875,     3.575,       0.975,   1007.7875, 67.95),
    ("Amz-Sub", 44, 2.85,       351.84375,    1.45,         676.8375,     3.55,        0.75,    695.7125,  69.15),

    ("Wiki325",   8,  88.93333333, 2324.166667,  11.46666667, 4604.633333,  8.7,        26.7,     4717.833333, 437.0667),
    ("Wiki325",   22, 34.4,        1127.583333, 4.566666667, 2191.166667,  8.566666667, 12.2,     2255.4, 491.6),
    ("Wiki325",   44, 20.43333333, 800.3333333, 2.666666667, 1527.533333,  8.766666667, 8,        1578.3, 505.5667),

]
data = pd.DataFrame(rows, columns=[
    "Dataset","Threads","LS_sec","OHT_sec","Requester_sec","Fetcher_sec","Updater_sec","Refresh_sec","Total", "Slowdown_sec"
])

datasets = ["Wiki10", "Amz-Sub", "Wiki325"]
threads_order = [8, 22, 44]


data["Other_sec"] = data["Total"] - (data["Requester_sec"] + data["Fetcher_sec"] + data["Updater_sec"])
data["Other_sec"] = data["Other_sec"].clip(lower=0)

data["RequesterTrans_sec"] = data["Fetcher_sec"] - data["OHT_sec"] - data["LS_sec"]
data["RequesterTrans_sec"] = data["RequesterTrans_sec"].clip(lower=0)


time_cols = ["LS_sec", "OHT_sec", "Requester_sec", "Fetcher_sec", "Updater_sec", "Total", "Other_sec", "RequesterTrans_sec", "Slowdown_sec"]
data[time_cols] = data[time_cols] / 100.0

STYLE_MAP = {
    "Requester": "#1f77b4",
    "Fetcher": "#ff7f0e",
    "Updater": "#2ca02c",
    "Other": "#9467bd",
    "OHT": "#8c564b",
    "LSH scan": "#e377c2",
    "ReqTrans": "#bcbd22",
    "Slowdown": "#d62728"
}


plt.rcParams.update({









    "font.size": 16,
    "axes.titlesize": 24,
    "axes.labelsize": 24,
    "xtick.labelsize": 24,
    "ytick.labelsize": 16,
    "legend.fontsize": 24,
    "axes.titlepad": 6,
    "pdf.fonttype": 42,
    "axes.linewidth": 1.2
})

def make_v6_from_df(df: pd.DataFrame, mode: str, out_base: str):
    if mode == "total":
        components = [("Requester", "Requester_sec"),
                      ("Slowdown", "Slowdown_sec"),
                      ("Fetcher", "Fetcher_sec"),
                      ("Updater", "Updater_sec")]

        denom_col = "Total"
        legend_ncol = 2
        ymax = float(np.nanmax(df[denom_col].to_numpy())) * 1.3
    elif mode == "fetcher":
        components = [("OHT", "OHT_sec"),
                      ("Slowdown", "Slowdown_sec"),
                      ("LSH scan", "LS_sec"),
                      ("ReqTrans", "RequesterTrans_sec")]
        denom_col = "Fetcher_sec"
        legend_ncol = 2
        ymax = float(np.nanmax(df[denom_col].to_numpy())) * 1.3
    else:
        raise ValueError("mode must be 'total' or 'fetcher'")

    comp_labels = [c[0] for c in components]
    comp_cols = [c[1] for c in components]





    fig, axes = plt.subplots(1, 3, figsize=(5.5, 4.0), sharey=False)

    x = np.array([0.0, 0.65, 1.30])
    bar_width = 0.36
    legend_handles = None

    for ax, ds_name in zip(axes, datasets):
        sub = df[df["Dataset"] == ds_name].copy().set_index("Threads").reindex(threads_order)

        bottoms = np.zeros(len(threads_order), dtype=float)
        local_handles = []
        for lab, col in zip(comp_labels, comp_cols):
            vals = sub[col].to_numpy(dtype=float)



            style = STYLE_MAP.get(lab, {"color": "gray"})

            b = ax.bar(x, vals, bottom=bottoms, width=bar_width, label=lab, color=style)

            bottoms += vals
            local_handles.append(b[0])

        ax.set_title(ds_name, pad=10.0)
        ax.set_xlim(-0.33, 1.63)
        ax.set_xticks(x)
        ax.set_xticklabels([str(t) for t in threads_order])
        ax.set_xlabel("Threads")

        ax.set_ylim(0, ymax)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=8))


        ax.set_axisbelow(True)
        ax.grid(axis="y", which="major", linewidth=0.6, alpha=0.5)
        if ax is axes[0]:
            ax.set_ylabel("Time ($10^2$ s)")

        if legend_handles is None:
            legend_handles = local_handles

    fig.legend(
        legend_handles, comp_labels,
        loc="upper center",
        ncol=legend_ncol,
        bbox_to_anchor=(0.5, 1.12),
        frameon=False,
        handlelength=1.0,
        handleheight=0.7,
        handletextpad=0.5,
        columnspacing=0.9,
        labelspacing=0.25,
        borderaxespad=0.2,
    )

    fig.subplots_adjust(top=0.75, bottom=0.22, left=0.12, right=0.99, wspace=0.28)
    pdf_path = f"{out_base}.pdf"

    eps_path = f"{out_base}.eps"
    fig.savefig(pdf_path, bbox_inches="tight")

    fig.savefig(eps_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return pdf_path, eps_path






pdf_total = make_v6_from_df(data, "total", "./opt_parallel_bs32_all_datasets_total_stacked_v6")
pdf_fetcher = make_v6_from_df(data, "fetcher", "./opt_parallel_bs32_all_datasets_fetcher_stacked_v6")

(pdf_total, pdf_fetcher)

