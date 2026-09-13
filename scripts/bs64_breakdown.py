import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


rows = [











    ("Wiki10", 8,  5.85,  813.4375,   6.45, 1632.875,    2.25, 3.25, 1655.825),
    ("Wiki10", 22, 2.65,  387.25,     2.85, 779.9,       2.15, 1.45, 795.95),
    ("Wiki10", 44, 1.95,  273.0625,   1.8,  551.275,     2.2,  1.0,  565.425),

    ("Amz-Sub", 8,  30.4, 5783.5,     13.6, 11413.93333, 5.666666667, 2.8, 11496.86667),
    ("Amz-Sub", 22, 12.8, 2643.1875,  5.7,  5122.725,    5.55,        2.1, 5185.575),
    ("Amz-Sub", 44, 8.65, 1799.5625,  3.5,  3457.775,    5.55,        1.4, 3514.375),

]
data = pd.DataFrame(rows, columns=[
    "Dataset","Threads","LS_sec","OHT_sec","Requester_sec","Fetcher_sec","Updater_sec","Refresh_sec","Total"
])

datasets = ["Wiki10", "Amz-Sub"]
threads_order = [8, 22, 44]


data["Other_sec"] = data["Total"] - (data["Requester_sec"] + data["Fetcher_sec"] + data["Updater_sec"])
data["Other_sec"] = data["Other_sec"].clip(lower=0)

data["RequesterTrans_sec"] = data["Fetcher_sec"] - data["OHT_sec"] - data["LS_sec"]
data["RequesterTrans_sec"] = data["RequesterTrans_sec"].clip(lower=0)


time_cols = ["LS_sec", "OHT_sec", "Requester_sec", "Fetcher_sec", "Updater_sec", "Total", "Other_sec", "RequesterTrans_sec"]
data[time_cols] = data[time_cols] / 100.0

STYLE_MAP = {
    "Requester": "#1f77b4",
    "Fetcher": "#ff7f0e",
    "Updater": "#2ca02c",
    "Other": "#9467bd",
    "OHT": "#8c564b",
    "LSH scan": "#e377c2",
    "ReqTrans": "#bcbd22"
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
                      ("Fetcher", "Fetcher_sec"),
                      ("Updater", "Updater_sec")]

        denom_col = "Total"
        legend_ncol = 2
        ymax = float(np.nanmax(df[denom_col].to_numpy())) * 1.06
    elif mode == "fetcher":
        components = [("OHT", "OHT_sec"),
                      ("LSH scan", "LS_sec"),
                      ("ReqTrans", "RequesterTrans_sec")]
        denom_col = "Fetcher_sec"
        legend_ncol = 2
        ymax = float(np.nanmax(df[denom_col].to_numpy())) * 1.06
    else:
        raise ValueError("mode must be 'total' or 'fetcher'")

    comp_labels = [c[0] for c in components]
    comp_cols = [c[1] for c in components]





    fig, axes = plt.subplots(1, 2, figsize=(5.5, 4.0), sharey=False)

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

pdf_total, eps_total = make_v6_from_df(data, "total", "./bs64_all_datasets_total_stacked_v6")
pdf_fetcher, eps_fetcher = make_v6_from_df(data, "fetcher", "./bs64_all_datasets_fetcher_stacked_v6")

(pdf_total, eps_total, pdf_fetcher, eps_fetcher)

