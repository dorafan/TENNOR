
import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import io



raw = """
Epoch Classification Acc	WTA 50HT	MP-WTA 1HT
0	0.107983	0.0751557
1	0.0995455	0.0434986
2	0.113344	0.110808
3	0.156379	0.187239
4	0.185476	0.256324
5	0.181391	0.259042
6	0.212723	0.307342
7	0.212839	0.330783
8	0.263528	0.350847
9	0.274334	0.353268
10	0.281303	0.35659
11	0.277362	0.379673
12	0.322536	0.405419
13	0.342423	0.402182
14	0.361106	0.435388
15	0.365227	0.432447
"""

df = pd.read_csv(io.StringIO(raw), sep="\t")




df = df.rename(columns={"Epoch Classification Acc": "Epoch"})





pts_per_inch = 72.27
text_width_pt = 505.89
fig_width_in = text_width_pt / pts_per_inch
fig_height_in = fig_width_in * 0.6

















plt.rcParams.update({









    "font.size": 22,
    "axes.titlesize": 26,
    "axes.labelsize": 26,
    "xtick.labelsize": 18,
    "ytick.labelsize": 20,
    "legend.fontsize": 24,
    "axes.titlepad": 8,
    "pdf.fonttype": 42,
    "axes.linewidth": 1.2
})




fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))

for col in df.columns[1:]:
    ax.plot(df["Epoch"], df[col], label=col)

ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.set_title("Wiki-325K")
ax.legend(loc="lower right")
ax.grid(True, axis="y")


plt.tight_layout()

plt.savefig("./wiki325k.pdf", bbox_inches="tight", format="pdf", backend="pdf")














