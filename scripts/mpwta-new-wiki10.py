
import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import io



raw = """
Epoch Classification Acc	WTA 50HT	MP-WTA 1HT
0	0.331875	0.178125
1	0.465625	0.246875
2	0.455	0.33125
3	0.536875	0.35125
4	0.60625	0.38125
5	0.589375	0.485625
6	0.635	0.499375
7	0.640625	0.5475
8	0.625	0.559375
9	0.64	0.605
10	0.5625	0.614375
11	0.63625	0.58625
12	0.66375	0.588125
13	0.65375	0.624375
14	0.6275	0.605
15	0.653125	0.633125
16	0.675	0.631875
17	0.65625	0.61875
18	0.6775	0.610625
19	0.670625	0.6275
20	0.65	0.620625
21	0.645625	0.656875
22	0.663125	0.62375
23	0.66375	0.645625
24	0.650625	0.6425
25	0.6525	0.61375
26	0.673125	0.61375
27	0.645625	0.63125
28	0.65625	0.643125
29	0.641839	0.624242
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
ax.set_title("Wiki10-31k")
ax.legend(loc="lower right")
ax.grid(True, axis="y")


plt.tight_layout()

plt.savefig("./wiki31.pdf", bbox_inches="tight", format="pdf", backend="pdf")














