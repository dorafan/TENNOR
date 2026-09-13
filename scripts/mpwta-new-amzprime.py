
import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import io



raw = """
Epoch Classification Acc	WTA 50HT	MP-WTA 1HT
0	0.00333807	0.00362216
1	0.0224432	0.0261364
2	0.0784801	0.0917614
3	0.119531	0.123082
4	0.151847	0.148722
5	0.164631	0.165128
6	0.179474	0.169957
7	0.174361	0.176207
8	0.178622	0.176847
9	0.181534	0.178835
10	0.182102	0.180753
11	0.186577	0.184446
12	0.186648	0.185582
13	0.186719	0.18267
14	0.189418	0.186222
15	0.189205	0.186364
16	0.188849	0.186932
17	0.189134	0.187287
18	0.190341	0.18821
19	0.189276	0.188139
20	0.189347	0.187784
21	0.188991	0.190696
22	0.19233	0.187997
23	0.190767	0.193466
24	0.191264	0.189347
25	0.19304	0.194531
26	0.190767	0.190909
27	0.191193	0.192543
28	0.191974	0.192756
29	0.191477	0.192756
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
ax.set_title("Amazon-Prime")
ax.legend(loc="lower right")
ax.grid(True, axis="y")


plt.tight_layout()

plt.savefig("./amazonprime.pdf", bbox_inches="tight", format="pdf", backend="pdf")














