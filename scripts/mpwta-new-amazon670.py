
import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import io



raw = """
Epoch Classification Acc	WTA 50HT	MP-WTA 1HT
0	0.00717181	0.00989801
1	0.0702014	0.0268502
2	0.187252	0.0763468
3	0.223274	0.129681
4	0.252916	0.171653
5	0.270469	0.203086
6	0.2892	0.22836
7	0.2882	0.248137
8	0.293495	0.261271
9	0.299536	0.271215
10	0.304406	0.279518
11	0.306538	0.28667
12	0.305485	0.292867
13	0.305688	0.295862
14	0.307917	0.297882
15	0.30876	0.301268
16	0.308617	0.301968
17	0.309741	0.305191
18	0.317887	0.307342
19	0.311101	0.308466
20	0.313677	0.308669
21	0.312291	0.308721
22	0.312467	0.310715
23	0.311885	0.311107
24	0.313088	0.312931
25	0.314121	0.312049
26	0.312095	0.313736
27	0.311454	0.314429
28	0.3152	0.313487
29	0.313036	0.313271
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
ax.set_title("Amazon-670K")
ax.legend(loc="lower right")
ax.grid(True, axis="y")


plt.tight_layout()

plt.savefig("./amazon670k.pdf", bbox_inches="tight", format="pdf", backend="pdf")














