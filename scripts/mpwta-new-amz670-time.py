
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker






raw = """
WTA_time\tWTA_acc\tMP_time\tMP_acc
867	0.00717181	348	0.00989801
2702	0.0702014	799	0.0268502
4719	0.187252	1257	0.0763468
6461	0.223274	1726	0.129681
8255	0.252916	2206	0.171653
9969	0.270469	2728	0.203086
11772	0.2892	3220	0.22836
13350	0.2882	3799	0.248137
14801	0.293495	4311	0.261271
15935	0.299536	4805	0.271215
17288	0.304406	5289	0.279518
18626	0.306538	5756	0.28667
20018	0.305485	6266	0.292867
21385	0.305688	6819	0.295862
22825	0.307917	7340	0.297882
24224	0.30876	7797	0.301268
25635	0.308617	8260	0.301968
27026	0.309741	8727	0.305191
28595	0.317887	9169	0.307342
29695	0.311101	9640	0.308466
31066	0.313677	10142	0.308669
32214	0.312291	10685	0.308721
33519	0.312467	11153	0.310715
34815	0.311885	11652	0.311107
36272	0.313088	12131	0.312931
37406	0.314121	12542	0.312049
38597	0.312095	12951	0.313736
39852	0.311454	13389	0.314429
41393	0.3152	13820	0.313487
42541	0.313036	14264	0.313271
"""
df = pd.read_csv(io.StringIO(raw), sep="\t")





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





wta = df[["WTA_time", "WTA_acc"]].dropna().sort_values("WTA_time")
mp  = df[["MP_time",  "MP_acc"]].dropna().sort_values("MP_time")





max_time = float(max(wta["WTA_time"].max(), mp["MP_time"].max()))
n = int(np.floor(np.log10(max_time))) if max_time > 0 else 0
scale = 10.0 ** n

wta_x = wta["WTA_time"] / scale
mp_x  = mp["MP_time"]  / scale


ax.ticklabel_format(axis="x", style="plain", useOffset=False)
ax.xaxis.get_offset_text().set_visible(False)





ax.plot(wta_x, wta["WTA_acc"], label="WTA 50HT")
ax.plot(mp_x,  mp["MP_acc"],  label="MP-WTA 1HT")





ax.set_xlabel(rf"Time ($10^{n}$ s)")
ax.set_ylabel("Accuracy")
ax.set_title("Wiki-325K")
ax.legend(loc="lower right")
ax.grid(True, axis="y")

plt.tight_layout()
plt.savefig("./amz670_time.pdf", bbox_inches="tight", format="pdf", backend="pdf")
plt.show()
