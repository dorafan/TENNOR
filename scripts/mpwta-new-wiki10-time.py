
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker






raw = """
WTA_time\tWTA_acc\tMP_time\tMP_acc
83	0.25622	50	0.160498
167	0.342385	97	0.286559
250	0.425819	145	0.403216
335	0.438107	195	0.409739
420	0.45085	247	0.42415
501	0.454794	298	0.461468
580	0.512136	349	0.520024
657	0.512591	398	0.546268
737	0.530036	448	0.557494
817	0.553095	495	0.6071
894	0.519873	544	0.56432
968	0.538987	597	0.554157
1043	0.515928	641	0.621359
1119	0.533829	691	0.597998
1196	0.537166	738	0.579794
1268	0.562955	785	0.623938
1342	0.530643	831	0.607403
1419	0.562803	879	0.601183
1491	0.549302	928	0.599818
1563	0.578125	978	0.63395
1636	0.6027	1027	0.611044
1709	0.577063	1075	0.619691
1785	0.563258	1124	0.613319
1857	0.59375	1174	0.605431
1929	0.553095	1225	0.63126
2002	0.565989	1277	0.621359
2079	0.583283	1332	0.604369
2149	0.595419	1378	0.625303
2223	0.599818	1423	0.628034
2287	0.580097	1468	0.608161
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
plt.savefig("./wiki31k_time.pdf", bbox_inches="tight", format="pdf", backend="pdf")
plt.show()
