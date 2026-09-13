





































































































import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker






raw = """
WTA_time\tWTA_acc\tMP_time\tMP_acc
6322	0.107983	2048	0.0751557
14814	0.0995455	4519	0.0434986
21572	0.113344	7048	0.110808
29553	0.156379	9514	0.187239
38112	0.185476	11925	0.256324
46673	0.181391	14262	0.259042
53247	0.212723	16529	0.307342
59540	0.212839	18801	0.330783
68042	0.263528	21067	0.350847
76609	0.274334	23286	0.353268
84953	0.281303	25499	0.35659
92813	0.277362	27676	0.379673
99679	0.322536	29791	0.405419
107717	0.342423	31958	0.402182
116233	0.361106	34089	0.435388
124860	0.365227	36140	0.432447
132407	0.373082	38195	0.428752
141707	0.386657	40333	0.43351
151480	0.405265	42409	0.432296
160580	0.400903	44473	0.437936
171443	0.422839	46565	0.44113
175495	0.422396	48614	0.448967
179957	0.408458	50650	0.446924
185329	0.409538	52674	0.451968
190410	0.420037	54718	0.451498
195890	0.427652	56817	0.445246
201534	0.43621	58855	0.44229
207391	0.436614	60868	0.449887
212902	0.437422	62921	0.44787
218262	0.427265	64905	0.449638
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
plt.savefig("./wiki325k_time.pdf", bbox_inches="tight", format="pdf", backend="pdf")
plt.show()
