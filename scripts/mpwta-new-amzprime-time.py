
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker






raw = """
WTA_time\tWTA_acc\tMP_time\tMP_acc
66	0.00333807	24	0.00362216
139	0.0224432	53	0.0261364
218	0.0784801	81	0.0917614
291	0.119531	82	0.123082
382	0.151847	139	0.148722
473	0.164631	167	0.165128
565	0.179474	197	0.169957
637	0.174361	227	0.176207
719	0.178622	255	0.176847
807	0.181534	284	0.178835
897	0.182102	311	0.180753
992	0.186577	341	0.184446
1084	0.186648	370	0.185582
1177	0.186719	399	0.18267
1253	0.189418	428	0.186222
1337	0.189205	457	0.186364
1415	0.188849	486	0.186932
1491	0.189134	514	0.187287
1878	0.190341	541	0.18821
1649	0.189276	569	0.188139
1750	0.189347	597	0.187784
1853	0.188991	624	0.190696
1952	0.19233	652	0.187997
2032	0.190767	680	0.193466
2121	0.191264	707	0.189347
2207	0.19304	734	0.194531
2305	0.190767	763	0.190909
2388	0.191193	789	0.192543
2475	0.191974	818	0.192756
2575	0.191477	843	0.192756
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
plt.savefig("./amzprime_time.pdf", bbox_inches="tight", format="pdf", backend="pdf")
plt.show()
