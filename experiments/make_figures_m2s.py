"""
Figures of the revised paper for the m2s runs (MFGF-style encoder, 3 seeds). Same pixel aspect as the images in the
template, so they can be swapped in place.

    python experiments/make_figures_m2s.py results/v6 results/bootstrap_ci.json results/figures
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

V6, CI, OUT = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text()), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
SUBS = ("TVPReid-PRID", "TVPReid-iLIDs", "TVPReid-Duke")
SHORT = {"TVPReid-PRID": "PRID", "TVPReid-iLIDs": "iLIDs", "TVPReid-Duke": "Duke"}
SUB_COLORS = {"TVPReid-PRID": "#1f3a93", "TVPReid-iLIDs": "#e07b20", "TVPReid-Duke": "#2e8b57"}
BASE = {  # Table 1 of [24], test split
    "MMT": {"TVPReid-PRID": 6.8, "TVPReid-iLIDs": 13.3, "TVPReid-Duke": 11.8},
    "X-Pool": {"TVPReid-PRID": 21.1, "TVPReid-iLIDs": 14.7, "TVPReid-Duke": 25.2},
    "Frozen-in-Time": {"TVPReid-PRID": 20.9, "TVPReid-iLIDs": 17.5, "TVPReid-Duke": 24.8},
    "MFGF": {s: CI["mfgf_published"][s]["R@1"] for s in SUBS},
}
COLORS = {"MMT": "#d4d8de", "X-Pool": "#adb3be", "Frozen-in-Time": "#868e99", "MFGF": "#e8923a", "SpaceTime-DSCA": "#1f3a93"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#444",
                     "axes.grid": True, "grid.color": "#e3e6ea", "axes.axisbelow": True, "axes.grid.axis": "y"})
vn = lambda x, d=1: f"{x:.{d}f}".replace(".", ",").replace("-", "−")


def save(fig, name, w_px, h_px):
    fig.set_size_inches(w_px / 200, h_px / 200)
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)


# Figure 1: Rank@1 on the test split
fig, ax = plt.subplots()
names = list(BASE) + ["SpaceTime-DSCA"]
x = np.arange(len(SUBS))
bw = 0.16
for i, n in enumerate(names):
    if n == "SpaceTime-DSCA":
        vals = [CI[s]["m2s"]["R@1"]["mean"] for s in SUBS]
        err = [[v - CI[s]["m2s"]["R@1"]["ci_low"] for v, s in zip(vals, SUBS)],
               [CI[s]["m2s"]["R@1"]["ci_high"] - v for v, s in zip(vals, SUBS)]]
        ax.bar(x + (i - 2) * bw, vals, bw, color=COLORS[n], label=n, yerr=err, capsize=2, error_kw={"lw": 0.8})
        for xi, v, s in zip(x, vals, SUBS):
            d = v - BASE["MFGF"][s]
            ax.text(xi + (i - 2) * bw, CI[s]["m2s"]["R@1"]["ci_high"] + 1.2, ("+" if d > 0 else "") + vn(d),
                    ha="center", fontsize=7, color="#2e8b57" if d > 0 else "#555", fontweight="bold")
    else:
        ax.bar(x + (i - 2) * bw, [BASE[n][s] for s in SUBS], bw, color=COLORS[n], label=n)
ax.set_xticks(x, [s for s in SUBS])
ax.set_ylabel("Rank@1 (%)")
ax.set_ylim(0, 62)
ax.legend(ncol=3, fontsize=7, frameon=False, loc="upper left")
fig.tight_layout()
save(fig, "fig1_test_r1.png", 999, 674)

# Figure 3: SpaceTime-DSCA - MFGF for every Rank@K, CI of SpaceTime-DSCA
fig, ax = plt.subplots()
ks = ("R@1", "R@5", "R@10")
for j, s in enumerate(SUBS):
    m = CI[s]["m2s"]
    d = [m[k]["mean"] - CI["mfgf_published"][s][k] for k in ks]
    lo = [m[k]["mean"] - m[k]["ci_low"] for k in ks]
    hi = [m[k]["ci_high"] - m[k]["mean"] for k in ks]
    xs = np.arange(len(ks)) + (j - 1) * 0.25
    ax.bar(xs, d, 0.25, color=SUB_COLORS[s], label=s, yerr=[lo, hi], capsize=2, error_kw={"lw": 0.8})
ax.axhline(0, color="#444", lw=0.8)
ax.set_xticks(np.arange(len(ks)), [k.replace("R@", "Rank@") for k in ks])
ax.set_ylabel("Chênh lệch so với MFGF (điểm %)")
ax.legend(ncol=3, fontsize=7, frameon=False, loc="upper right")
fig.tight_layout()
save(fig, "fig3_delta_mfgf.png", 999, 615)

# Figure 4: training dynamics on validation (seed 42); circle = epoch kept as checkpoint (mean of R@1/5/10)
fig, (a, b) = plt.subplots(2, 1, sharex=True)
for s in SUBS:
    h = json.loads((V6 / "hist" / "checkpoints" / s / "m2s_seed42" / "train_history.json").read_text())
    ep, r1, mdr = h["eval_epochs"], h["rank1"], h["mdr"]
    sel = int(np.argmax([(p + q + r) / 3 for p, q, r in zip(h["rank1"], h["rank5"], h["rank10"])]))
    a.plot(ep, r1, "-o", ms=2.5, lw=1.4, color=SUB_COLORS[s], label=s)
    a.plot(ep[sel], r1[sel], "o", ms=7, mfc="white", mec=SUB_COLORS[s], mew=1.5)
    a.annotate(f"{vn(r1[sel], 2)}% (ep. {ep[sel]})", (ep[sel], r1[sel]),
               xytext={"TVPReid-PRID": (0, 9), "TVPReid-iLIDs": (18, -16), "TVPReid-Duke": (-30, 10)}[s], textcoords="offset points",
               ha="center", fontsize=7, color=SUB_COLORS[s], fontweight="bold")
    b.plot(ep, mdr, "-o", ms=2.5, lw=1.4, color=SUB_COLORS[s])
a.set_ylabel("Rank@1 (%)")
a.set_ylim(0, 80)
a.legend(ncol=3, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18))
a.text(0.99, 0.05, "(a)", transform=a.transAxes, ha="right", fontweight="bold")
b.set_yscale("log", base=2)
b.set_yticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
b.set_ylabel("Median Rank (thang log)")
b.set_xlabel("Epoch")
b.text(0.99, 0.88, "(b)", transform=b.transAxes, ha="right", fontweight="bold")
fig.tight_layout()
save(fig, "fig4_dynamics.png", 999, 1065)

# Figure 6: cosine T_full x V_video on PRID validation (seed 42), first caption of the first six videos
with np.load(V6 / "emb" / "TVPReid-PRID" / "m2s_seed42" / "val.npz") as z:
    E = {k: z[k] for k in z.files}
g = E["g_vid"][:6].astype(str)
qi = [int(np.where(E["q_vid"].astype(str) == v)[0][0]) for v in g]
gi = list(range(6))
S = E["T_full"][qi] @ E["V_video"][gi].T
fig, ax = plt.subplots()
im = ax.imshow(S, cmap="RdBu_r", vmin=-1, vmax=1)
for i in range(6):
    for j in range(6):
        ax.text(j, i, vn(S[i, j], 2), ha="center", va="center", fontsize=7, color="white" if abs(S[i, j]) > 0.55 else "#222")
    ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec="black", lw=1.5))
ax.set_xticks(range(6), [f"V{i + 1}" for i in range(6)])
ax.set_yticks(range(6), [f"T{i + 1}" for i in range(6)])
ax.grid(False)
fig.colorbar(im, fraction=0.046, pad=0.04, label="cos")
fig.tight_layout()
save(fig, "fig6_similarity.png", 999, 825)
diag = np.diag(S)
off = S[~np.eye(6, dtype=bool)]
print(f"fig6 diag {diag.min():.2f}-{diag.max():.2f}; off-diag max {off.max():.2f}; off<0.3 {np.mean(off < 0.3):.2f}; neg {np.mean(off < 0):.2f}")


# Figure 2 (architecture) is drawn by experiments/paper/fig_architecture.py
print("saved", sorted(p.name for p in OUT.iterdir()))
