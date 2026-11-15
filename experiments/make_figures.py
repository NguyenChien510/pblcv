"""
Regenerates the result figures of the revised paper from results/summary.json.

  fig1_test_r1.png       Figure 1: Rank@1 on the test split, protocol of [24]; baselines from [24],
                         SpaceTime-DSCA = mean over seeds with 95% bootstrap CI
  fig3_delta_mfgf.png    Figure 3: SpaceTime-DSCA - MFGF for every Rank@K (test split), CI of SpaceTime-DSCA
  fig5_ablation.png      Figure 5: cumulative ablation on PRID (test), gain of every step with paired 95% CI

Sizes keep the aspect ratio of the images in the .docx, so experiments/fill_paper.py can swap them in place.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MFGF_PUBLISHED, SUBSETS, load_json  # noqa: E402

BASELINES = {  # Table 1 of [24], Rank@1 on the test split
    "MMT": {"TVPReid-PRID": 6.8, "TVPReid-iLIDs": 13.3, "TVPReid-Duke": 11.8},
    "X-Pool": {"TVPReid-PRID": 21.1, "TVPReid-iLIDs": 14.7, "TVPReid-Duke": 25.2},
    "Frozen-in-Time": {"TVPReid-PRID": 20.9, "TVPReid-iLIDs": 17.5, "TVPReid-Duke": 24.8},
}
COLORS = {"MMT": "#d4d8de", "X-Pool": "#adb3be", "Frozen-in-Time": "#868e99", "MFGF": "#e8923a", "ours": "#1f3a93"}
SUB_COLORS = {"TVPReid-PRID": "#1f3a93", "TVPReid-iLIDs": "#e07b20", "TVPReid-Duke": "#2e8b57"}
DPI = 172
SIZES = {"fig1": (5.81, 3.924), "fig3": (5.81, 3.576), "fig5": (5.81, 3.924)}   # inches, same aspect as the .docx


def vn(x, d=1):
    s = f"{abs(x):.{d}f}".replace(".", ",")
    return ("−" if x < 0 and round(abs(x), d) != 0 else "") + s


def vns(x, d=1):
    r = round(x, d)
    return ("+" if r > 0 else "") + vn(x, d) if r != 0 else vn(0.0, d)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#3a4250")
    ax.spines["bottom"].set_color("#3a4250")
    ax.tick_params(colors="#3a4250", labelsize=10)
    ax.yaxis.grid(True, color="#e3e6ea", linewidth=0.8)
    ax.set_axisbelow(True)


def fig1(summary, out):
    main = summary.get("main", {})
    subs = [s for s in SUBSETS if s in main]
    if not subs:
        return False
    fig, ax = plt.subplots(figsize=SIZES["fig1"], dpi=DPI)
    names = ["MMT", "X-Pool", "Frozen-in-Time", "MFGF", "ours"]
    width, x = 0.16, np.arange(len(subs))
    for i, nme in enumerate(names):
        pos = x + (i - 2) * width
        if nme == "ours":
            r1 = [main[s]["mfgf_protocol"]["R@1"] for s in subs]
            vals = np.array([r["mean"] for r in r1])
            err = np.array([[v - r["ci_low"] for v, r in zip(vals, r1)], [r["ci_high"] - v for v, r in zip(vals, r1)]])
            ax.bar(pos, vals, width * 0.92, color=COLORS[nme], label="SpaceTime-DSCA (TB 3 seed, KTC 95%)",
                   yerr=err, capsize=2.5, error_kw={"elinewidth": 1, "ecolor": "#111"})
            for p, v, s, e in zip(pos, vals, subs, err[1]):
                d = v - MFGF_PUBLISHED[s]["R@1"]
                ax.text(p, v + e + 1.0, vn(v), ha="center", va="bottom", fontsize=9, color=COLORS[nme], fontweight="bold")
                ax.text(p, v + e + 5.0, vns(d), ha="center", va="bottom", fontsize=9,
                        color="#1b8a3a" if d >= 0 else "#c0392b", fontweight="bold")
        else:
            vals = [MFGF_PUBLISHED[s]["R@1"] if nme == "MFGF" else BASELINES[nme][s] for s in subs]
            ax.bar(pos, vals, width * 0.92, color=COLORS[nme], label=nme)
            if nme == "MFGF":
                for p, v in zip(pos, vals):
                    ax.text(p, v + 0.8, vn(v), ha="center", va="bottom", fontsize=9, color=COLORS[nme], fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(subs)
    ax.set_ylabel("Rank@1 trên tập test (%)", fontsize=10, color="#3a4250")
    top = max(main[s]["mfgf_protocol"]["R@1"]["ci_high"] for s in subs)
    ax.set_ylim(0, max(55, top + 12))
    style(ax)
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.22), handlelength=1.2,
              columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return True


def fig3(summary, out):
    main = summary.get("main", {})
    subs = [s for s in SUBSETS if s in main]
    if not subs:
        return False
    metrics = ["R@1", "R@5", "R@10", "R@50"]
    fig, ax = plt.subplots(figsize=SIZES["fig3"], dpi=DPI)
    width, x = 0.25, np.arange(len(metrics))
    for i, s in enumerate(subs):
        bs = main[s]["mfgf_protocol"]
        d = np.array([bs[m]["mean"] - MFGF_PUBLISHED[s][m] for m in metrics])
        lo = np.array([bs[m]["ci_low"] - MFGF_PUBLISHED[s][m] for m in metrics])
        hi = np.array([bs[m]["ci_high"] - MFGF_PUBLISHED[s][m] for m in metrics])
        pos = x + (i - (len(subs) - 1) / 2) * width
        ax.bar(pos, d, width * 0.9, color=SUB_COLORS[s], label=s, yerr=[d - lo, hi - d], capsize=2,
               error_kw={"elinewidth": 0.9, "ecolor": "#222"})
        for p, v, h, l in zip(pos, d, hi, lo):
            ax.text(p, (h + 0.6) if v >= 0 else (l - 0.6), vns(v), ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=7.5, color=SUB_COLORS[s], fontweight="bold")
    ax.axhline(0, color="#3a4250", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(["Rank@1", "Rank@5", "Rank@10", "Rank@50"])
    ax.set_ylabel("Chênh lệch so với MFGF\n(điểm phần trăm)", fontsize=10, color="#3a4250")
    style(ax)
    ax.legend(ncol=3, fontsize=8.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return True


def fig5(summary, out):
    rows = summary.get("ablation", {}).get("cumulative", [])
    if len(rows) != 6:
        return False
    labels = ["Baseline", "+ ImageNet\ninit", "+ Divided\nST-Attn", "+ DSCA", "+ MIM", "+ Chấm điểm\nđa tầng"]
    vals = [r["summary"]["R@1"]["mean"] for r in rows]
    fig, ax = plt.subplots(figsize=SIZES["fig5"], dpi=DPI)
    x = np.arange(len(rows))
    ax.bar(x[0], vals[0], 0.62, color="#adb3be")
    ax.text(x[0], vals[0] + 1.0, vn(vals[0], 2), ha="center", va="bottom", fontsize=9, fontweight="bold", color="#1f2430")
    for i in range(1, len(rows)):
        d = rows[i]["delta_prev"]
        base, step = vals[i - 1], vals[i] - vals[i - 1]
        ax.bar(x[i], min(base, vals[i]), 0.62, color="#e6e9ee")
        ax.bar(x[i], abs(step), 0.62, bottom=min(base, vals[i]), color="#4f7bd0" if step >= 0 else "#d0574f")
        ax.errorbar(x[i] + 0.22, vals[i], yerr=[[d["diff"] - d["ci_low"]], [d["ci_high"] - d["diff"]]], fmt="none",
                    ecolor="#222", elinewidth=0.9, capsize=2)
        ax.text(x[i], max(base, vals[i]) + 1.0, vn(vals[i], 2), ha="center", va="bottom", fontsize=9,
                fontweight="bold", color="#1f2430")
        ax.text(x[i], min(base, vals[i]) + abs(step) / 2, vns(step, 2), ha="center", va="center", fontsize=8,
                color="white", fontweight="bold")
        ax.plot([x[i - 1] + 0.31, x[i] - 0.31], [base, base], ls=":", color="#7a828f", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Rank@1 trên test TVPReid-PRID (%)", fontsize=9.5, color="#3a4250")
    ax.set_ylim(0, max(50, max(vals) + 8))
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return True


def main(results=None):
    results = Path(results or Path(__file__).resolve().parent.parent / "results")
    summary = load_json(results / "summary.json")
    out = results / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in (("fig1_test_r1.png", fig1), ("fig3_delta_mfgf.png", fig3), ("fig5_ablation.png", fig5)):
        ok = fn(summary, out / name)
        print(f"[{'+' if ok else '!'}] {name}: {'saved' if ok else 'skipped (results missing)'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
