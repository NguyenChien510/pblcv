"""
Builds results/tokens.json for experiments/fill_paper.py from the m2s runs (MFGF-style encoder, 3 seeds) and the
PRID leave-one-out ablation (seed 42). Every number is read from results/v6/runs/*.json (test scored once, every
choice made on validation); nothing is typed by hand.

    python experiments/paper_tokens_m2s.py results/v6/runs results/tokens.json
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MFGF_PUBLISHED, bootstrap_summary, metrics_from_ranks, paired_bootstrap  # noqa: E402

RUNS = Path(sys.argv[1])
SEEDS = (42, 1234, 2025)
SUBS = {"PRID": "TVPReid-PRID", "ILIDS": "TVPReid-iLIDs", "DUKE": "TVPReid-Duke"}
MINUS = "−"


def vn(x, d=1):
    return f"{x:.{d}f}".replace(".", ",").replace("-", MINUS)


def signed(x, d=1):
    r = round(x, d)
    return vn(0.0, d) if r == 0 else ("+" if r > 0 else MINUS) + vn(abs(x), d)


def ci(lo, hi, d=1):
    return f"[{vn(lo, d)}; {vn(hi, d)}]"


def load(sub, tag):
    return json.loads((RUNS / sub / f"{tag}.json").read_text())


def ranks(d, mode="val_selected", proto="mfgf"):
    t = d["modes"][mode]["test"][proto]
    order = np.argsort(np.asarray(t["q_vid"], dtype=str), kind="stable")
    return np.asarray(t["ranks"])[order]


tok = {}
for key, sub in SUBS.items():
    runs = [load(sub, f"m2s_seed{s}") for s in SEEDS]
    R = np.stack([ranks(d) for d in runs])
    b = bootstrap_summary(R)
    for m, k in (("R@1", "R1"), ("R@5", "R5"), ("R@10", "R10"), ("R@50", "R50"), ("MdR", "MDR")):
        d = 2 if m == "MdR" else 1
        tok[f"TEST_{key}_{k}"] = vn(b[m]["mean"], d)
        tok[f"TESTSD_{key}_{k}"] = "± " + vn(b[m]["std"], d)
    r1 = b["R@1"]
    tok[f"TESTCI_{key}_R1"] = ci(r1["ci_low"], r1["ci_high"])
    tok[f"TESTCIW_{key}_R1"] = vn(r1["ci_high"] - r1["ci_low"])
    tok[f"TESTDELTA_{key}_R1"] = signed(r1["mean"] - MFGF_PUBLISHED[sub]["R@1"])
    tok[f"TESTMSD_{key}_R1"] = f"{vn(r1['mean'])} ± {vn(r1['std'])}"
    tok[f"TESTSDN_{key}_R1"] = vn(r1["std"])
    for s, x in zip(SEEDS, r1["per_seed"]):
        tok[f"SEED_{key}_{s}"] = vn(x)
    cap2 = [metrics_from_ranks(ranks(d, proto="mfgf_cap2"))["R@1"] for d in runs]
    all2 = [d["modes"]["val_selected"]["test"]["all2"]["metrics"]["R@1"] for d in runs]
    val = [d["modes"]["val_selected"]["val"]["all2"]["R@1"] for d in runs]
    tok[f"CAP2_{key}_R1"] = vn(np.mean(cap2))
    tok[f"ALL2_{key}_R1"] = f"{vn(np.mean(all2))} ± {vn(np.std(all2, ddof=1))}"
    tok[f"VALSEL_{key}_R1"] = vn(np.mean(val))
    tok[f"_cap2diff_{key}"] = abs(np.mean(cap2) - r1["mean"])
    tok[f"_all2diff_{key}"] = abs(np.mean(all2) - r1["mean"])
    tok[f"_mean_{key}"] = r1["mean"]
    tok[f"_lo_{key}"], tok[f"_hi_{key}"] = r1["ci_low"], r1["ci_high"]
    # Table 7: weights chosen on validation (mean over seeds) and test R@1 with chosen / default / joint-only weights
    W = np.mean([d["modes"]["val_selected"]["weights"] for d in runs], axis=0)
    tok[f"W_{key}"] = "(" + "; ".join(vn(w, 2) for w in W) + ")"
    tok[f"W_{key}_VAL_R1"] = vn(np.mean(val))
    tok[f"W_{key}_TEST_SEL"] = vn(r1["mean"])
    for mode, k in (("default", "DEF"), ("global", "GLB")):
        tok[f"W_{key}_TEST_{k}"] = vn(np.mean([d["modes"][mode]["test"]["mfgf"]["metrics"]["R@1"] for d in runs]))

tok["CAP_MAXDIFF"] = vn(max(tok.pop(f"_cap2diff_{k}") for k in SUBS))
tok["ALL2_MAXDIFF"] = vn(max(tok.pop(f"_all2diff_{k}") for k in SUBS))
means = {k: tok.pop(f"_mean_{k}") for k in SUBS}
bounds = {k: (tok.pop(f"_lo_{k}"), tok.pop(f"_hi_{k}")) for k in SUBS}
tok["TEST_MIN_R1"], tok["TEST_MAX_R1"] = vn(min(means.values())), vn(max(means.values()))

# Leave-one-out ablation on PRID, seed 42 (the multi-level row uses the joint-only weights of the full run)
full_d = load("TVPReid-PRID", "m2s_seed42")
full = ranks(full_d)[None]
rows = {"full": full_d}
for name in ("no_init", "no_temporal", "no_dsca", "no_mim"):
    rows[name] = load("TVPReid-PRID", f"m2s_abl_{name}_seed42")
for name, d in rows.items():
    m = d["modes"]["val_selected"]["test"]["mfgf"]["metrics"]
    tok[f"ABLL_{name}_R1"], tok[f"ABLL_{name}_R5"], tok[f"ABLL_{name}_MDR"] = vn(m["R@1"]), vn(m["R@5"]), vn(m["MdR"])
    if name != "full":
        p = paired_bootstrap(ranks(d)[None], full, "R@1")
        tok[f"ABLL_{name}_D"] = f"{signed(p['diff'])} {ci(p['ci_low'], p['ci_high'])}"
g = full_d["modes"]["global"]["test"]["mfgf"]["metrics"]
tok["ABLL_no_multilevel_R1"], tok["ABLL_no_multilevel_R5"], tok["ABLL_no_multilevel_MDR"] = vn(g["R@1"]), vn(g["R@5"]), vn(g["MdR"])
p = paired_bootstrap(ranks(full_d, "global")[None], full, "R@1")
tok["ABLL_no_multilevel_D"] = f"{signed(p['diff'])} {ci(p['ci_low'], p['ci_high'])}"
tok["ABL_NSEED"] = "1"

Path(sys.argv[2]).write_text(json.dumps(tok, indent=1, ensure_ascii=False), encoding="utf-8")
for k in sorted(tok):
    print(f"{k:24s} {tok[k]}")
print({k: (round(means[k], 2), round(bounds[k][0], 2), round(bounds[k][1], 2)) for k in SUBS})
