"""Bootstrap CIs (10k resamples of test queries) for the m2s main rows, m2s vs m2f (iLIDs), PRID ablations."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/minhchien/pblcv/experiments")
from common import MFGF_PUBLISHED, bootstrap_summary, paired_bootstrap  # noqa: E402

RUNS = Path(sys.argv[1])
SEEDS = (42, 1234, 2025)


def ranks(sub, tag):
    d = json.loads((RUNS / sub / f"{tag}.json").read_text())
    t = d["modes"]["val_selected"]["test"]["mfgf"]
    order = np.argsort(np.asarray(t["q_vid"], dtype=str), kind="stable")   # align queries across runs
    return np.asarray(t["ranks"])[order], np.asarray(t["q_vid"], dtype=str)[order]


def stack(sub, tags):
    rs = [ranks(sub, t) for t in tags]
    for r in rs[1:]:
        assert (r[1] == rs[0][1]).all(), "query sets differ"
    return np.stack([r[0] for r in rs])


out = {"mfgf_published": MFGF_PUBLISHED}
print("MFGF_PUBLISHED", MFGF_PUBLISHED)
for sub in ("TVPReid-PRID", "TVPReid-iLIDs", "TVPReid-Duke"):
    R = stack(sub, [f"m2s_seed{s}" for s in SEEDS])
    s = bootstrap_summary(R)
    out[sub] = {"m2s": s}
    print(f"\n{sub}  n_queries={s['_n_queries']} seeds={s['_n_seeds']}")
    for m in ("R@1", "R@5", "R@10", "R@50", "MdR"):
        x = s[m]
        print(f"  {m:5s} {x['mean']:6.2f} ± {x['std']:5.2f} (std seeds)  95% CI [{x['ci_low']:6.2f}, {x['ci_high']:6.2f}]")
    if sub == "TVPReid-iLIDs":
        F = stack(sub, [f"m2f_seed{s}" for s in SEEDS])
        out[sub]["m2f"] = bootstrap_summary(F)
        out[sub]["m2s_minus_m2f"] = {m: paired_bootstrap(R, F, m) for m in ("R@1", "R@5", "R@10")}
        for m, p in out[sub]["m2s_minus_m2f"].items():
            print(f"  m2s-m2f {m}: {p['diff']:+.2f} CI [{p['ci_low']:+.2f}, {p['ci_high']:+.2f}] p={p['p_two_sided']:.3f}")
    if sub == "TVPReid-PRID":
        full = stack(sub, ["m2s_seed42"])
        out[sub]["ablation_seed42"] = {}
        for ab in ("no_init", "no_temporal", "no_dsca", "no_mim"):
            A = stack(sub, [f"m2s_abl_{ab}_seed42"])
            out[sub]["ablation_seed42"][ab] = {m: paired_bootstrap(full, A, m) for m in ("R@1", "R@5", "R@10")}
            for m, p in out[sub]["ablation_seed42"][ab].items():
                print(f"  full-{ab} {m}: {p['diff']:+.2f} CI [{p['ci_low']:+.2f}, {p['ci_high']:+.2f}] p={p['p_two_sided']:.3f}")

Path(sys.argv[2]).write_text(json.dumps(out, indent=2, default=float))
print("\nsaved", sys.argv[2])
