"""
Training-free post-processing of finished runs: (A) score-level ensemble of several runs and (B) gallery-side
re-ranking. Everything is chosen on the VALIDATION split; the test split is scored once with the chosen setting.

Inputs (written by run_revision.py --stage encode/eval):
  results/emb/<sub>/<tag>/{val,test,bank}.npz    embeddings
  results/runs/<sub>/<tag>.json                   val-selected fusion weights and hubness beta of each run

(A) Ensemble. Each run gives a text->video score matrix (its own val-selected fusion weights and hubness beta).
    Every row is z-scored so that runs are on one scale, then the matrices of a subset of runs are averaged.
    All subsets of up to --max_models runs are tried; the subset with the best val mean(R@1, R@5, R@10) wins
    (ties: fewer runs).
(B) Re-ranking, on top of the chosen ensemble, using gallery-gallery similarity of the videos (V_video of the
    ensemble runs, concatenated):
    - DBA (database-side augmentation): S'(q, g) = (1 - lam) S(q, g) + lam * mean_{g' in kNN_k(g)} S(q, g')
    - QE (query expansion):             S''(q, :) = S'(q, :) + alpha * mean_{g in top_m(q)} Gz(g, :)
    lam = 0 / alpha = 0 (no re-ranking) are part of the grids, so the val score never drops.

Example:
  python experiments/ensemble_rerank.py --subs TVPReid-PRID TVPReid-iLIDs TVPReid-Duke
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import REPO, SUBSETS, load_json, load_npz, metrics_from_ranks, ranks_for, save_json  # noqa: E402

DEFAULT_TAGS = ["v2_seed42", "v2j2_seed42", "v2_joint_seed42", "v2_joint2_seed42", "v2_joint3_seed42", "v2s_seed42"]
DBA_K = (1, 2, 3, 5, 10)
DBA_LAM = (0.0, 0.1, 0.2, 0.3, 0.5)
QE_M = (1, 2, 3)
QE_ALPHA = (0.0, 0.1, 0.2, 0.3, 0.5)


def zrows(S):
    return (S - S.mean(axis=1, keepdims=True)) / (S.std(axis=1, keepdims=True) + 1e-8)


def score(S, E, q_idx):
    """Mean of R@1/5/10 (the checkpoint-selection criterion) plus the metrics, video-level match."""
    m = metrics_from_ranks(ranks_for(S, E["q_vid"][q_idx], E["g_vid"]))
    return (m["R@1"] + m["R@5"] + m["R@10"]) / 3.0, m


def run_matrices(results: Path, sub: str, tag: str, split: str):
    """Text->video score matrix of one run (all queries x gallery), aligned on sorted gallery / query keys."""
    emb = results / "emb" / sub / tag
    E = load_npz(emb / f"{split}.npz")
    run = load_json(results / "runs" / sub / f"{tag}.json")
    w = tuple(run["modes"]["val_selected"]["weights"])
    beta = float(run.get("hub_beta", 0.0) or 0.0)
    if beta and (emb / "bank.npz").exists():
        E = common.with_hubness(E, common.hub_vectors(E, load_npz(emb / "bank.npz")), beta)
    S = common.fused_similarity(E, w)
    g_order = np.argsort(E["g_vid"], kind="stable")
    q_order = np.lexsort((E["q_cap"], E["q_vid"]))
    E = {"q_vid": E["q_vid"][q_order], "q_cap": E["q_cap"][q_order], "g_vid": E["g_vid"][g_order],
         "V_video": E["V_video"][g_order]}
    return zrows(S[q_order][:, g_order]), E, {"weights": list(w), "hub_beta": beta}


def gallery_sim(Es):
    V = np.concatenate([E["V_video"] for E in Es], axis=1)
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
    G = V @ V.T
    np.fill_diagonal(G, -np.inf)                       # neighbours exclude the video itself
    return G


def dba(S, G, k, lam):
    if lam == 0.0:
        return S
    nn = np.argsort(-G, axis=1)[:, :k]                 # [G, k]
    return (1.0 - lam) * S + lam * S[:, nn].mean(axis=2)


def qe(S, G, m, alpha):
    if alpha == 0.0:
        return S
    Gz = zrows(np.where(np.isfinite(G), G, 0.0))
    top = np.argsort(-S, axis=1)[:, :m]               # [Q, m]
    return S + alpha * Gz[top].mean(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", nargs="+", default=list(SUBSETS))
    ap.add_argument("--tags", nargs="+", default=DEFAULT_TAGS)
    ap.add_argument("--max_models", type=int, default=3)
    ap.add_argument("--results", default=str(REPO / "results"))
    args = ap.parse_args()
    results = Path(args.results)
    rows = []
    for sub in args.subs:
        tags = [t for t in args.tags if (results / "emb" / sub / t / "val.npz").exists()
                and (results / "emb" / sub / t / "test.npz").exists() and (results / "runs" / sub / f"{t}.json").exists()]
        if not tags:
            print(f"[!] {sub}: no evaluated runs found")
            continue
        val = {t: run_matrices(results, sub, t, "val") for t in tags}
        test = {t: run_matrices(results, sub, t, "test") for t in tags}
        Ev, Et = val[tags[0]][1], test[tags[0]][1]
        for t in tags:                                  # every run must share the same queries and gallery
            assert (val[t][1]["g_vid"] == Ev["g_vid"]).all() and (val[t][1]["q_vid"] == Ev["q_vid"]).all(), t
            assert (test[t][1]["g_vid"] == Et["g_vid"]).all() and (test[t][1]["q_vid"] == Et["q_vid"]).all(), t
        qv_all = np.arange(len(Ev["q_vid"]))            # val: all captions (same as checkpoint selection)
        qt_mfgf = np.where(Et["q_cap"] == 0)[0]          # test: MFGF protocol (first caption per video)

        # (A) ensemble subset, chosen on val
        cands = []
        for n in range(1, min(args.max_models, len(tags)) + 1):
            for combo in itertools.combinations(tags, n):
                S = np.mean([val[t][0] for t in combo], axis=0)
                s, _ = score(S, Ev, qv_all)
                cands.append((s, -n, combo))
        cands.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best_val, _, combo = cands[0]
        single = max((c for c in cands if len(c[2]) == 1), key=lambda x: x[0])

        # (B) re-ranking on top of the chosen ensemble, chosen on val
        Sv = np.mean([val[t][0] for t in combo], axis=0)
        St = np.mean([test[t][0] for t in combo], axis=0)
        Gv, Gt = gallery_sim([val[t][1] for t in combo]), gallery_sim([test[t][1] for t in combo])
        grid = []
        for k, lam in itertools.product(DBA_K, DBA_LAM):
            s, _ = score(dba(Sv, Gv, k, lam), Ev, qv_all)
            grid.append((s, -lam, -k, k, lam))
        _, _, _, k_sel, lam_sel = max(grid)
        Sv1 = dba(Sv, Gv, k_sel, lam_sel)
        grid_qe = []
        for m, a in itertools.product(QE_M, QE_ALPHA):
            s, _ = score(qe(Sv1, Gv, m, a), Ev, qv_all)
            grid_qe.append((s, -a, -m, m, a))
        _, _, _, m_sel, a_sel = max(grid_qe)

        def test_metrics(S):
            return score(S[qt_mfgf], {"q_vid": Et["q_vid"][qt_mfgf], "g_vid": Et["g_vid"]},
                         np.arange(len(qt_mfgf)))[1]

        out = {"sub": sub, "candidates": tags, "n_test_videos": int(len(Et["g_vid"])),
               "selection": "val mean(R@1,R@5,R@10), all captions; test = MFGF protocol, scored once",
               "runs": {t: val[t][2] for t in tags},
               "single_best": {"tags": list(single[2]), "val": single[0],
                               "test": test_metrics(test[single[2][0]][0])},
               "ensemble": {"tags": list(combo), "val": best_val, "test": test_metrics(St)},
               "rerank": {"tags": list(combo), "dba_k": k_sel, "dba_lambda": lam_sel, "qe_m": m_sel, "qe_alpha": a_sel,
                          "val": score(qe(Sv1, Gv, m_sel, a_sel), Ev, qv_all)[0],
                          "test": test_metrics(qe(dba(St, Gt, k_sel, lam_sel), Gt, m_sel, a_sel))},
               "val_top10_ensembles": [{"tags": list(c), "val": s} for s, _, c in cands[:10]]}
        save_json(out, results / "ensemble" / f"{sub}.json")
        for name in ("single_best", "ensemble", "rerank"):
            m = out[name]["test"]
            rows.append(f"{sub:14s} {name:11s} R@1 {m['R@1']:6.2f}  R@5 {m['R@5']:6.2f}  R@10 {m['R@10']:6.2f}  "
                        f"MdR {m['MdR']:4.1f}  | VAL {out[name]['val']:6.2f}  {'+'.join(out[name]['tags'])}")
        print(f"[+] {sub}: ensemble {'+'.join(combo)} | DBA k={k_sel} lam={lam_sel} | QE m={m_sel} alpha={a_sel}")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
