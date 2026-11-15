"""
WiSE-FT (weight-space ensembling, Wortsman et al., CVPR 2022) of a joint model and a model fine-tuned from it:
    theta(a) = (1 - a) * theta_joint + a * theta_finetuned
No training. Each interpolated model is written to checkpoints/<sub>/<tag>/best.pth, encoded and evaluated with
the usual run_revision pipeline (fusion weights and hubness beta chosen on val, test scored once). The endpoints
a = 0 (joint model) and a = 1 (fine-tuned model) are the existing runs. The final a of each subset is chosen on
the validation split (mean of R@1, R@5, R@10), never on test.

Pairs (joint run -> fine-tuned run):
  v2:   all/v2_joint_seed<N>  -> <sub>/v2_seed<N>     tags v2wise<a*100>_seed<N>
  v2j2: all/v2_joint2_seed<N> -> <sub>/v2j2_seed<N>   tags v2j2wise<a*100>_seed<N>

Example:
  python experiments/wise_ft.py --subs TVPReid-PRID TVPReid-iLIDs --data_root /path/to/TVPReid_dataset
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import run_revision as rr  # noqa: E402
from common import SUBSETS, load_json, save_json  # noqa: E402

PAIRS = {"v2": ("v2_joint_seed{s}", "v2_seed{s}"), "v2j2": ("v2_joint2_seed{s}", "v2j2_seed{s}")}
ALPHAS = (0.25, 0.5, 0.75)


def interpolate(joint_ckpt: Path, ft_ckpt: Path, alpha: float, out: Path) -> None:
    import torch

    a, b = common.load_checkpoint(str(joint_ckpt)), common.load_checkpoint(str(ft_ckpt))
    sa, sb = a["model_state_dict"], b["model_state_dict"]
    if sa.keys() != sb.keys():
        raise RuntimeError(f"{joint_ckpt} and {ft_ckpt} have different parameters")
    state = {k: ((1.0 - alpha) * sa[k].float() + alpha * sb[k].float()).to(sb[k].dtype)
             if sb[k].is_floating_point() else sb[k] for k in sb}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": state, "config": b.get("config", {}), "epoch": b.get("epoch"),
                "wise": {"alpha": alpha, "joint": str(joint_ckpt), "finetuned": str(ft_ckpt)}}, out)


def val_mean(results: Path, sub: str, tag: str):
    f = results / "runs" / sub / f"{tag}.json"
    if not f.exists():
        return None
    m = load_json(f)["modes"]["val_selected"]["val"]["all2"]
    return (m["R@1"] + m["R@5"] + m["R@10"]) / 3.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", nargs="+", default=list(SUBSETS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--pairs", nargs="+", choices=list(PAIRS), default=list(PAIRS))
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--results", default=str(rr.RESULTS))
    ap.add_argument("--ckpt_root", default=str(rr.CKPT_ROOT))
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    rr.RESULTS, rr.CKPT_ROOT = Path(args.results), Path(args.ckpt_root)
    ns = SimpleNamespace(force=False, reuse_legacy=False, dry_run=args.dry_run, data_root=args.data_root,
                         batch_size=args.batch_size, num_workers=args.num_workers, val_only=False,
                         grid_step=0.05, hub=True)
    rows = []
    for sub in args.subs:
        for seed in args.seeds:
            for pair in args.pairs:
                joint_tag, ft_tag = (t.format(s=seed) for t in PAIRS[pair])
                joint_ckpt = rr.CKPT_ROOT / "all" / joint_tag / "best.pth"
                ft_ckpt = rr.CKPT_ROOT / sub / ft_tag / "best.pth"
                if not (joint_ckpt.exists() and ft_ckpt.exists()):
                    print(f"[!] {sub}/{pair} seed {seed}: missing {joint_ckpt if not joint_ckpt.exists() else ft_ckpt}")
                    continue
                cands = [(0.0, joint_tag), (1.0, ft_tag)]
                for a in ALPHAS:
                    tag = f"{pair}wise{int(round(a * 100))}_seed{seed}"
                    run = SimpleNamespace(sub=sub, name="full", seed=seed, tag=tag, flags=[f"--wise {a}"], recipe=pair)
                    best = rr.CKPT_ROOT / sub / tag / "best.pth"
                    emb = rr.RESULTS / "emb" / sub / tag
                    encoded = all((emb / f"{s}.npz").exists() for s in ("val", "test", "bank"))
                    if not encoded and not best.exists() and not args.dry_run:
                        interpolate(joint_ckpt, ft_ckpt, a, best)
                        print(f"[+] {sub}/{tag}: alpha={a} ({joint_tag} -> {ft_tag})")
                    if rr.encode_run(ns, run):
                        rr.stage_eval(ns, [run])
                        best.unlink(missing_ok=True)           # embeddings are kept; the weights are cheap to rebuild
                    cands.append((a, tag))
                scored = [(val_mean(rr.RESULTS, sub, t), a, t) for a, t in cands]
                scored = [x for x in scored if x[0] is not None]
                if not scored:
                    continue
                v, a_sel, t_sel = max(scored, key=lambda x: (x[0], -abs(x[1] - 1.0)))   # ties -> closer to fine-tuned
                test = load_json(rr.RESULTS / "runs" / sub / f"{t_sel}.json")["modes"]["val_selected"]["test"]["mfgf"]["metrics"]
                out = {"sub": sub, "seed": seed, "pair": pair, "selected_alpha": a_sel, "selected_tag": t_sel,
                       "val_by_alpha": {str(a): v for v, a, _ in sorted(scored, key=lambda x: x[1])},
                       "test_selected": test}
                save_json(out, rr.RESULTS / "wise" / sub / f"{pair}_seed{seed}.json")
                curve = "  ".join(f"a={a:.2f}:{v:.2f}" for v, a, _ in sorted(scored, key=lambda x: x[1]))
                rows.append(f"{sub:14s} {pair:5s} chosen a={a_sel:.2f}  TEST R@1 {test['R@1']:6.2f}  R@5 {test['R@5']:6.2f}  "
                            f"R@10 {test['R@10']:6.2f}  MdR {test['MdR']:4.1f}  | VAL {v:6.2f}  [{curve}]")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
