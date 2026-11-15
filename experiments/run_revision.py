"""
One entry point for every experiment requested by review DG25 (submission 2).

Stages (run them in order; each one skips work that is already done, so the script can be re-run after a
Colab disconnect):
  split      check the official train/val/test split: sizes, person IDs shared between splits  -> results/split_check.json
  train      train.py for every run of experiments/plan.py (3 seeds x 3 subsets + PRID ablations); every finished
             run is encoded at once and slimmed (latest.pth deleted, optimizer state stripped; ablation
             checkpoints deleted) so that 27 runs fit on Google Drive. --no_slim keeps everything.
  encode     encode the val and test splits once per run (embeddings -> results/emb/<sub>/<tag>/{val,test}.npz)
  eval       choose fusion weights on VAL, then score TEST once per run       -> results/runs/<sub>/<tag>.json
  aggregate  mean +- std over seeds, bootstrap CIs, ablation deltas, paper tokens -> results/summary.json, tables.md
  figures    figures 1, 3 and 5 of the paper                                    -> results/figures/*.png
  bench      GPU memory / latency of SpaceTimeViT vs ViT+S3D                     -> results/bench.json
  mt         Vietnamese queries through machine translation (EN->VI->EN)         -> results/mt/<sub>/<tag>.json
  all        split, train, encode, eval, aggregate, figures

Examples:
  python experiments/run_revision.py --stage split --data_root /content/dataset
  python experiments/run_revision.py --stage train --subs TVPReid-PRID --data_root /content/dataset
  python experiments/run_revision.py --stage all --data_root /content/dataset
  python experiments/run_revision.py --stage train --dry_run          # print the train.py commands only
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import (DEFAULT_WEIGHTS, REPO, SUBSETS, load_json, load_npz, metrics_from_ranks, protocol_ranks,  # noqa: E402
                    save_json)
from plan import ABLATION_RUNS, ABLATION_SUB, CUHK_ARGS, CUHK_SUB, JOINT_SUB, RECIPES, SEEDS, Run, build_runs  # noqa: E402

RESULTS = REPO / "results"
CKPT_ROOT = REPO / "checkpoints"      # train.py always writes to <repo>/checkpoints/<sub>/<tag>/
GLOBAL_WEIGHTS = (1.0, 0.0, 0.0)
HUB_BETAS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)


def tag_dir(run) -> Path:
    return CKPT_ROOT / run.sub / run.tag


def legacy_checkpoint(run) -> Path:
    """The seed-42 runs of the previous submission were saved directly in checkpoints/<sub>/best.pth."""
    return CKPT_ROOT / run.sub / "best.pth"


def resolve_checkpoint(run, reuse_legacy: bool):
    if run.name.startswith("joint"):              # --joint_eval: the joint model scored on one subset
        best = CKPT_ROOT / JOINT_SUB / run.tag / "best.pth"
        return best if best.exists() else None
    best = tag_dir(run) / "best.pth"
    if best.exists():
        return best
    if reuse_legacy and run.name == "full" and run.seed == 42 and legacy_checkpoint(run).exists():
        return legacy_checkpoint(run)
    return None


def finished_epochs(run) -> int:
    hist = tag_dir(run) / "train_history.json"
    if not hist.exists():
        return 0
    try:
        return int(max(load_json(hist).get("epochs", [0]) or [0]))
    except Exception:
        return 0


# ----------------------------------------------------------------------------------------------
def stage_split(args):
    cmd = [sys.executable, str(REPO / "experiments" / "check_split.py"), "--out", str(RESULTS / "split_check.json")]
    if args.data_root:
        cmd += ["--data_root", args.data_root]
    print("$", " ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, cwd=REPO, check=True)


def check_train_script() -> None:
    """An old train.py silently ignores --seed/--tag and writes to checkpoints/<sub>/best.pth, overwriting the
    seed-42 checkpoint of the previous submission. Refuse to train with such a script."""
    src = (REPO / "train.py").read_text(encoding="utf-8")
    missing = [f for f in ("--seed", "--tag", "--no_temporal_attn", "--no_dsca", "--s3d_motion", "--backbone") if f'"{f}"' not in src]
    if missing:
        sys.exit(f"[x] {REPO / 'train.py'} has no {' '.join(missing)} flag(s): it is the old version and would "
                 f"overwrite checkpoints/<sub>/best.pth. Extract revision_code.zip into {REPO} first "
                 f"(cell 0.4 of paper.ipynb).")
    import yaml
    model_cfg = (yaml.safe_load((REPO / "configs" / "default.yaml").read_text(encoding="utf-8")) or {}).get("model", {})
    expected = {"embed_dim": 768, "vit_depth": 6, "vit_heads": 12, "num_frames": 16}
    wrong = {k: model_cfg.get(k) for k, v in expected.items() if model_cfg.get(k) != v}
    if wrong:
        sys.exit(f"[x] {REPO / 'configs' / 'default.yaml'} differs from the seed-42 runs ({wrong}, expected "
                 f"{expected}). Extract revision_code.zip into {REPO} first.")


def stage_train(args, runs):
    check_train_script()
    for run in runs:
        if run.recipe == "v1" and run.name == "full" and run.seed == 42 and args.reuse_legacy \
                and legacy_checkpoint(run).exists() and not (tag_dir(run) / "best.pth").exists():
            print(f"[=] {run.sub}/{run.tag}: reusing the previous seed-42 run {legacy_checkpoint(run)}")
            continue
        done = finished_epochs(run)
        if done >= run.epochs:
            print(f"[=] {run.sub}/{run.tag}: finished ({done}/{run.epochs} epochs)")
            continue
        cmd = [sys.executable, str(REPO / "train.py")] + run.train_args()
        if run.sub == CUHK_SUB:              # stage 1 of recipe m3c: text-image pre-training, its own script
            cmd = [sys.executable, str(REPO / "experiments" / "pretrain_cuhk.py"), "--out_dir", str(tag_dir(run)),
                   "--seed", str(run.seed)] + CUHK_ARGS
            if args.data_root:
                cmd += ["--data_root", args.data_root]
            print(f"[>] {run.sub}/{run.tag}: CUHK pre-training")
            print("$", " ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, cwd=REPO, check=True)
                if finished_epochs(run) < run.epochs:
                    sys.exit(f"[x] {run.sub}/{run.tag}: pretrain_cuhk.py exited without finishing")
            continue
        if run.init_run is not None:
            init = tag_dir(run.init_run)
            if finished_epochs(run.init_run) < run.init_run.epochs or not (init / "best.pth").exists():
                print(f"[!] {run.sub}/{run.tag}: waiting for the joint run {run.init_run.sub}/{run.init_run.tag}")
                if args.dry_run:
                    print("$", " ".join(cmd + ["--init_from", str(init / "best.pth")]))
                continue
            if done == 0:
                cmd += ["--init_from", str(init / "best.pth")]
        if args.data_root:
            cmd += ["--data_root", args.data_root]
        if done > 0 and (tag_dir(run) / "latest.pth").exists():
            cmd += ["--resume", str(tag_dir(run) / "latest.pth")]
        print(f"[>] {run.sub}/{run.tag}: {done}/{run.epochs} epochs done")
        print("$", " ".join(cmd))
        if args.dry_run:
            continue
        t0 = time.time()
        subprocess.run(cmd, cwd=REPO, check=True)
        print(f"[+] {run.sub}/{run.tag} trained in {(time.time() - t0) / 3600:.2f} h")
        if finished_epochs(run) < run.epochs:
            sys.exit(f"[x] {run.sub}/{run.tag}: train.py exited but {tag_dir(run)} has no finished history; "
                     f"check that train.py honours --tag before training anything else.")
        if args.slim:
            # Encode right away, then shrink the run folder: a finished run needs ~4 GB (best.pth + latest.pth
            # with optimizer state); afterwards only the weights (~0.66 GB) or nothing (ablations) are kept.
            if run.sub in (JOINT_SUB, CUHK_SUB):
                slim_run(run, drop=False)          # fine-tunes start from its best.pth; not evaluated itself
            elif encode_run(args, run):
                slim_run(run, drop=(run.name != "full" and not args.keep_ablation_ckpt))


def slim_run(run, drop: bool) -> None:
    import torch

    d = tag_dir(run)
    (d / "latest.pth").unlink(missing_ok=True)
    best = d / "best.pth"
    if not best.exists():
        return
    if drop:
        best.unlink()
        print(f"[-] {run.sub}/{run.tag}: checkpoint removed (embeddings kept in results/emb)")
        return
    ck = common.load_checkpoint(str(best))
    keep = {k: ck[k] for k in ("model_state_dict", "criterion_state_dict", "epoch", "metrics", "best_score",
                               "config", "history") if k in ck}
    if len(keep) < len(ck):
        torch.save(keep, best)
        print(f"[-] {run.sub}/{run.tag}: optimizer state stripped from best.pth")


def encode_run(args, run, device=None) -> bool:
    """Encodes the val and test splits of one run. Returns True when both embedding files exist afterwards."""
    import torch

    out_dir = RESULTS / "emb" / run.sub / run.tag
    todo = [s for s in ("val", "test") if args.force or not (out_dir / f"{s}.npz").exists()]
    need_bank = run.recipe != "v1" and (args.force or not (out_dir / "bank.npz").exists())
    if not todo and not need_bank:
        print(f"[=] {run.sub}/{run.tag}: embeddings exist")
        return True
    ckpt = resolve_checkpoint(run, args.reuse_legacy)
    if ckpt is None:
        print(f"[!] {run.sub}/{run.tag}: no best.pth yet -> run --stage train first")
        return False
    done = finished_epochs(run)
    if run.recipe != "v1" and (tag_dir(run) / "train_history.json").exists() and done < run.epochs:
        # a run cut by the time budget: its best.pth is provisional, encoding it now would freeze stale embeddings
        print(f"[!] {run.sub}/{run.tag}: training unfinished ({done}/{run.epochs} epochs) -> not encoded yet")
        return False
    print(f"[>] {run.sub}/{run.tag}: encoding {todo} with {ckpt}")
    if args.dry_run:
        return False
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, info = common.load_model(str(ckpt), device)
    for split in todo:
        ds = common.build_split_dataset(run.sub, split, args.data_root, info["num_frames"], info["img_size"],
                                       info.get("frame_layout", "uniform"), info.get("dsca_mode", "keyword"))
        E = common.encode_split(model, info, ds, device, batch_size=args.batch_size, num_workers=args.num_workers)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / f"{split}.npz", **E)
        save_json({**info, "split": split, "n_queries": int(len(E["q_vid"])), "n_videos": int(len(E["g_vid"]))},
                  out_dir / f"{split}_info.json")
        print(f"    {split}: {len(E['q_vid'])} captions, {len(E['g_vid'])} videos")
    if need_bank:
        # Train captions only (no video, no val/test text): reference bank for hubness reduction at eval time
        ds = common.build_split_dataset(run.sub, "train", args.data_root, info["num_frames"], info["img_size"],
                                       info.get("frame_layout", "uniform"), info.get("dsca_mode", "keyword"))
        np.savez_compressed(out_dir / "bank.npz", **common.encode_bank(model, info, ds, device))
        print(f"    bank: {len(ds.samples)} train captions (hubness reduction)")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return True


def stage_encode(args, runs):
    for run in runs:
        if run.sub not in (JOINT_SUB, CUHK_SUB):
            encode_run(args, run)


def weight_grid(step: float):
    n = int(round(1.0 / step))
    for i, j in itertools.product(range(n + 1), range(n + 1)):
        if i + j <= n:
            yield (round(i * step, 4), round(j * step, 4), round((n - i - j) * step, 4))


def select_weights(E_val, step: float):
    """Grid search on the validation split (all captions as queries, video-level match)."""
    rows = []
    for w in weight_grid(step):
        r, _, _ = protocol_ranks(E_val, w, "all2", "video")
        m = metrics_from_ranks(r)
        rows.append({"w": list(w), "R@1": m["R@1"], "R@5": m["R@5"], "MdR": m["MdR"]})
    dist = lambda w: sum((a - b) ** 2 for a, b in zip(w, DEFAULT_WEIGHTS))
    rows.sort(key=lambda x: (-x["R@1"], -x["R@5"], x["MdR"], dist(x["w"])))
    return tuple(rows[0]["w"]), rows


def score_run(E_val, E_test, weights):
    res = {"weights": list(weights), "val": {}, "test": {}}
    for proto in ("all2", "mfgf"):
        r, _, _ = protocol_ranks(E_val, weights, proto, "video")
        res["val"][proto] = metrics_from_ranks(r)
    for proto in ("mfgf", "mfgf_cap2", "all2"):
        if len(common.protocol_query_index(E_test, proto)) == 0:
            continue                                   # e.g. a split with a single caption per video
        r, qv, qc = protocol_ranks(E_test, weights, proto, "video")
        rp, _, _ = protocol_ranks(E_test, weights, proto, "pid")
        res["test"][proto] = {"ranks": r.tolist(), "q_vid": qv.tolist(), "q_cap": qc.tolist(),
                              "metrics": metrics_from_ranks(r), "ranks_pid": rp.tolist(),
                              "metrics_pid": metrics_from_ranks(rp)}
    return res


def stage_eval(args, runs):
    for run in runs:
        emb = RESULTS / "emb" / run.sub / run.tag
        out = RESULTS / "runs" / run.sub / f"{run.tag}.json"
        if not ((emb / "val.npz").exists() and (emb / "test.npz").exists()):
            print(f"[!] {run.sub}/{run.tag}: embeddings missing -> run --stage encode first")
            continue
        if out.exists() and not args.force and not args.val_only:
            print(f"[=] {run.sub}/{run.tag}: evaluated")
            continue
        E_val = load_npz(emb / "val.npz")
        w_sel, grid = select_weights(E_val, args.grid_step)
        beta, beta_rows = 0.0, []
        if args.hub and (emb / "bank.npz").exists():
            bank = load_npz(emb / "bank.npz")
            hub_val = common.hub_vectors(E_val, bank)
            for b in HUB_BETAS:
                r, _, _ = protocol_ranks(common.with_hubness(E_val, hub_val, b), w_sel, "all2", "video")
                m = metrics_from_ranks(r)
                beta_rows.append({"beta": b, "R@1": m["R@1"], "R@5": m["R@5"], "R@10": m["R@10"]})
            # mean of R@1/5/10 on val (same criterion as checkpoint selection); ties -> smaller beta
            beta = max(beta_rows, key=lambda x: ((x["R@1"] + x["R@5"] + x["R@10"]) / 3, -x["beta"]))["beta"]
            E_val = common.with_hubness(E_val, hub_val, beta)
        if args.val_only:
            vm = {p: metrics_from_ranks(protocol_ranks(E_val, w_sel, p, "video")[0]) for p in ("all2", "mfgf")}
            save_json({"sub": run.sub, "tag": run.tag, "weights": list(w_sel), "hub_beta": beta,
                       "hub_grid": beta_rows, "val": vm}, out.with_suffix(".val.json"))
            m = vm["all2"]
            print(f"[+] {run.sub}/{run.tag}: w={w_sel} beta={beta}  VAL R@1={m['R@1']:.2f} R@5={m['R@5']:.2f} "
                  f"R@10={m['R@10']:.2f} MdR={m['MdR']:.1f}  (test not evaluated)")
            continue
        E_test = load_npz(emb / "test.npz")
        if beta:
            E_test = common.with_hubness(E_test, common.hub_vectors(E_test, load_npz(emb / "bank.npz")), beta)
        info = load_json(emb / "test_info.json")
        result = {
            "sub": run.sub, "name": run.name, "seed": run.seed, "tag": run.tag, "flags": run.flags,
            "checkpoint": info.get("checkpoint"), "epoch": info.get("epoch"),
            "val_metrics_at_checkpoint": info.get("val_metrics"),
            "n_test_videos": int(len(E_test["g_vid"])), "n_val_videos": int(len(E_val["g_vid"])),
            "grid_step": args.grid_step, "val_grid_top10": grid[:10], "recipe": run.recipe,
            "hub_beta": beta, "hub_grid_val": beta_rows,
            "modes": {
                "val_selected": score_run(E_val, E_test, w_sel),
                "default": score_run(E_val, E_test, DEFAULT_WEIGHTS),
                "global": score_run(E_val, E_test, GLOBAL_WEIGHTS),
            },
        }
        save_json(result, out)
        m = result["modes"]["val_selected"]["test"]["mfgf"]["metrics"]
        print(f"[+] {run.sub}/{run.tag}: w={w_sel} (val)  TEST [24]-protocol R@1={m['R@1']:.2f} R@5={m['R@5']:.2f} "
              f"MdR={m['MdR']:.1f}  ({len(E_test['g_vid'])} test videos)")


def main():
    global RESULTS, CKPT_ROOT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True,
                    choices=["split", "train", "encode", "eval", "aggregate", "figures", "bench", "mt", "all"])
    ap.add_argument("--data_root", default=None, help="Folder that contains TVPReid-PRID/, TVPReid-iLIDs/, TVPReid-Duke/")
    ap.add_argument("--subs", nargs="+", default=list(SUBSETS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--ablation_seeds", nargs="+", type=int, default=[42],
                    help="Seeds for the PRID ablation runs (the review asks for seeds on the main results; "
                         "use 42 1234 2025 here as well if GPU time allows)")
    ap.add_argument("--no_ablations", action="store_true")
    ap.add_argument("--ablations_only", action="store_true",
                    help="Only the ablation runs (the full run of --ablation_seeds already exists elsewhere)")
    ap.add_argument("--ablation_runs", nargs="+", choices=list(ABLATION_RUNS), default=list(ABLATION_RUNS),
                    help="Which PRID ablation runs to include: no_init no_temporal no_dsca no_mim are the leave-one-out "
                         "runs (Table 6b); cum1 cum2 cum3 are only needed for the cumulative Table 6")
    ap.add_argument("--no_reuse_legacy", dest="reuse_legacy", action="store_false",
                    help="Retrain seed 42 instead of reusing checkpoints/<sub>/best.pth of the previous submission")
    ap.add_argument("--recipe", choices=list(RECIPES), default="v1",
                    help="v1 = submitted model; v2s = v2 flags per subset; v2 = joint all-subset run + per-subset fine-tune")
    ap.add_argument("--hub", action="store_true",
                    help="eval: hubness reduction against the train-caption bank, beta chosen on val (0 = off)")
    ap.add_argument("--joint_eval", action="store_true",
                    help="recipe v2: encode/eval the joint model (before fine-tuning) on each subset in --subs")
    ap.add_argument("--joint_name", choices=["joint", "joint2", "joint3"], default="joint",
                    help="with --joint_eval: the 15-epoch joint model or a continued one (joint2 = 25, joint3 = 35 epochs)")
    ap.add_argument("--joint_extra", choices=["joint2", "joint3"], default=None,
                    help="stage train: train only this continued joint model of each seed")
    ap.add_argument("--joint2", dest="joint_extra", action="store_const", const="joint2",
                    help="same as --joint_extra joint2")
    ap.add_argument("--val_only", action="store_true",
                    help="eval: report validation metrics only (development runs; the test split is not scored)")
    ap.add_argument("--grid_step", type=float, default=0.05)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--force", action="store_true", help="Recompute embeddings / evaluations that already exist")
    ap.add_argument("--no_slim", dest="slim", action="store_false",
                    help="Keep latest.pth and the optimizer state after a run (needs ~4 GB per run)")
    ap.add_argument("--keep_ablation_ckpt", action="store_true",
                    help="With slimming, keep the (stripped) best.pth of ablation runs instead of deleting it")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--results", default=str(RESULTS), help="Output folder (default <repo>/results)")
    ap.add_argument("--ckpt_root", default=str(CKPT_ROOT), help="Checkpoint folder (default <repo>/checkpoints)")
    args = ap.parse_args()
    RESULTS, CKPT_ROOT = Path(args.results), Path(args.ckpt_root)

    runs = build_runs(args.subs, args.seeds, args.ablation_seeds, ablations=not args.no_ablations,
                      ablation_names=args.ablation_runs, recipe=args.recipe)
    if args.ablations_only:
        runs = [r for r in runs if r.name != "full"]
    if args.joint_eval:
        runs = [Run(sub, args.joint_name, s, recipe="v2") for sub in args.subs for s in args.seeds]
    elif args.joint_extra:
        runs = [Run(JOINT_SUB, args.joint_extra, s, recipe="v2") for s in args.seeds]
    stages = ["split", "train", "encode", "eval", "aggregate", "figures"] if args.stage == "all" else [args.stage]
    for st in stages:
        print(f"\n========== stage: {st} ==========")
        if st == "split":
            stage_split(args)
        elif st == "train":
            stage_train(args, runs)
        elif st == "encode":
            stage_encode(args, runs)
        elif st == "eval":
            stage_eval(args, runs)
        elif st == "aggregate":
            import aggregate
            aggregate.main(RESULTS)
        elif st == "figures":
            import make_figures
            make_figures.main(RESULTS)
        elif st == "bench":
            cmd = [sys.executable, str(REPO / "experiments" / "bench_cost.py"), "--out", str(RESULTS / "bench.json")]
            print("$", " ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, cwd=REPO, check=True)
        elif st == "mt":
            cmd = [sys.executable, str(REPO / "experiments" / "eval_mt_vi.py"), "--subs", *args.subs,
                   "--seeds", *map(str, args.seeds)]
            if args.data_root:
                cmd += ["--data_root", args.data_root]
            print("$", " ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, cwd=REPO, check=True)


if __name__ == "__main__":
    main()
