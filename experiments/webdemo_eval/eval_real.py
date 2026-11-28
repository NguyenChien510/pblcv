"""
Analyses the two labelled videos with the current web-demo engine and reports top-1 / top-3 / MRR on the query sets
in queries.py, with the engine's constraints and with all constraint penalties set to zero.

    .venv/bin/python experiments/webdemo_eval/eval_real.py ~/Desktop/Nhom9/TESTVIDEO.mp4 ok.mp4
Takes about 10 minutes on a laptop CPU (the station clip dominates).
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from webapp import engine as E, video_io  # noqa: E402
from queries import EXT, GT  # noqa: E402


def ranks(eng, an, qs, no_penalty=False):
    out = []
    for q, ids in qs:
        scores = eng.search(an, q)["scores"]
        if no_penalty:
            scores = [dict(s, penalty=0.0) for s in scores]
        order = [s["track_id"] for s in E.SearchEngine.rank(scores, E.DEFAULT_WEIGHTS)]
        out.append(min(order.index(i) + 1 for i in ids))
    return np.array(out)


def main() -> None:
    testvideo, station = (Path(p).expanduser() for p in sys.argv[1:3])
    eng = E.SearchEngine(ROOT, use_clip=True)
    fmt = lambda r: f"{(r == 1).sum()}/{len(r)} top-1, {(r <= 3).sum()}/{len(r)} top-3, MRR {np.mean(1 / r):.3f}"
    with tempfile.TemporaryDirectory() as tmp:
        for name, src, sets in (("TESTVIDEO", testvideo, {"GT9": GT["TESTVIDEO"]}),
                                ("station", station, {"GT15": GT["ok"], "EXT46": EXT})):
            proxy = Path(tmp) / f"{name}.mp4"
            video_io.make_proxy(src, proxy, lambda f: None)
            an = eng.analyze(proxy, name, "eval", {"max_frames": 0, "stride": 1, "conf": 0.35}, lambda *a: None)
            print(f"== {name}: {len(an.tracks)} tracks", flush=True)
            for key, qs in sets.items():
                print(f"   {key:6s} {fmt(ranks(eng, an, qs))}   | no constraints: {fmt(ranks(eng, an, qs, True))}", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(min(10, torch.get_num_threads()))
    main()
