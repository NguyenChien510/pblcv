"""
Analyses the two labelled videos with the current web-demo engine and reports top-1 / top-3 / MRR on the query sets
in queries.py, with the engine's constraints and with all constraint penalties set to zero. With --vi it also runs
the Vietnamese wordings of the same queries (queries_vi.py: VI and the terse VI2, each also without diacritics),
which go through webapp/vi_query.py inside SearchEngine.search.

    .venv/bin/python experiments/webdemo_eval/eval_real.py ~/Desktop/Nhom9/TESTVIDEO.mp4 ok.mp4 [--vi]
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
from queries_vi import EXT_VI, EXT_VI2, GT_VI, GT_VI2, unaccented  # noqa: E402


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
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    with_vi = "--vi" in sys.argv
    testvideo, station = (Path(p).expanduser() for p in args[:2])
    eng = E.SearchEngine(ROOT, use_clip=True)
    fmt = lambda r: f"{(r == 1).sum()}/{len(r)} top-1, {(r <= 3).sum()}/{len(r)} top-3, MRR {np.mean(1 / r):.3f}"
    with tempfile.TemporaryDirectory() as tmp:
        for name, src, sets, vi in (("TESTVIDEO", testvideo, {"GT9": GT["TESTVIDEO"]}, {"GT9": (GT_VI, GT_VI2, "TESTVIDEO")}),
                                    ("station", station, {"GT15": GT["ok"], "EXT46": EXT},
                                     {"GT15": (GT_VI, GT_VI2, "ok"), "EXT46": (EXT_VI, EXT_VI2, None)})):
            proxy = Path(tmp) / f"{name}.mp4"
            video_io.make_proxy(src, proxy, lambda f: None)
            an = eng.analyze(proxy, name, "eval", {"max_frames": 0, "stride": 1, "conf": 0.35}, lambda *a: None)
            print(f"== {name}: {len(an.tracks)} tracks", flush=True)
            for key, qs in sets.items():
                print(f"   {key:6s} {fmt(ranks(eng, an, qs))}   | no constraints: {fmt(ranks(eng, an, qs, True))}", flush=True)
                if with_vi:
                    v1, v2, sub = vi[key]
                    v1, v2 = (v1[sub], v2[sub]) if sub else (v1, v2)
                    for label, texts in (("VI", v1), ("VI2", v2)):
                        qa = [(t, ids) for t, (_, ids) in zip(texts, qs)]
                        qu = [(unaccented(t), ids) for t, ids in qa]
                        print(f"     {label:4s} {fmt(ranks(eng, an, qa))}   | without diacritics: {fmt(ranks(eng, an, qu))}",
                              flush=True)


if __name__ == "__main__":
    torch.set_num_threads(min(10, torch.get_num_threads()))
    main()
