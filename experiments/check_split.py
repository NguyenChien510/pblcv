"""
Checks the official TVPReid split used for training/selection/testing (review DG25, criterion 3).

For every subset and split it reports the number of videos, captions and person IDs, the number of person IDs
shared between splits, video_id overlaps (must be 0) and how many test videos share their person ID with another
test video. Sizes are compared with Table 2 of [24] (MFGF).

Person IDs are parsed exactly like data/dataset.py does (first integer in the video_id). The script prints a few
video_ids with the parsed IDs so the parsing can be checked by eye.

    python experiments/check_split.py --data_root /content/dataset
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MFGF_SPLIT_VIDEOS, SUBSETS, build_split_dataset, save_json, short_name  # noqa: E402


def split_stats(ds):
    vids = {}
    caps = Counter()
    for s in ds.samples:
        vids[str(s["video_id"])] = int(s["person_id"])
        caps[str(s["video_id"])] += 1
    return vids, caps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--subs", nargs="+", default=list(SUBSETS))
    ap.add_argument("--out", default="results/split_check.json")
    args = ap.parse_args()

    report = {}
    for sub in args.subs:
        per = {}
        for split in ("train", "val", "test"):
            ds = build_split_dataset(sub, split, args.data_root)
            vids, caps = split_stats(ds)
            per[split] = {"vids": vids, "caps": caps,
                          "synthetic": bool(ds.samples and ds.samples[0].get("dataset") == "Synthetic")}
        pid_sets = {k: set(v["vids"].values()) for k, v in per.items()}
        vid_sets = {k: set(v["vids"].keys()) for k, v in per.items()}
        test_pid_counts = Counter(per["test"]["vids"].values())
        val_pid_counts = Counter(per["val"]["vids"].values())
        all_vids = {**per["train"]["vids"], **per["val"]["vids"], **per["test"]["vids"]}
        rep = {
            "videos": {k: len(v["vids"]) for k, v in per.items()},
            "captions": {k: int(sum(v["caps"].values())) for k, v in per.items()},
            "captions_per_video": {k: dict(Counter(v["caps"].values())) for k, v in per.items()},
            "person_ids": {k: len(s) for k, s in pid_sets.items()},
            "person_ids_total": len(set(all_vids.values())),
            "videos_total": len(all_vids),
            "shared_ids": {
                "train_val": len(pid_sets["train"] & pid_sets["val"]),
                "train_test": len(pid_sets["train"] & pid_sets["test"]),
                "val_test": len(pid_sets["val"] & pid_sets["test"]),
            },
            "test_videos_with_shared_ids_in_train": sum(1 for p in per["test"]["vids"].values() if p in pid_sets["train"]),
            "shared_video_ids": {
                "train_val": len(vid_sets["train"] & vid_sets["val"]),
                "train_test": len(vid_sets["train"] & vid_sets["test"]),
                "val_test": len(vid_sets["val"] & vid_sets["test"]),
            },
            "test_videos_whose_pid_repeats_in_test_gallery": sum(c for c in test_pid_counts.values() if c > 1),
            "val_videos_whose_pid_repeats_in_val_gallery": sum(c for c in val_pid_counts.values() if c > 1),
            "mfgf_table2_videos": dict(zip(("train", "val", "test"), MFGF_SPLIT_VIDEOS.get(sub, (None,) * 3))),
            "matches_mfgf_table2": [len(per[k]["vids"]) for k in ("train", "val", "test")] == list(MFGF_SPLIT_VIDEOS.get(sub, ())),
            "examples": [f"{v} -> pid {p}" for v, p in list(per["test"]["vids"].items())[:6]],
            "dummy_data": any(v["synthetic"] for v in per.values()),
        }
        report[sub] = rep
        print(f"\n=== {sub} ({short_name(sub)}) ===")
        print(f"  videos train/val/test : {rep['videos']}   (Table 2 of [24]: {rep['mfgf_table2_videos']}, "
              f"match={rep['matches_mfgf_table2']})")
        print(f"  captions              : {rep['captions']}  per-video counts: {rep['captions_per_video']}")
        print(f"  person IDs            : {rep['person_ids']}  total={rep['person_ids_total']}")
        print(f"  IDs shared            : {rep['shared_ids']}  (test videos whose ID is also in train: "
              f"{rep['test_videos_with_shared_ids_in_train']})")
        print(f"  video_id overlaps     : {rep['shared_video_ids']}  (must all be 0)")
        print(f"  test videos sharing a pid inside the test gallery: {rep['test_videos_whose_pid_repeats_in_test_gallery']}")
        print(f"  example ids           : {rep['examples']}")
        if rep["person_ids_total"] == rep["videos_total"]:
            print("  [!] every video has its own ID number -> the release has no identity labels; identity overlap "
                  "between splits cannot be checked (the counts above are video counts)")
        if rep["dummy_data"]:
            print("  [!] DUMMY DATA - the data root has no captions; these numbers are meaningless")
    save_json(report, args.out)
    print(f"\n[+] saved {args.out}")


if __name__ == "__main__":
    main()
