"""
Aggregates results/runs/**.json into the numbers of the revised paper.

Outputs (in results/):
  summary.json  every aggregated number (means, std over seeds, bootstrap CIs, paired ablation deltas)
  tokens.json   {TOKEN: "formatted value"} consumed by experiments/fill_paper.py (Vietnamese number format)
  tables.md     the same numbers as Markdown tables, for checking by eye

Main results: test split, protocol of [24] (first caption of every video), video-level match, fusion weights and
checkpoint chosen on validation. Point estimate = mean over seeds; CI = percentile bootstrap (10,000 resamples of
the test queries) of the seed-averaged metric. Ablation deltas use a paired bootstrap on the same test queries.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (KS, MFGF_PUBLISHED, SUBSETS, bootstrap_summary, load_json, metrics_from_ranks,  # noqa: E402
                    paired_bootstrap, save_json, short_name)
from plan import ABLATION_SUB, CUMULATIVE_ROWS, LEAVE_ONE_OUT_ROWS  # noqa: E402

B = 10000
MINUS = "−"
SUB_VI = {"TVPReid-PRID": "PRID", "TVPReid-iLIDs": "iLIDs", "TVPReid-Duke": "Duke"}
CUM_LABEL = {2: "khởi tạo từ ImageNet-1K", 3: "thêm chú ý không–thời gian phân chia", 4: "thêm phân rã DSCA",
             5: "thêm ràng buộc ℒMIM", 6: "dùng chấm điểm đa tầng"}
LOO_LABEL = {"no_init": "khởi tạo ImageNet-1K", "no_temporal": "chú ý thời gian", "no_dsca": "phân rã DSCA",
             "no_mim": "ràng buộc ℒMIM", "no_multilevel": "chấm điểm đa tầng"}


# ----------------------------------------------------------------------------------------------
# formatting (Vietnamese decimal comma)
# ----------------------------------------------------------------------------------------------
def vn(x, d=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    s = f"{x:.{d}f}".replace(".", ",")
    return s.replace("-", MINUS)


def vn_signed(x, d=1):
    if x is None:
        return "–"
    r = round(x, d)
    if r == 0:
        return vn(0.0, d)
    return ("+" if r > 0 else MINUS) + vn(abs(x), d)


def vn_ci(lo, hi, d=1):
    return f"[{vn(lo, d)}; {vn(hi, d)}]"


def join_vi(items):
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " và " + items[-1]


# ----------------------------------------------------------------------------------------------
def load_runs(results: Path):
    runs = {}
    for f in sorted((results / "runs").glob("*/*.json")):
        r = load_json(f)
        runs.setdefault(r["sub"], {}).setdefault(r["name"], {})[int(r["seed"])] = r
    return runs


def stack(run_list, mode, proto, key="ranks"):
    """Aligns the queries of several runs (same split) and stacks their ranks -> [n_runs, n_queries]."""
    base = run_list[0]["modes"][mode]["test"][proto]
    keys = list(zip(base["q_vid"], base["q_cap"]))
    pos = {k: i for i, k in enumerate(keys)}
    R = np.zeros((len(run_list), len(keys)), dtype=np.int64)
    for j, r in enumerate(run_list):
        t = r["modes"][mode]["test"][proto]
        kk = list(zip(t["q_vid"], t["q_cap"]))
        if set(kk) != set(keys):
            raise ValueError(f"query sets differ between runs of {r['sub']}")
        for k, rank in zip(kk, t[key]):
            R[j, pos[k]] = rank
    return R, np.array(base["q_vid"])


def main(results=None):
    results = Path(results or Path(__file__).resolve().parent.parent / "results")
    runs = load_runs(results)
    if not runs:
        print("[!] no results/runs/*.json yet -> run the eval stage first")
        return
    S, T, md = {}, {}, []

    # ------------------------------------------------------------------ main results (Table 4 / 4b)
    S["main"] = {}
    md.append("## Table 4 / 4b - test split, protocol of [24] (1 caption/video), mean over seeds\n")
    md.append("| subset | seeds | R@1 | R@5 | R@10 | R@50 | MdR | R@1 95% CI | MFGF R@1 |\n|---|---|---|---|---|---|---|---|---|")
    for sub in SUBSETS:
        full = runs.get(sub, {}).get("full", {})
        if not full:
            continue
        seeds = sorted(full)
        rl = [full[s] for s in seeds]
        R, _ = stack(rl, "val_selected", "mfgf")
        bs = bootstrap_summary(R, B=B, seed=1)
        R2, q2 = stack(rl, "val_selected", "all2")
        bs2 = bootstrap_summary(R2, clusters=q2, B=B, seed=2)
        if "mfgf_cap2" in rl[0]["modes"]["val_selected"]["test"]:
            Rc2, _ = stack(rl, "val_selected", "mfgf_cap2")
            cap2 = float(np.mean([metrics_from_ranks(r)["R@1"] for r in Rc2]))
        else:
            cap2 = float("nan")
        Rp, _ = stack(rl, "val_selected", "mfgf", key="ranks_pid")
        pid_r1 = float(np.mean([metrics_from_ranks(r)["R@1"] for r in Rp]))
        val_r1 = float(np.mean([r["modes"]["val_selected"]["val"]["all2"]["R@1"] for r in rl]))
        S["main"][sub] = {"seeds": seeds, "n_test_queries": int(R.shape[1]), "mfgf_protocol": bs,
                          "all2_protocol": bs2, "cap2_R@1_mean": cap2, "pid_match_R@1_mean": pid_r1,
                          "val_R@1_mean_selected_weights": val_r1, "epochs": [r["epoch"] for r in rl]}
        n = short_name(sub)
        for m, tok in (("R@1", "R1"), ("R@5", "R5"), ("R@10", "R10"), ("R@50", "R50"), ("MdR", "MDR")):
            T[f"TEST_{n}_{tok}"] = vn(bs[m]["mean"])
            T[f"TESTSD_{n}_{tok}"] = "±" + vn(bs[m]["std"]) if len(seeds) > 1 else "–"
        T[f"TESTSDN_{n}_R1"] = vn(bs["R@1"]["std"]) if len(seeds) > 1 else "–"
        T[f"TESTCI_{n}_R1"] = vn_ci(bs["R@1"]["ci_low"], bs["R@1"]["ci_high"])
        T[f"TESTCIW_{n}_R1"] = vn(bs["R@1"]["ci_high"] - bs["R@1"]["ci_low"])
        T[f"TESTDELTA_{n}_R1"] = vn_signed(bs["R@1"]["mean"] - MFGF_PUBLISHED[sub]["R@1"])
        for s, v in zip(seeds, bs["R@1"]["per_seed"]):
            T[f"SEED_{n}_{s}"] = vn(v)
        T[f"TESTMSD_{n}_R1"] = vn(bs["R@1"]["mean"]) + (" ± " + vn(bs["R@1"]["std"]) if len(seeds) > 1 else "")
        T[f"CAP2_{n}_R1"] = vn(cap2)
        T[f"ALL2_{n}_R1"] = vn(bs2["R@1"]["mean"]) + (" ± " + vn(bs2["R@1"]["std"]) if len(seeds) > 1 else "")
        T[f"VALSEL_{n}_R1"] = vn(val_r1)
        md.append(f"| {sub} | {seeds} | {vn(bs['R@1']['mean'])}±{vn(bs['R@1']['std'])} | {vn(bs['R@5']['mean'])} | "
                  f"{vn(bs['R@10']['mean'])} | {vn(bs['R@50']['mean'])} | {vn(bs['MdR']['mean'])} | "
                  f"{vn_ci(bs['R@1']['ci_low'], bs['R@1']['ci_high'])} | {vn(MFGF_PUBLISHED[sub]['R@1'])} |")
    md.append("")

    if S["main"]:
        means = {sub: v["mfgf_protocol"]["R@1"]["mean"] for sub, v in S["main"].items()}
        T["TEST_MIN_R1"] = vn(min(means.values()))
        T["TEST_MAX_R1"] = vn(max(means.values()))
        cap_diffs = [abs(v["cap2_R@1_mean"] - v["mfgf_protocol"]["R@1"]["mean"]) for v in S["main"].values()
                     if not np.isnan(v["cap2_R@1_mean"])]
        if cap_diffs:
            T["CAP_MAXDIFF"] = vn(max(cap_diffs))
        T["ALL2_MAXDIFF"] = vn(max(abs(v["all2_protocol"]["R@1"]["mean"] - v["mfgf_protocol"]["R@1"]["mean"])
                                   for v in S["main"].values()))
        T["MAIN_NSEED"] = str(min(len(v["seeds"]) for v in S["main"].values()))
        cls = {}
        for sub, v in S["main"].items():
            ci = v["mfgf_protocol"]["R@1"]
            ref = MFGF_PUBLISHED[sub]["R@1"]
            cls[sub] = "above" if ci["ci_low"] > ref else ("below" if ci["ci_high"] < ref else "inside")
        above = [SUB_VI[s] for s in SUBSETS if cls.get(s) == "above"]
        inside = [SUB_VI[s] for s in SUBSETS if cls.get(s) == "inside"]
        below = [SUB_VI[s] for s in SUBSETS if cls.get(s) == "below"]
        parts = []
        if above:
            parts.append(f"trên {join_vi(above)}, Rank@1 cao hơn MFGF và khoảng tin cậy bootstrap 95% nằm hoàn toàn "
                         f"trên giá trị công bố của MFGF")
        if inside:
            parts.append(f"trên {join_vi(inside)}, khoảng tin cậy 95% chứa giá trị của MFGF nên chưa thể khẳng định "
                         f"khác biệt")
        if below:
            parts.append(f"trên {join_vi(below)}, Rank@1 thấp hơn MFGF và khoảng tin cậy 95% nằm dưới giá trị công bố")
        T["CMP_SENTENCE"] = "; ".join(parts) + "."
        T["CMP_SENTENCE"] = T["CMP_SENTENCE"][0].upper() + T["CMP_SENTENCE"][1:]
        short = []
        if above:
            short.append(f"cao hơn có ý nghĩa thống kê so với MFGF trên {join_vi(above)}")
        if inside:
            short.append(f"chưa khác biệt có ý nghĩa thống kê trên {join_vi(inside)}")
        if below:
            short.append(f"thấp hơn MFGF trên {join_vi(below)}")
        T["ABS_COMPARE"] = "; ".join(short)
        T["CONCL_COMPARE"] = T["ABS_COMPARE"]
        worse = []
        for sub, v in S["main"].items():
            bs = v["mfgf_protocol"]
            for m in ("R@5", "R@10"):
                if bs[m]["mean"] < MFGF_PUBLISHED[sub][m]:
                    worse.append(f"{m} trên {SUB_VI[sub]} ({vn(bs[m]['mean'])} so với {vn(MFGF_PUBLISHED[sub][m])})")
            if bs["MdR"]["mean"] > MFGF_PUBLISHED[sub]["MdR"]:
                worse.append(f"MdR trên {SUB_VI[sub]} ({vn(bs['MdR']['mean'])} so với {vn(MFGF_PUBLISHED[sub]['MdR'])})")
        T["CMP_OTHER_SENTENCE"] = (f"SpaceTime-DSCA còn kém MFGF ở {join_vi(worse)}; ở các cặp chỉ số–tập con còn lại "
                                   f"(R@5, R@10, MdR) mô hình không thấp hơn MFGF."
                                   if worse else "SpaceTime-DSCA không thấp hơn MFGF ở R@5, R@10 và MdR trên cả ba "
                                                 "tập con.")
        S["comparison_class"] = cls

    # ------------------------------------------------------------------ weights (Table 7)
    S["weights"] = {}
    md.append("## Table 7 - fusion weights chosen on validation\n")
    md.append("| subset | w (mean over seeds) | val R@1 | test R@1 (chosen) | test R@1 (0.50/0.35/0.15) | test R@1 (1/0/0) |\n|---|---|---|---|---|---|")
    deltas_def, deltas_glb = [], []
    for sub in SUBSETS:
        full = runs.get(sub, {}).get("full", {})
        if not full:
            continue
        rl = [full[s] for s in sorted(full)]
        w = np.mean([r["modes"]["val_selected"]["weights"] for r in rl], axis=0)
        val = np.mean([r["modes"]["val_selected"]["val"]["all2"]["R@1"] for r in rl])
        t_sel = np.mean([r["modes"]["val_selected"]["test"]["mfgf"]["metrics"]["R@1"] for r in rl])
        t_def = np.mean([r["modes"]["default"]["test"]["mfgf"]["metrics"]["R@1"] for r in rl])
        t_glb = np.mean([r["modes"]["global"]["test"]["mfgf"]["metrics"]["R@1"] for r in rl])
        per_seed_w = {int(r["seed"]): r["modes"]["val_selected"]["weights"] for r in rl}
        S["weights"][sub] = {"mean_w": w.tolist(), "per_seed_w": per_seed_w, "val_R@1": val,
                             "test_R@1_selected": t_sel, "test_R@1_default": t_def, "test_R@1_global": t_glb}
        n = short_name(sub)
        T[f"W_{n}"] = "(" + "; ".join(vn(x, 2) for x in w) + ")"
        T[f"W_{n}_VAL_R1"] = vn(val)
        T[f"W_{n}_TEST_SEL"] = vn(t_sel)
        T[f"W_{n}_TEST_DEF"] = vn(t_def)
        T[f"W_{n}_TEST_GLB"] = vn(t_glb)
        deltas_def.append(t_sel - t_def)
        deltas_glb.append(t_sel - t_glb)
        md.append(f"| {sub} | {T[f'W_{n}']} | {vn(val)} | {vn(t_sel)} | {vn(t_def)} | {vn(t_glb)} |")
    md.append("")
    if deltas_def:
        T["W_SENTENCE"] = (f"So với bộ mặc định (0,50; 0,35; 0,15), trọng số chọn trên validation làm Rank@1 trên test "
                           f"thay đổi từ {vn_signed(min(deltas_def))} đến {vn_signed(max(deltas_def))} điểm phần trăm; "
                           f"so với chỉ dùng nhánh toàn thể, mức thay đổi là từ {vn_signed(min(deltas_glb))} đến "
                           f"{vn_signed(max(deltas_glb))} điểm.")

    # ------------------------------------------------------------------ ablations on PRID (Tables 6 / 6b)
    # Each table is computed from the seeds for which all of its runs exist, so the leave-one-out table (required by
    # the review) does not wait for the cumulative runs and vice versa.
    abl = runs.get(ABLATION_SUB, {})

    def seeds_for(names):
        if not all(abl.get(n) for n in names):
            return []
        return sorted(set.intersection(*[set(abl[n]) for n in names]))

    def R_of(name, mode, seeds):
        return stack([abl[name][s] for s in seeds], mode, "mfgf")[0]

    cum_seeds = seeds_for({name for name, _ in CUMULATIVE_ROWS})
    loo_seeds = seeds_for({name for name, _, _ in LEAVE_ONE_OUT_ROWS})
    S["ablation"] = {"cumulative_seeds": cum_seeds, "leave_one_out_seeds": loo_seeds, "cumulative": [], "leave_one_out": []}
    if cum_seeds or loo_seeds:
        T["ABL_NSEED"] = str(min(len(x) for x in (cum_seeds, loo_seeds) if x))

    if cum_seeds:
        md.append(f"## Table 6 - cumulative ablation, {ABLATION_SUB} test, protocol of [24], seeds {cum_seeds}\n")
        md.append("| # | run / scoring | R@1 | R@5 | MdR | delta R@1 vs previous [95% CI] |\n|---|---|---|---|---|---|")
        prev = None
        sig_steps, cum_parts = [], []
        for i, (name, mode) in enumerate(CUMULATIVE_ROWS, start=1):
            R = R_of(name, mode, cum_seeds)
            bs = bootstrap_summary(R, B=2000, seed=10 + i)
            row = {"row": i, "run": name, "mode": mode, "summary": bs}
            T[f"ABLC_{i}_R1"] = vn(bs["R@1"]["mean"]) + (" ± " + vn(bs["R@1"]["std"]) if len(cum_seeds) > 1 else "")
            T[f"ABLC_{i}_R5"] = vn(bs["R@5"]["mean"])
            T[f"ABLC_{i}_MDR"] = vn(bs["MdR"]["mean"])
            dtxt = ""
            if prev is not None:
                d = paired_bootstrap(R, prev, "R@1", B=B, seed=100 + i)
                row["delta_prev"] = d
                T[f"ABLC_D{i}"] = vn_signed(d["diff"])
                T[f"ABLC_D{i}_CI"] = vn_ci(d["ci_low"], d["ci_high"])
                dtxt = f"{vn_signed(d['diff'])} {vn_ci(d['ci_low'], d['ci_high'])}"
                cum_parts.append(f"{vn_signed(d['diff'])} điểm ({vn_ci(d['ci_low'], d['ci_high'])}) khi {CUM_LABEL[i]}")
                if d["ci_low"] > 0 or d["ci_high"] < 0:
                    sig_steps.append(CUM_LABEL[i])
            S["ablation"]["cumulative"].append(row)
            md.append(f"| {i} | {name} / {mode} | {T[f'ABLC_{i}_R1']} | {T[f'ABLC_{i}_R5']} | {T[f'ABLC_{i}_MDR']} | {dtxt} |")
            prev = R
        nq = int(prev.shape[1])
        T["ABLC_SENTENCE"] = ("So với cấu hình liền trước, Rank@1 thay đổi " + join_vi(cum_parts) + ". "
                              + (f"Các bước có khoảng tin cậy 95% không chứa 0 là: {join_vi(sig_steps)}; các bước còn "
                                 f"lại chưa đủ bằng chứng thống kê với {nq} truy vấn."
                                 if sig_steps else
                                 f"Không bước nào có khoảng tin cậy 95% loại trừ 0 với {nq} truy vấn, nên các mức "
                                 f"thay đổi chỉ được xem là xu hướng."))
        md.append("")
    else:
        print("[!] cumulative ablation runs (cum1-cum3, no_mim, full) incomplete -> Table 6 tokens left empty")

    if loo_seeds:
        md.append(f"## Table 6b - leave-one-out ablation (delta vs full model, paired bootstrap), seeds {loo_seeds}\n")
        md.append("| removed | R@1 | R@5 | MdR | delta R@1 [95% CI] |\n|---|---|---|---|---|")
        R_full = R_of("full", "val_selected", loo_seeds)
        loo_parts, drops = [], []
        for name, mode, key in LEAVE_ONE_OUT_ROWS:
            R = R_of(name, mode, loo_seeds)
            bs = bootstrap_summary(R, B=2000, seed=7)
            row = {"key": key, "run": name, "mode": mode, "summary": bs}
            T[f"ABLL_{key}_R1"] = vn(bs["R@1"]["mean"]) + (" ± " + vn(bs["R@1"]["std"]) if len(loo_seeds) > 1 else "")
            T[f"ABLL_{key}_R5"] = vn(bs["R@5"]["mean"])
            T[f"ABLL_{key}_MDR"] = vn(bs["MdR"]["mean"])
            if key == "full":
                T["ABLL_full_D"] = "–"
            else:
                d = paired_bootstrap(R, R_full, "R@1", B=B, seed=200 + len(loo_parts))
                row["delta_full"] = d
                T[f"ABLL_{key}_D"] = f"{vn_signed(d['diff'])} {vn_ci(d['ci_low'], d['ci_high'])}"
                loo_parts.append(f"bỏ {LOO_LABEL[key]}: {vn_signed(d['diff'])} ({vn_ci(d['ci_low'], d['ci_high'])})")
                drops.append((d["diff"], key, d))
            S["ablation"]["leave_one_out"].append(row)
            md.append(f"| {key} | {T[f'ABLL_{key}_R1']} | {T[f'ABLL_{key}_R5']} | {T[f'ABLL_{key}_MDR']} | {T[f'ABLL_{key}_D']} |")
        md.append("")
        drops.sort(key=lambda x: x[0])
        sig_drops = [(dd, k) for dd, k, d in drops if d["ci_high"] < 0]
        T["ABLL_SENTENCE"] = ("So với mô hình đầy đủ, Rank@1 thay đổi như sau khi bỏ từng thành phần: " + join_vi(loo_parts)
                              + ". " + (f"Các thành phần mà việc bỏ đi làm Rank@1 giảm có ý nghĩa thống kê: "
                                        f"{join_vi(LOO_LABEL[k] for _, k in sig_drops)}."
                                        if sig_drops else
                                        "Không thành phần nào khi bỏ đi làm Rank@1 giảm có ý nghĩa thống kê ở mức 95%."))
        top = [(dd, k) for dd, k, _ in drops if dd < 0][:2]
        if top:
            T["ABS_ABLATION"] = ("bỏ " + join_vi(LOO_LABEL[k] for _, k in top) + " làm Rank@1 giảm nhiều nhất ("
                                 + join_vi(vn_signed(dd) for dd, _ in top) + " điểm phần trăm)")
        else:
            T["ABS_ABLATION"] = "không thành phần nào khi bỏ đi làm Rank@1 giảm"
    else:
        print("[!] leave-one-out runs (no_init, no_temporal, no_dsca, no_mim, full) incomplete -> Table 6b tokens left empty")

    # ------------------------------------------------------------------ split check (Table 3)
    sc_path = results / "split_check.json"
    if sc_path.exists():
        sc = load_json(sc_path)
        S["split_check"] = sc
        tot_id = 0
        ovl = []
        all_match, no_vid_overlap, per_video_ids = True, True, True
        for sub in SUBSETS:
            if sub not in sc:
                continue
            r = sc[sub]
            all_match &= bool(r["matches_mfgf_table2"])
            no_vid_overlap &= all(v == 0 for v in r["shared_video_ids"].values())
            # In the TVPReid release every video_id carries its own number (person0001 ... person1134 for 1134 PRID
            # videos), so the parsed "person ID" identifies the video, not the person.
            per_video_ids &= r["person_ids_total"] == r["videos_total"]
            ovl.append((SUB_VI[sub], r["shared_ids"]["train_test"], r["person_ids"]["test"]))
            tot_id += r["person_ids_total"]
        for sub in SUBSETS:
            if sub in sc:
                n = short_name(sub)
                r = sc[sub]
                T[f"SPLIT_{n}_NID"] = "–" if per_video_ids else f"{r['person_ids_total']:,}".replace(",", ".")
                T[f"SPLIT_{n}_IDOVL"] = "–" if per_video_ids else f"{r['shared_ids']['train_test']:,}".replace(",", ".")
        T["SPLIT_TOTAL_NID"] = "–" if per_video_ids else f"{tot_id:,}".replace(",", ".")
        first = ("Kích thước các tập train/val/test đọc từ dữ liệu "
                 + ("khớp Bảng 2 của [24] ở cả ba tập con" if all_match else "KHÔNG khớp hoàn toàn Bảng 2 của [24] (xem results/split_check.json)")
                 + (" và không có video nào xuất hiện ở hai tập" if no_vid_overlap else "; CÓ video xuất hiện ở hai tập"))
        if per_video_ids:
            second = (". Tuy nhiên, bản phát hành TVPReid đặt cho mỗi video một mã riêng (ví dụ 1.134 mã cho 1.134 video "
                      "của PRID) và không kèm mã danh tính gốc của PRID-2011, iLIDS-VID, DukeMTMC-VideoReID, nên không thể "
                      "kiểm tra phép chia có tách rời theo danh tính hay không; vì phép chia của [24] thực hiện theo video, "
                      "cùng một người vẫn có thể xuất hiện ở cả train và test (ví dụ qua hai camera).")
            T["SPLIT_LIMIT"] = ("phép chia của [24] thực hiện theo video và bản phát hành không kèm mã danh tính, nên chưa "
                                "kiểm chứng được khả năng tổng quát hóa sang người chưa xuất hiện khi huấn luyện")
        elif any(o for _, o, _ in ovl):
            second = ("; tuy nhiên phép chia không tách rời theo danh tính: lần lượt "
                      + join_vi(f"{o} trên {nt} danh tính của tập test ({s})" for s, o, nt in ovl)
                      + " cũng xuất hiện trong tập train.")
            T["SPLIT_LIMIT"] = ("phép chia của [24] không tách rời theo danh tính (Bảng 3), nên kết quả chưa phản ánh hoàn "
                                "toàn khả năng tổng quát hóa sang danh tính chưa gặp khi huấn luyện")
        else:
            second = "; phép chia đồng thời tách rời theo danh tính (không có danh tính nào xuất hiện ở cả train và test)."
            T["SPLIT_LIMIT"] = "đánh giá mới giới hạn trong một phép chia cố định của [24]"
        T["SPLIT_SENTENCE"] = first + second

    # ------------------------------------------------------------------ bench / MT (optional stages)
    bench = results / "bench.json"
    if bench.exists():
        b = load_json(bench)
        S["bench"] = b
        if b.get("device") == "cuda" and "train_peak_gb" in b.get("ours", {}):
            T["BENCH_GPU"] = b.get("gpu", "GPU")
            T["BENCH_MEM_OURS"] = vn(b["ours"]["train_peak_gb"], 2)
            T["BENCH_MEM_MFGF"] = vn(b["vit_s3d"]["train_peak_gb"], 2)
            diff = 100.0 * (1 - b["ours"]["train_peak_gb"] / b["vit_s3d"]["train_peak_gb"])
            T["BENCH_MEM_DIFF"] = (f"ít hơn {vn(diff, 0)}%" if diff >= 0 else f"nhiều hơn {vn(-diff, 0)}%")
            nock = b["ours"].get("train_peak_gb_no_checkpointing")
            if nock:
                d2 = 100.0 * (nock / b["vit_s3d"]["train_peak_gb"] - 1)
                T["BENCH_MEM_DIFF"] += (f" nhờ gradient checkpointing mà SpaceTimeViT dùng khi huấn luyện (nếu tắt "
                                        f"checkpointing, SpaceTimeViT cần {vn(nock, 2)} GB, "
                                        + (f"nhiều hơn {vn(d2, 0)}%" if d2 >= 0 else f"ít hơn {vn(-d2, 0)}%") + ")")
            T["BENCH_LAT_OURS"] = vn(b["ours"]["infer_ms_per_video"], 1)
            T["BENCH_LAT_MFGF"] = vn(b["vit_s3d"]["infer_ms_per_video"], 1)
            T["BENCH_BATCH"] = str(b.get("batch", 8))
        else:
            print("[!] results/bench.json was measured without a GPU -> resource tokens left empty")
    mt_files = sorted((results / "mt").glob("*/*.json"))
    if mt_files:
        per = {}
        for f in mt_files:
            r = load_json(f)
            per.setdefault(r["sub"], []).append(r)
        S["mt"] = {}
        parts = []
        for sub in SUBSETS:
            if sub not in per:
                continue
            en = float(np.mean([r["R@1_en"] for r in per[sub]]))
            vi = float(np.mean([r["R@1_mt"] for r in per[sub]]))
            S["mt"][sub] = {"R@1_en": en, "R@1_mt": vi, "n_runs": len(per[sub])}
            n = short_name(sub)
            T[f"MT_{n}_EN"] = vn(en)
            T[f"MT_{n}_VI"] = vn(vi)
            T[f"MT_{n}_DROP"] = vn(en - vi)
            parts.append(f"từ {vn(en)}% xuống {vn(vi)}% trên {SUB_VI[sub]}")
        T["MT_SENTENCE"] = "Rank@1 trên tập test giảm " + join_vi(parts) + "."

    save_json(S, results / "summary.json")
    save_json(T, results / "tokens.json")
    (results / "tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[+] {len(T)} tokens -> {results / 'tokens.json'}")
    print(f"[+] tables -> {results / 'tables.md'}")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
