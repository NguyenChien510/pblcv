"""
Vietnamese queries through machine translation (review DG25, item 5).

The model is trained on English captions (bert-base-uncased). A Vietnamese user query would be translated to
English before retrieval. Without human-written Vietnamese captions, we simulate it with a round trip:
English test caption -> Vietnamese (opus-mt-en-vi) -> English (opus-mt-vi-en), and measure the Rank@1 drop on the
test split (protocol of [24], weights chosen on validation). The round trip contains two MT steps, so the drop is a
pessimistic estimate for a single VI->EN step.

    pip install sentencepiece sacremoses
    python experiments/eval_mt_vi.py --subs TVPReid-PRID TVPReid-iLIDs TVPReid-Duke --seeds 42 1234 2025
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import REPO, SUBSETS, load_json, load_npz, metrics_from_ranks, ranks_for, save_json  # noqa: E402

RESULTS = REPO / "results"
EN_VI, VI_EN = "Helsinki-NLP/opus-mt-en-vi", "Helsinki-NLP/opus-mt-vi-en"


def translate(texts, model_name, device, batch_size=32):
    from transformers import MarianMTModel, MarianTokenizer
    tok = MarianTokenizer.from_pretrained(model_name)
    mdl = MarianMTModel.from_pretrained(model_name).to(device).eval()
    out = []
    import torch
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(list(texts[i:i + batch_size]), return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
            gen = mdl.generate(**enc, num_beams=4, max_new_tokens=200)
            out += tok.batch_decode(gen, skip_special_tokens=True)
            print(f"\r    {model_name}: {len(out)}/{len(texts)}", end="", flush=True)
    print()
    del mdl
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", nargs="+", default=list(SUBSETS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1234, 2025])
    ap.add_argument("--data_root", default=None)
    args = ap.parse_args()

    import torch
    from data.text_decomposer import TextDecomposer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decomposer = TextDecomposer(language="en")
    for sub in args.subs:
        for seed in args.seeds:
            tag = f"seed{seed}"
            run_json = RESULTS / "runs" / sub / f"{tag}.json"
            npz = RESULTS / "emb" / sub / tag / "test.npz"
            if not (run_json.exists() and npz.exists()):
                print(f"[!] {sub}/{tag}: run the encode and eval stages first")
                continue
            run = load_json(run_json)
            E = load_npz(npz)
            idx = np.where(E["q_cap"] == 0)[0]                      # protocol of [24]: first caption per video
            cache = RESULTS / "mt" / f"{sub}_test_roundtrip.json"
            if cache.exists():
                tr = load_json(cache)
            else:
                src = [str(t) for t in E["q_text"][idx]]
                vi = translate(src, EN_VI, device)
                back = translate(vi, VI_EN, device)
                tr = {"video_id": E["q_vid"][idx].tolist(), "en": src, "vi": vi, "en_back": back}
                save_json(tr, cache)
            model, info = common.load_model(run["checkpoint"], device)
            back = tr["en_back"]
            t_full = common.encode_texts(model, back, device)
            if info.get("use_dsca", True):
                dec = [decomposer.decompose(t) for t in back]
                t_app = common.encode_texts(model, [d[0] for d in dec], device)
                t_mot = common.encode_texts(model, [d[1] for d in dec], device)
            else:
                t_app, t_mot = t_full, t_full
            w1, w2, w3 = run["modes"]["val_selected"]["weights"]
            sim = w1 * t_full @ E["V_video"].T + w2 * t_app @ E["V_app"].T + w3 * t_mot @ E["V_mot"].T
            r_mt = ranks_for(sim, np.array(tr["video_id"]), E["g_vid"])
            m_en = run["modes"]["val_selected"]["test"]["mfgf"]["metrics"]
            m_mt = metrics_from_ranks(r_mt)
            out = {"sub": sub, "seed": seed, "tag": tag, "weights": [w1, w2, w3], "n_queries": int(len(r_mt)),
                   "R@1_en": m_en["R@1"], "R@1_mt": m_mt["R@1"], "metrics_en": m_en, "metrics_mt": m_mt,
                   "examples": [{"en": a, "vi": b, "en_back": c} for a, b, c in
                                list(zip(tr["en"], tr["vi"], tr["en_back"]))[:5]]}
            save_json(out, RESULTS / "mt" / sub / f"{tag}.json")
            print(f"[+] {sub}/{tag}: R@1 EN={m_en['R@1']:.2f}  EN->VI->EN={m_mt['R@1']:.2f}")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
