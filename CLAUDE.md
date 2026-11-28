# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

SpaceTime-DSCA (code also calls it "SpaceTime-DAMR"/`DAMRModel`): PyTorch Text-to-Video Person Retrieval on the TVPReid benchmark (`TVPReid-Duke`, `TVPReid-PRID`, `TVPReid-iLIDs`). Pure ViT + BERT. It deliberately avoids CLIP pretraining so comparison with the MFGF (ACM MM '24) baseline is fair. README.md is in Vietnamese and holds the research rationale, ablation tables and results.

There is no test suite, linter or build step. `pbl6remake_cvclip.ipynb` is the Colab runner.

## Commands

```bash
# Environment: the system Python has no pip/ensurepip, so use uv (~/.local/bin/uv).
# The local machine has no NVIDIA GPU, so the CPU torch wheel is used.
~/.local/bin/uv venv .venv --python 3.12
~/.local/bin/uv pip install --python .venv/bin/python -r requirements.txt \
    --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
source .venv/bin/activate


# Train (CLI flags override configs/default.yaml)
python train.py --sub_dataset TVPReid-PRID --epochs 35 --lr 0.0001 --batch_size 16
python train.py --sub_dataset TVPReid-PRID --resume latest      # or 'best', or a .pth path
python train.py --data_root /path/to/dataset --no_pretrained_vit

# Evaluate (defaults to checkpoints/<sub_dataset>/best.pth)
python test.py --sub_dataset TVPReid-PRID --checkpoint checkpoints/TVPReid-PRID/best.pth --confusion_matrix --cm_size 10
python test.py --sub_dataset TVPReid-PRID --dashboard          # regenerate training dashboard from ckpt history

# Training-free inference tricks (utils/rerank.py): fusion weights, flip-TTA, hubness reduction with a caption bank.
# Without these flags test.py reproduces the original numbers exactly.
python test.py --sub_dataset TVPReid-PRID --weights 0.5 0.35 0.15 --tta_flip --hub train --hub_beta 0.5
python test.py --sub_dataset TVPReid-PRID --sweep              # one encode, full grid -> reports/<sub>/inference_sweep.json

# Single text query against the val gallery
python test.py --sub_dataset TVPReid-PRID --query "A man in a dark grey jacket walking away"

# Web demo (FastAPI + static UI); settings from .env (Supabase gallery, admin password), see webapp section
python web_demo.py --port 8000 --no-browser

# Deploy (Docker + Caddy basic auth; model weights are mounted from ./checkpoints, not baked in). See INSTALL.md
# sections 4 (VPS) and 5 (Compute Engine L4). GPU: add -f docker-compose.gpu.yml (CUDA torch via TORCH_INDEX=cu126).
cd deploy && cp .env.example .env && docker compose up -d --build
# AWS EC2 + CI/CD (INSTALL.md section 6): .github/workflows/deploy.yml builds the image on every push to main that
# touches the app, pushes ghcr.io/<owner>/pblcv-webdemo:<sha>, then SSHes to vars.EC2_HOST (secret EC2_SSH_KEY) and
# pipes deploy/remote-deploy.sh: pull, APP_IMAGE into deploy/.env, `up -d --no-build`, wait for /api/status, roll
# back to the previous image on failure. deploy/ec2-setup.sh <name>.duckdns.org prepares a fresh Ubuntu instance
# (Docker, swap, model files, a systemd timer that points the DuckDNS name at the current public IP at boot and every
# 5 min, .env, deploy key). Target: c7i.4xlarge (Paid plan + 16-vCPU quota). Without EC2_HOST the deploy job is skipped.
```

If there are no captions under the data root, `TVPRDataset` silently builds a dummy index (32 train / 16 val samples). That is fine for a smoke test, but if metrics look meaningless, check this first.

## Paths and data layout (`utils/paths.py`)

- `paths` is a module-level singleton. `paths.set_sub_dataset(name)` creates `checkpoints/<sub>/`, `reports/<sub>/` and `outputs/<sub>/` inside the repo (`all` when no sub-dataset is given). `normalize_sub_dataset_name` maps loose names such as "prid" or "duke" to their canonical names.
- The data root is resolved in this order: `--data_root` → the `DATA_ROOT` env var → the first existing path among `/content/dataset`, `/content/TVPReid_dataset`, `C:/dataset`, `../dataset`, `../TVPReid_dataset` and `./dataset`. The default dataset location is next to the repo, not inside it.
- The dataset scanner finds any directory with a `captions/` subfolder. It joins `captions/{train,val,test}.csv` (`video_id` column) with `captions/*.json` (video_id → list of captions). It also accepts a CSV with a direct `caption` column, or JSON-only annotations.
- Checkpoints (`*.pth`) are gitignored. `reports/<sub>/` holds committed `train_history.json` and dashboard PNGs.

## Architecture

Data flow: `data/dataset.py` → `models/damr_main.py::DAMRModel` → `loss/criterion.py::DAMRCriterion`, with evaluation in `utils/metrics.py`.

- **Text decomposition (DSCA)**: `data/text_decomposer.py::TextDecomposer` runs inside `__getitem__`. It splits each caption into `q_app` (clothing/colors/accessories) and `q_mot` (motion verbs/direction). It is vocabulary-rule based, with an optional `llm_cache` dict override. Each batch dict carries `caption`, `q_app`, `q_mot`, `video_frames` `[B,T,C,H,W]` (T=16, 224²), `person_id` and `video_id`.
- **Text encoder**: `models/text_encoder.py` is a single shared BERT-base-uncased that encodes `T_full`, `T_app` and `T_mot`, all 768-D.
- **Video encoder**: `models/spacetime_vit.py::SpaceTimeViT` is a TimeSformer-style divided space-time ViT (6 blocks, 12 heads). Its spatial layers load ImageNet-1K weights from torchvision `vit_b_16`, falling back to HF `google/vit-base-patch16-224` if that fails, and its temporal attention `out_proj` is zero-initialized. It returns `(V_app, V_mot, V_video)`: attention-pooled appearance, a Bi-GRU motion stream, and an MLP fusion of the two.
- **Loss**: `L = InfoNCE(T_full, V_video) + λ_app·InfoNCE(T_app, V_app) + λ_mot·InfoNCE(T_mot, V_mot) + λ_mim·L_MIM`. `L_MIM` is the cosine orthogonality between V_app and V_mot (`models/mim_loss.py`); it is computed in the model forward pass and returned as `l_mim`.
- **Eval protocol** (`train.py::evaluate`): every caption is a query. The gallery is **deduplicated by `video_id`**, because each video has 2 captions; this deduplication is essential for correct Rank@K/MdR. The score is `0.50·sim(T_full,V_video) + 0.35·sim(T_app,V_app) + 0.15·sim(T_mot,V_mot)`. These weights are hardcoded and the `alpha` argument is ignored here. `test.py` instead goes through `compute_similarity_matrix` using `--alpha` (default 0.85), so numbers from the two paths can differ.

## Web demo (`web_demo.py` → `webapp/`)

`web_demo.py` is only a launcher. The app is split into a one-off **analyse** step per video and a cheap **search** step per prompt:

- **Prepare** (`webapp/video_io.py`): every input video is re-encoded to a constant-frame-rate H.264 `proxy.mp4` (max 720p) using the ffmpeg bundled with `imageio-ffmpeg`. *All* analysis reads the proxy, so frame index `i` equals player time `i / fps` in the browser overlay. Do not analyse the original file.
- **Analyse** (`webapp/engine.py::SearchEngine.analyze`):
  - Pass 1 runs `YOLO.track` with ultralytics **BoT-SORT**: Kalman filter, two-stage high/low-score Hungarian matching, and native ReID features (`with_reid`, `model: auto`). The tracker yaml is written to a temp file for each run. The UI `conf` is used as `track_high_thresh`; the detector runs at `DETECT_CONF_LOW`. `imgsz` is 960 in Accurate mode and 640 in Balanced/Fast mode. This pass keeps only boxes plus a per-detection quality score (conf × (1 − occlusion) × size). The greedy tracker `_track_greedy` is only a fallback for non-YOLO detectors.
  - Tracks with fewer than `MIN_TRACK_DETECTIONS` detections or a median height under `MIN_TRACK_HEIGHT` are dropped.
  - Pass 2 re-reads only the frames each 16-frame clip needs. In each of the 16 time bins it picks the best-quality (least occluded) detection.
  - Each clip is batch-encoded into `V_app/V_mot/V_video`. On CPU this costs about 2s per track, which dominates analysis time in crowded scenes.
  - **Re-ID (split + stitch).** OpenVINO `person-reidentification-retail-0277` (about 8MB, `checkpoints/reid/`) is a real person re-ID model. Do **not** use `V_app` for re-ID: it is a text-retrieval embedding, and different people reach about 0.92 there. It is only the fallback when the re-ID model is missing.
    - During tracking, a re-ID sample is taken per track every `REID_SAMPLE_S` (0.25s), only from boxes with overlap ≤ `REID_SAMPLE_MAX_OCCL`.
    - `_split_by_appearance` cuts a track where the mean embeddings before and after differ (cosine < `SPLIT_MAX_SIM` 0.40). This fixes BoT-SORT jumping to the person who occludes the tracked one.
    - `_stitch` joins tracklet B onto A when all of these hold:
      - B starts at most `STITCH_MAX_GAP_S` (8s) after A ends;
      - B starts near A's last position;
      - the genders are not clearly different;
      - the cosine of the sample-mean embeddings `Track.reid` is ≥ `REID_MIN_SIM` 0.60.
    - Calibration on a crowded station clip: different people reach a p99 of about 0.40, the same person a median of about 0.79, and 0.52–0.56 produced visible wrong joins.
    - On that clip, mixed-identity tracks fell from 37% to 21% and 27 tracks span more than 10s. Merged tracks are re-cropped, and ids are renumbered by start time.
    - **Order:** with the re-ID model, `analyze` stitches *before* CLIP and the video encoder run, so each final track is encoded once (encoding first repeated about half of that work: 280 encodes for 149 final tracks on the station clip; TESTVIDEO went from about 165 s to 130 s with the same tracks and top-1). Without the re-ID model the old order stays, because stitching then compares `V_app`.
    - CPU speed notes (i5-1235U, TESTVIDEO): detection + tracking is about 60% of the time (YOLO ~95 ms/frame at 960, OpenVINO re-ID ~35 ms per sampled crop). An OpenVINO export of YOLO was slower than PyTorch, 12 torch threads were slower than 10, re-ID on a background thread gained nothing, and detecting every 2nd frame at 960 lost people (top-1 8/9 -> 7/9). Video encoder (mfgf, ~1.5-1.9 s/track: ViT on 4 frames ~1.1 s, S3D ~0.45 s): batch size 1/4/8 makes no difference, an OpenVINO conversion was 2.5x slower, and dynamic int8 on the Linear layers cut station-clip encoding 266 s -> 195 s but dropped top-1 there from 15/15 to 14/15. CLIP on 1 crop instead of 3 (105 s -> 36 s on the station clip) lost one top-1 on each labelled video. Both of those two are enabled as **speed mode** (2026-11-21, on request; constants `CLIP_CROPS = 1` and `ENCODER_INT8 = True` under `# SPEED MODE` in `webapp/engine.py`; undo with `CLIP_CROPS = 3`, `ENCODER_INT8 = False`). With int8, `load_checkpoint` always rebuilds the model (no fp32 weights to load into) and `checkpoint_info.encoder_int8` reports it; on GPU the encoder stays fp32. Combined, measured 2026-11-27 on the same 24 labelled queries: TESTVIDEO 8/9 -> 7/9 top-1 and the station clip 15/15 -> 14/15 (back to 15/15 once the bag constraint fuses the detector's handbags, see Objects below).
  - The result is cached in memory as an `Analysis`, keyed by `(video_id, max_frames, stride, conf, TRACKER_VERSION)`. Only the first `max_frames` frames are processed. Bump `TRACKER_VERSION` when tracking logic changes.
- **Search** returns three raw cosines per track: `T_full↔V_video`, `T_app↔V_app` and `T_mot↔V_mot`. `T_app`/`T_mot` come from `TextDecomposer.extract_concepts`. Fusion weights (`DEFAULT_WEIGHTS`, 0.45/0.45/0.09 for joint/app/mot) and the softmax(τ=0.07) "match probability" are computed **in the browser** (`static/js/app.js::rerank`). The same logic is mirrored in `SearchEngine.rank` (used by the eval scripts), so keep the two in sync. The annotated-MP4 download and the ZIP report were removed on 2026-11-28 (UI, endpoints, `engine.render`, `webapp/report.py`).
- **Vietnamese queries** (`webapp/vi_query.py`, 2026-11-28): `search` first calls `vi_query.translate(prompt)` and everything downstream (decomposer, BERT, CLIP, `query_attributes`) uses the English it returns; `prompt` in the result stays the original, plus `prompt_en`, `prompt_lang`, `unknown_words`.
  - **Detection.** Vietnamese letters, or, for text typed without diacritics, at least two unaccented Vietnamese words outnumbering common English ones. Anything else is returned unchanged; 0 of 13,118 TVPReid captions and none of the 70 English eval queries are detected as Vietnamese.
  - **Rules.** A person-description glossary (`LEXICON`: people, garments, items, hair, colours, shades, patterns, sizes, sleeves, verbs, actions, directions; longest phrase first). An input word matches a glossary word exactly, or, when the input word has no diacritics at all, with the diacritics stripped. In a sentence typed with diacritics an unaccented word is taken literally when it has an exact reading ("áo cam" = orange). Ambiguous unaccented readings are resolved by position: after a garment/item/colour/"màu" a modifier wins (do = đỏ, dam = đậm); before a holdable item a verb wins (cam tui = cầm túi); after a person an age/trait wins (hoi = hói). Noun phrases are rebuilt as [size/age] [sleeves] [pattern] [colours] [material] noun, because Vietnamese puts modifiers after the noun and `_parse_colors` needs the colour right before the garment ("áo khoác xanh đậm" -> "a dark blue jacket"). A garment right after the person or an intransitive action gets "wearing", an item or hair gets "with". Unknown words are dropped and returned for the UI.
  - **UI.** The DSCA panel shows an "EN" row with the English and the unknown words.
  - **Measured** (`experiments/webdemo_eval/queries_vi.py`, `eval_real.py --vi`) on the 70 labelled real-footage queries rewritten in Vietnamese, top-1:
    - untranslated Vietnamese 1/70;
    - rules 52/70, identical without diacritics, against 49/70 for the English originals;
    - a second, terse wording (VI2, written after the glossary and not used to tune it) 49/70 as first measured and 51/70 after adding the three gaps it showed ("cam" in accented text, "đi tới camera", "tiến").
    - Helsinki-NLP/opus-mt-vi-en was tried and rejected: 37/70, 6/70 without diacritics, 28/70 on VI2. It loses "vali", turns "người" into "man" and moves colours to the wrong garment.
  - **Limits.** Free-form phrases outside the glossary are not understood; bare "xanh" is blue; unaccented "ba lo" is a backpack unless it follows "ao".
- **CLIP (hidden branch)**: every track also gets a CLIP image embedding, and `search` returns a `clip` score mapped onto the joint score's mean/std. It is not one of the tunable weights and has no UI: `rank`/`rerank` compute `(Σ w·dsca + CLIP_WEIGHT·clip) / (1 + CLIP_WEIGHT)` with `CLIP_WEIGHT = 0.2` (sent to the browser as `status.clip_weight`). The embedding is the mean over `CLIP_CROPS` preview crops (1 in speed mode, 3 before). On two hand-labelled real videos (24 queries, re-measured 2026-11-21 with 3 crops) it moves top-1 from 7/9 to 8/9 and from 12/15 to 15/15; every weight from 0.1 to 1.0 gives the same top-1 there, and CLIP alone gets 5/9 and 14/15. It does not help on TVPReid tracklets. `--no_clip` disables it.
- **Attribute matching**: the DSCA scores alone are weak on gender (roughly chance), colours and carried items, so query constraints are checked against dedicated models.
  - **Models.** OpenVINO `person-attributes-recognition-crossroad-0230` (about 3MB) is auto-downloaded to `checkpoints/attributes/`. It gives 8 probabilities (male, bag, backpack, hat, longsleeves, longpants, longhair, coat). `yolov8n-pose.pt` lives in `checkpoints/`.
    - **PA-100K model** (`_init_par_model`, `PAR_*`): PaddleClas PULC person_attribute (PP-LCNet, ~7MB, Apache-2.0), read by OpenVINO's Paddle frontend from `checkpoints/attributes/person_attribute_infer/` (auto-downloaded locally; `deploy/ec2-setup.sh` downloads it on the server, where `checkpoints/` is mounted read-only). `par_attrs` overrides male, hat, longpants, bag and adds shorts, skirt (Skirt&Dress), longcoat; backpack is the mean of both models. crossroad-0230 still gives long hair, long sleeves and the colour points. Ranking AUC on all 6,559 TVPReid tracklets with labels from the captions, crossroad -> PA-100K: female 0.80 -> 0.90, hat 0.68 -> 0.77, backpack 0.76 -> 0.79 (0.81 averaged), long trousers 0.83 -> 0.95, shorts 0.91 -> 0.95 (7 positives), skirt/dress -> 0.95, long coat 0.58 -> 0.68, bag 0.61 -> 0.74.
  - **Per track** (`SearchEngine._attributes`, run over the 5 preview crops):
    - The attribute probabilities are averaged into `Track.attrs`.
    - Pose keypoints locate the top garment (shoulders → hips) and the bottom garment (hips → knees). `_garment_regions` returns no bottom region when the hips are not visible.
    - `_color_hist` assigns each region's pixels to the nearest of the `COLOR_PROTOS` in L-down-weighted Lab space, giving `attrs["top_color"]` and `attrs["bottom_color"]` (colour → fraction).
  - **Query side.** `query_attributes(prompt)` parses gender words, colour+garment pairs (`TOP_GARMENTS`/`BOTTOM_GARMENTS`) and `ATTR_PATTERNS` into constraints `{key, value, weight}` (`CONSTRAINT_WEIGHTS`). There is no negation handling; it was removed on purpose.
    - The weights are small and **gender and top colour have weight 0** (re-chosen 2026-11-28, comment above `CONSTRAINT_WEIGHTS`). Measured through the demo's scoring on TVPReid (PRID/iLIDs val+test, 100 Duke val + 100 Duke test videos; choices on val): no constraints 59.9 val / 57.0 test mean R@1, crossroad defaults 58.8 / 54.8, PA-100K defaults 55.6 / 55.7, PA-100K without gender and top colour 60.2 / 57.2; 2x and 3x weights lose 5-20 points. The DSCA model is trained on these captions, so in-domain it already knows gender and colours; per-constraint ablation showed gender and top colour each costing about 1.3 points alone. Real footage (station clip): with no constraints the top-1 was never confidently of the wrong gender in 66 gender queries; on 46 extra station queries written from crop sheets (accessories, directions; `experiments/webdemo_eval/queries.py`, split in halves; `eval_real.py` re-runs everything) the chosen weights give 27/46 top-1 against 24/46 without constraints, GT15 15/15 against 14/15, TESTVIDEO 7/9 either way. Gains are a few queries, i.e. near the noise of these small sets.
    - "coat" (p ≈ 0.1 for everyone) and "wears long pants" (AUC 0.52) are not constraints; the first also made the attribute filter drop every track.
  - **Actions** (`act_stand`, `act_walk`, `act_run`, `act_sit`; `ACTION_PATTERNS`/`ACTION_WEIGHTS`, 0.10, sitting 0.08) are measured, not left to the text model: TVPReid captions mention walking in 57% of cases, running 0.8%, standing 0.3%, sitting 0.02%. `SearchEngine._actions` runs after stitching: `track_speeds` gives box-centre speed in box heights/s over 1 s windows (per contiguous run, skipping boxes at the frame border, nothing for tracks under `ACTION_MIN_TRACK_S`), `action_probs` turns it into probabilities (still < `STAND_MAX_SPEED` 0.13, running > 1.5), and sitting comes from pose keypoints in `_attributes` (`_sit_score`). Calibrated by eye on `ok.mp4` (34 standing people: median 0.06, max 0.125) and `TESTVIDEO.mp4` (far walkers: 0.13-0.9). No runner or sitting person was available, so those two are uncalibrated. Action constraints work without the attribute model.
    - **Direction** (`act_left`, `act_right`, `act_toward`, `act_away`; `DIRECTION_PATTERNS`, weight 0.10): `track_direction` fits least-squares slopes of the box centre (body heights/s), log box height and feet position per contiguous run off the frame border; `direction_probs` maps lateral slope (mid 0.10 bh/s) and an approach score (log-height growth + feet moving down) to probabilities. Left/right are as seen by the camera. Checked against drawn trajectories: all 16 measurable TESTVIDEO tracks and the most lateral station tracks. On 8 direction queries: 4/8 top-1 against 3/8 without constraints.
  - **Objects** (`suitcase`, and the detector's handbags folded into `bag`): YOLO also returns COCO suitcase/handbag boxes in the tracking pass (`OBJECT_CLASSES`). `_split_objects`, a detector callback registered before ultralytics' tracker callback, moves them out of the results (and their rows out of `result.feats`) so BoT-SORT sees exactly the person-only boxes; tracks stayed identical (18/18, 149/149) and analysis time did not change. After stitching, `object_fractions` gives each object box to the best-fitting visible person (`_object_fit`: bag mostly inside the box below the head; suitcase beside/in front of the legs, bottom near the feet) and `object_p` maps the fraction of the track's frames to a probability (sigmoid around 0.25 for suitcases, 0.20 for handbags). `attrs["bag"]` becomes the mean of the attribute model (`bag_attr`) and the handbag probability: ranking AUC on 23 eye-checked station tracks 0.82 -> 0.96 (the positives were partly picked from the detector's candidates, so optimistic). Suitcase: 7 of 8 eye-checked carriers got p >= 0.9, one occluded suitcase was missed, one passer-by got 0.57. `SUITCASE_PATTERN` (longer phrases first, so "trolley bag" does not also ask for a handbag) adds the `suitcase` constraint without the attribute model, weight 0.15 (picked on the same clip). Tried and dropped: bicycles (too rarely detected next to the owner), an AVA SlowFast action model (only walk/stand/carry fired on this footage, +2.8 s CPU per second of video), CLIP zero-shot (3/8 vs 7/8 for suitcases). Strollers have no COCO class and no labelled footage.
  - **Scoring.** `search` adds per track `attr_p` (from `constraint_p`), `penalty = Σ weight·(1 − p)`, `attr_min` and `attr_fail`. Fused = weighted cosines − penalty. `SearchEngine.rank` and `app.js::rerank` both implement this, so keep them in sync.
  - **"Matching attributes only" filter** (re-designed 2026-11-28): hides tracks whose `attr_fail` is non-empty. `attr_filter_cuts` decides per constraint:
    - actions, directions, suitcase, long pants, long sleeves and long hair (`FILTER_ABS_KEYS`) fail below 0.5;
    - the other weighted constraints fail only in the bottom `FILTER_REL_PCT` (25) % of this video's scores;
    - gender, top colour and hat (`FILTER_SKIP_KEYS`) never filter.
    - Why: the old rule (any constraint with p < 0.5) kept the right person in only 25/70 labelled queries (top-1 49 -> 19/70, 6% of people left). The attribute scores rank well but are not calibrated on small distant people (backpack: the right person scored a median 0.42 yet was above 82% of the video). Gender and top colour are unreliable, and hat ranked at chance on this footage (8/8 people with caps failed).
    - New rule on the same 70 queries (English / Vietnamese / VI2): right person kept 67/70 each, top-1 50 / 51 / 50 (no filter 49 / 52 / 51), about 60% of people left. The three misses are an occluded suitcase, an occluded shoulder bag and wide striped trousers scored 0.38 on long pants. Chosen on these same queries; there is no held-out set.
  - **Fallbacks.** If openvino or the attribute model is missing, constraints are empty and ranking is pure DSCA. If the pose model is missing, colours fall back to the attribute model's colour points.
- **Gallery** (`webapp/store.py`, `webapp/supabase_client.py`, 2026-11-28): with `SUPABASE_URL` + `SUPABASE_SECRET_KEY` (env or a `.env` next to `web_demo.py`, loaded by `load_env_file`), analysed videos go to a shared gallery on Supabase: rows in `videos`/`analyses` and objects in the private bucket `videos` (`deploy/supabase/schema.sql`: RLS on, no policies, so only the server's secret key gets in; the browser never talks to Supabase).
  - **Analysis format.** An analysis is a zip of JSON + npz (`allow_pickle=False`) + preview JPEGs, without the model clips (`Track.clip`), produced by `dump_analysis` and read by `load_analysis`. Size: 0.44 MB for 18 tracks, 3.7 MB for 149, against 55 / 454 MB with the clips. A round trip gave identical search results on 140/140 labelled queries.
  - **Proxy storage.** Proxies over `STORE_PART_MB` (45) are stored in parts, because the Free plan caps one object at 50 MB.
  - **Re-encoding.** `SearchEngine.encode_tracks` re-crops clips from the proxy when a loaded analysis has none, e.g. after a checkpoint switch.
  - **Lookup and saving.** Analyse looks in memory, then in the gallery (same video, `params_key`, `TRACKER_VERSION`), and only then analyses. Saving runs in a second `JobManager` (`net_jobs`), so a transfer never waits behind an analysis.
  - **Re-analysis.** "Phân tích lại" (`reanalyse`) forces a fresh run. The stored analysis is replaced only once the new one is saved.
  - **Local cache.** `uploads/videos` becomes a disk cache trimmed to `CACHE_MAX_GB` and is not wiped at start in this mode.
  - **Limits.** Storage over `STORAGE_LIMIT_MB` (950) means the video is analysed but not kept. Nothing is deleted automatically.
  - **Supabase errors** become `SupabaseError`, returned as 503. The app keeps working when Supabase is down or paused.
  - **Measured** against a local stand-in: a stored analysis loads in about 1 s, versus 60 s to analyse 200 frames. A cache miss downloads a byte-identical proxy.
  - **No Supabase.** Without `SUPABASE_URL` the old in-memory mode is unchanged.
- **Admin** (`/admin`, `webapp/admin.py`, `webapp/monitor.py`): users need no login; only `/admin` has a password.
  - **Password.** The hash is scrypt from the standard library, made with `python -m webapp.admin hash-password`. It is read from `app_settings` on Supabase (the admin page can change it), else from `ADMIN_PASSWORD_HASH`.
  - **Session.** An HMAC-signed cookie valid for 12 h. Five wrong passwords from one IP within 10 min lock that IP out for 10 min.
  - **Page contents.** CPU/RAM/disk (psutil), jobs of both managers with IP and video plus cancel, people online (IPs seen within 5 min), blocked IPs, the gallery (delete, re-analyse), and the activity log (memory, plus Supabase `activity_log` kept 30 days and flushed every 5 s and at shutdown).
  - **Client IP.** uvicorn trusts `X-Forwarded-For` only with `TRUST_PROXY=1`, i.e. behind Caddy, where the app port is reachable only from Caddy.
  - **Public-site guards:**
    - one queued or running analysis per IP;
    - at most `MAX_QUEUE` (5) active jobs;
    - `UPLOADS_PER_HOUR` (10) uploads per IP;
    - blocked IPs get 403, except on `/admin`;
    - only the IP that started a job can cancel it through the public API.
- **Jobs** (`webapp/jobs.py`): prepare and analyse run on a single worker thread, and the UI polls `/api/jobs/{id}`. HTTP handlers are plain `def`, so they run in FastAPI's threadpool and never block the event loop. Checkpoint switching is refused while a job is busy. Embeddings are re-encoded lazily when the active checkpoint changes.
- **Storage**: files live in `uploads/videos/<id>/`, which is wiped at every server start because the index is in memory. Videos are addressed only by id; there is no path-based file access.
- **Checkpoints and architecture**: `SearchEngine.load_checkpoint` rebuilds the model from the config stored in the checkpoint, so both backbones load: `spacetime` (16 uniform frames) and `mfgf` (recipes m2s/m3: 4 spread frames + 16 consecutive frames at about 15 fps, picked in `_pick_clip_frames`). `Track.clip` is in the layout of the checkpoint that analysed the video; when a checkpoint with the other layout is selected, `encode_tracks` re-crops from the proxy video (`Analysis.video_path`). The decomposer follows the checkpoint's `dsca_mode` (`keyword` or `clause`).
  - Default: `PREFERRED_CHECKPOINT` = `checkpoints/m3c_joint_seed42.pth` when present (Kaggle notebook `dorisnguyen510/spacetime-dsca-m3c`, output `srccv/checkpoints/all/m3c_joint_seed42/best.pth`; uses CUHK-PEDES pre-training). Otherwise `best_weights.pth`, then any `best*`.
  - Measured through the demo's own scoring on test tracklets (PRID 142 / iLIDs 75 / Duke 100 videos, full captions, no attribute penalty, R@1): 63.0 / 38.0 / 77.5 for m3c joint against 7.4 / 14.7 / 59.5 for the Duke-trained `best_weights.pth`. CLIP alone scores 13 / 9 / 19 there but is strong on sharp real footage.
- **Startup**: `SearchEngine` passes `pretrained_vit=False` when a checkpoint exists, because the backbone weights would be overwritten anyway; this skips a 331MB download. If `bert-base-uncased` cannot be loaded, `TextEncoder` silently falls back to an untrained encoder, so checkpoints fail the full-match check in `SearchEngine.load_checkpoint`. The engine prints a warning when that happens.
- **UI**: `webapp/static/` is vanilla ES modules with no build step and no CDN. Strings live in `js/i18n.js` (VI/EN). Theme tokens are in `css/app.css`.
  - **Wide screens (≥ 1240 px).** The page is an app shell that fits the viewport; `body` does not scroll. There are three columns, each scrolling on its own: the steps in the sidebar; the player and timeline in the stage, where the player keeps the video's aspect ratio and the timeline takes the leftover height; and the ranking in `.panel`.
  - **Narrower layouts.** Until there is something to rank, the panel is hidden (`:has(#results[hidden])`) and the player gets its width. Between 1100 and 1239 px the panel sits under the player. Below that, the old single-column order applies (`display: contents`).
  - **Short screens.** At ≤ 820 px of height the best-match summary and the score note are hidden.
  - **Admin page.** `/admin` (`admin.html`, `css/admin.css`, `js/admin.js`) uses tabs (overview, gallery, activity, security) that each fill one screen.

### Gotchas

- `DAMRModel.forward` accepts several legacy call signatures: `(texts, frames)`, a batch dict, the 4-arg `(q_app, q_mot, keyframes, motion_frames)`, or kwargs. Several outputs are aliases: `T` = `T_full` and `V` = `V_video`. Keep these working, because `train.py` and `test.py` call it in different ways.
- `test.py` hardcodes the model hyperparameters (`embed_dim=768`, `vit_depth=6`, `vit_heads=12`, 16 frames, `spacetime` backbone), so it builds the wrong architecture for `mfgf` checkpoints; `experiments/common.load_model` and `webapp/engine.py` read the architecture from the checkpoint config.
- `models/motion_difference.py` and `models/visual_appearance.py` (CLIP-based) are legacy and not imported by the current pipeline.
- Checkpoints store `model_state_dict` and `history`. `paths.get_checkpoint_path` falls back to `best_damr.pth`, then to the root `checkpoints/`.
