# Hướng dẫn: recipe `m3` – train lại iLIDs (3 seed) và ablation PRID trên Kaggle 2×T4

Tài liệu này dành cho model/người thực thi. Đọc hết trước khi sửa code. Việc gồm 3 phần:
(1) thêm recipe `m3` vào code, (2) tạo notebook `paper_kaggle_m3.ipynb`, (3) chạy trên Kaggle và tổng hợp.

## 0. Bối cảnh và mục tiêu

Cấu hình trong bài hiện tại là `m2s` (`experiments/plan.py`): Space-Time ViT 4 khung + S3D 16 khung + BERT,
ViT khởi tạo từ timm ImageNet-21k (`PBL_NO_FROZEN=1`). Kết quả test thật (`results/v6/runs`, giao thức MFGF):

| | R@1 | MFGF |
|---|---|---|
| PRID (3 seed) | 46,0 ± 4,3 | 32,3 |
| Duke (3 seed) | 47,6 ± 2,5 | 35,2 |
| **iLIDs (3 seed)** | **27,1 ± 5,4** (32,0 / 28,0 / 21,3); R@5 60,4; R@10 76,0 | 30,0; 71,7; 91,7 |
| PRID ablation seed 42 | full 42,3; no_init 21,8; **no_temporal 45,1; no_dsca 51,4; no_mim 49,3** | |

Mục tiêu: (a) iLIDs cao hơn MFGF; (b) trên PRID, bỏ thành phần nào cũng thấp hơn mô hình đầy đủ.

### Quy tắc bắt buộc (không được vi phạm)

1. Mọi lựa chọn (checkpoint, trọng số chấm điểm, β, chọn `m3` hay `m3j`, giữ hay bỏ một thay đổi) chỉ dựa trên
   **validation**. Tập test chỉ chấm **một lần** cho cấu hình đã chốt, và chấm đủ cả 3 seed.
2. Không chọn seed, không bỏ seed xấu, không chạy lại một seed vì số test thấp, không sửa số bằng tay.
3. Không ai bảo đảm trước được "100% hơn MFGF" hay "ablation luôn thua": test iLIDs chỉ có 75 video
   (1 truy vấn = 1,33 điểm), độ lệch chuẩn qua seed trên PRID là 4,3 điểm. Nếu kết quả cuối không đạt mục tiêu,
   **báo cáo đúng số đó** cho người dùng và dừng lại; không tìm cách khác để ra số đẹp trên test.
4. Mọi cờ mới mặc định tắt: không truyền cờ thì `train.py` và `run_revision.py` cho đúng kết quả `m2s` cũ.

## 1. Vì sao các thành phần đang thua, và cách sửa (recipe `m3`)

`m3 = m2s + 6 cờ mới`. Mỗi cờ sửa một nguyên nhân cụ thể tìm thấy trong code.

| # | Nguyên nhân | Cờ mới | Nơi sửa |
|---|---|---|---|
| 1 | `RandomHorizontalFlip` nằm trong transform của từng khung → một clip trộn khung lật và không lật; lật còn mâu thuẫn với "left/right" trong câu | `--clip_flip` | `data/dataset.py` |
| 2 | `Q_mot` gần như giống nhau giữa các video ("A video of a woman walking") nhưng InfoNCE coi là mẫu âm → `L_mot` kẹt ở ≈1,4 và đẩy nhiễu vào `V_mot`, `V_video` | `--text_pos` | `loss/criterion.py`, `train.py` |
| 3 | `Q_app` là túi từ khóa rời ("short, black, hair", mất "book") → ít thông tin hơn câu gốc, nên bỏ DSCA lại thắng | `--dsca_mode clause` | `data/text_decomposer.py`, `data/dataset.py` |
| 4 | `timeattn`, `norm3`, `temporal_embed` là tham số mới (khởi tạo 0) nhưng bị xếp vào nhóm LR backbone 2e-5 → gần như không học, nên "full" ≈ "no_temporal" | `--temporal_head_lr` | `train.py` (đoạn chia `pretrained_params` / `head_params`) |
| 5 | `L_MIM = |cos(V_app, V_mot)|` trên vector thô, trong khi `T_app` và `T_mot` (cùng BERT, cùng chủ thể) rất giống nhau → ép trực giao mâu thuẫn trực tiếp với hai InfoNCE | `--mim_mode centered` | `models/mim_loss.py`, `models/damr_main.py` |
| 6 | Val nhỏ (iLIDs 37 video) → checkpoint chọn theo đỉnh nhiễu | `--ema 0.99` | `train.py` |

Chi tiết cài đặt:

1. **`--clip_flip`** (`TVPRDataset`, tham số `clip_flip`): bỏ `RandomHorizontalFlip` khỏi transform train; trong
   `__getitem__` rút một lần `flip = random.random() < 0.5`, lật **tất cả 20 khung** bằng `torch.flip(frames, dims=[-1])`.
   Khi lật, đổi `left`↔`right` trong caption (regex `\b(left|right)\b`, giữ hoa/thường) **trước khi** gọi
   `decomposer.decompose`. Chỉ áp dụng lúc train.
2. **`--text_pos`**: `DAMRCriterion.forward` nhận thêm `app_keys`, `mot_keys` (list chuỗi). Luồng `common` giữ
   `video_keys` như cũ; luồng `app` dùng khóa = chuỗi `q_app` chuẩn hóa (`lower().strip()`), luồng `mot` dùng khóa =
   chuỗi `q_mot`. `InfoNCELoss` đã có sẵn multi-positive qua `keys` (hàm `multi_pos_ce`) – chỉ cần truyền đúng khóa
   cho từng luồng (dùng lại `_keys_tensor`, mỗi luồng một bảng id riêng). `train.py`: truyền `batch["q_app"]`,
   `batch["q_mot"]`. Khi `--no_dsca` thì ba luồng cùng dùng `video_keys`.
3. **`--dsca_mode clause`** (mặc định `keyword` = hành vi cũ). `TextDecomposer(mode="clause")`:
   - tách caption thành mệnh đề theo `.`, `;`, `,` và `while / and then / then`;
   - mệnh đề chứa **từ** thuộc `MOTION_VERBS_EN ∪ MOTION_DIRECTIONS_EN` (so khớp theo từ, không theo chuỗi con) → vào Q_mot;
   - mệnh đề chứa từ thuộc `CLOTHING_ITEMS_EN ∪ COLORS_EN ∪ APPEARANCE_TRAITS_EN` → vào Q_app;
     một mệnh đề có thể vào **cả hai**; mệnh đề không khớp nhóm nào → Q_app;
   - giữ nguyên văn mệnh đề (không rút thành từ khóa): `Q_app = "A {subject}: " + "; ".join(app_clauses)`,
     `Q_mot = "A video of a {subject}: " + "; ".join(mot_clauses)`; nếu một bên rỗng thì dùng fallback cũ.
   - `test.py`, `webapp/engine.py`, `experiments/common.py` (encode) phải dùng cùng `dsca_mode` với checkpoint:
     lưu `dsca_mode` vào `cfg["model"]` trong checkpoint, `common.load_model` đọc ra `info["dsca_mode"]`,
     `build_split_dataset` và `encode_bank` truyền xuống dataset. Thiếu bước này thì val/test encode sai prompt.
4. **`--temporal_head_lr`**: trong vòng chia nhóm tham số, tên chứa `timeattn`, `norm3` hoặc `temporal_embed`
   → `head_params` (LR 1e-4). Không áp dụng khi nạp được trọng số Frozen-in-Time thật (recipe `m2f`).
5. **`--mim_mode centered`** (mặc định `raw`): trừ trung bình theo batch của `V_app` và `V_mot` (trên vector đã chuẩn
   hóa), chuẩn hóa lại rồi mới lấy `mean(|cos|)`. Với batch 1 thì dùng `raw`. `DAMRModel` nhận `mim_mode` qua kwargs
   và lưu trong `cfg["model"]`.
6. **`--ema DECAY`** (0 = tắt): giữ bản EMA của `model.state_dict()` (chỉ tensor float), cập nhật sau mỗi
   `optimizer.step()` với `d = min(DECAY, (1 + n) / (10 + n))`. Mỗi epoch: đánh giá val bằng trọng số EMA; `best.pth`
   lưu `model_state_dict` = EMA (thêm `raw_state_dict` trong `latest.pth` để `--resume` tiếp tục đúng cả hai bản).
   Dùng `0.99` (iLIDs chỉ ≈30 bước/epoch; PRID ≈57).

Sau khi cài: smoke test CPU với dữ liệu giả (không có `captions/` thì `TVPRDataset` tự sinh 32/16 mẫu):

```bash
source .venv/bin/activate
python train.py --sub_dataset TVPReid-PRID --epochs 1 --batch_size 4 --tag smoke_m3 --backbone mfgf --no_pretrained_vit \
  --unique_video_batches --learnable_tau --weight_decay 0.05 --select_metric mean_r1_5_10 \
  --clip_flip --text_pos --dsca_mode clause --temporal_head_lr --mim_mode centered --ema 0.99
python train.py --sub_dataset TVPReid-PRID --epochs 1 --batch_size 4 --tag smoke_m2s --backbone mfgf --no_pretrained_vit   # cờ tắt vẫn chạy
```

Kiểm tra thêm bằng tay: in `decompose()` của 5 caption thật (ví dụ câu trong `experiments/paper/fig_architecture.py`)
ở cả hai `mode`; với `--clip_flip`, 20 khung của một mẫu phải cùng lật.

### `experiments/plan.py`

```python
M3_FLAGS = MFGF_FLAGS + ["--clip_flip", "--text_pos", "--dsca_mode", "clause", "--temporal_head_lr",
                         "--mim_mode", "centered", "--ema", "0.99"]
RECIPES.update({"m3": M3_FLAGS, "m3j": M3_FLAGS})
```

- `m3`: train riêng từng tập, số epoch như `EPOCHS_M2S` (PRID 30, iLIDs 30, Duke 20).
- `m3j` (chỉ để cứu iLIDs): một run chung `sub_dataset all` (train của cả 3 tập con, 5.329 video) **8 epoch**,
  tag `m3j_joint_seed42`, rồi fine-tune iLIDs **12 epoch** cho từng seed bằng `--init_from`. Cần tổng quát hóa
  `Run.init_run`, `Run.tag`, `Run.epochs` và `build_runs` (hiện gắn cứng `recipe="v2"`): run chung của `m3j` luôn là
  seed 42 (một mô hình chung, ba seed fine-tune) – ghi rõ điều này khi báo cáo.
- `ABLATION_RUNS` giữ nguyên; với `m3` thì `no_dsca` vẫn là `--no_dsca`, `no_mim` là `--lambda_mim 0`,
  `no_temporal` là `--no_temporal_attn`, `no_init` là `--no_pretrained_vit`.
- `run_revision.py`: `--recipe` nhận `m3`, `m3j`; `--stage eval` thêm `--grid_step 0.1` cho hai recipe này
  (lưới 0,05 đã overfit 74 truy vấn val của iLIDs: trọng số chọn cho 27,1 trong khi (1;0;0) cho 29,8).

## 2. Tạo notebook

1. Chép `paper_kaggle.ipynb` → `paper_kaggle_m3.ipynb` (sửa trực tiếp JSON; script dựng notebook cũ đã mất).
2. **Ô 2 (cấu hình)**: đặt `PHASE = "m3_s1"` và thêm hai nhánh `if` (dùng lại hàm `_rr` của nhánh `mfgf_abl3`;
   mọi lệnh train có tiền tố `env PBL_NO_FROZEN=1`; `eval` luôn kèm `--hub --grid_step 0.1`):

   **`m3_s1`** (phiên 1, `TIME_BUDGET_H, TRAIN_BUDGET_H = 11.3, 10.6`) – **chỉ in VAL** (`--val_only`):
   **ĐÃ CÀI** trong `paper_kaggle_m3.ipynb` (bản người dùng chốt ngày 11/11/2026):
   - GPU0: iLIDs `m3` seed 42/1234/2025 → encode → eval (val + test) → PRID `m3` seed 42 `no_dsca` → encode → eval.
   - GPU1: PRID `m3` seed 42 **full** (mốc so sánh của ablation) → `no_mim` → `no_temporal` → encode → eval.
   - `no_init` không train lại (dùng run m2s cũ); `m3j` chưa chạy.

   **`m3_s2`** (phiên 2, gắn Output phiên 1 làm Input):
   - GPU0: nốt run chung `m3j` → fine-tune iLIDs 3 seed → encode → eval `--val_only`
     → PRID `m3` seed 1234: full, no_dsca.
   - GPU1: PRID `m3` seed 1234: no_mim, no_temporal → seed 2025: full, no_dsca, no_mim, no_temporal.
   - `no_init` chỉ cần seed 42 (chênh ≈20 điểm, không cần lặp).

   **`m3_test`** (phiên cuối, không train): encode + eval **có test** cho đúng các run đã chốt ở mục 3.
3. **Ô 4**: thêm `m3_s1`, `m3_s2`, `m3_test` vào tuple `PHASE in (...)` của smoke test; trong đoạn smoke thêm một
   bước train 1 batch với đủ 6 cờ mới (gọi `train.py` trên 1 batch hoặc gọi trực tiếp criterion với `mot_keys`).
   Lệnh `--dry_run` kiểm tra cấu hình đổi sang `--recipe m3`.
4. **Ô 8**: thêm các phase mới vào tuple ablation (bỏ `best.pth` của run đã xong để Output < 20 GB).
5. **Ô 12**: in bảng VAL cho mọi run `m3*`; chỉ in TEST khi `PHASE == "m3_test"`.
6. Chạy thử cục bộ: `PBL_DRY_RUN=1 KAGGLE_ROOT=/tmp/kg jupyter nbconvert --execute` (hoặc chạy từng ô bằng python)
   để chắc notebook in đúng danh sách lệnh.

## 3. Chạy trên Kaggle

1. Đóng gói code: `python experiments/pack_for_colab.py` → `revision_code.zip`. Mở zip kiểm tra có `train.py`,
   `data/`, `loss/`, `models/` (kể cả `frozen_vit.py`, `mfgf_video.py`), `experiments/plan.py` bản mới.
2. Kaggle → Dataset `pblcv-code` → *New Version* → tải `revision_code.zip` lên.
3. Tạo notebook mới từ `paper_kaggle_m3.ipynb`. *Add Input*: `pblcv-code` (bản mới), dataset TVPReid
   (`dorisnguyen510/dataset`). Settings: **GPU T4 ×2**, **Internet On**. Tài khoản phải xác minh số điện thoại.
4. **Save Version → Save & Run All (Commit)**. Không chạy tương tác.
5. Xong phiên: tab *Output* → tải `kaggle_results_<thời điểm>.zip` → giải nén vào `results/v7/`. Đọc
   `logs/status.json`: việc `partial` thì chạy tiếp ở phiên sau (gắn Output phiên trước làm Input).

### Quy tắc chốt cấu hình (làm ở máy local, chỉ đọc số VAL)

Tiêu chí: trung bình qua 3 seed của `mean(R@1, R@5, R@10)` trên val, giao thức 2 câu/video, đã chọn trọng số và β.

- **iLIDs**: so `m2s` (76,43 – đã có trong `results/ilids_m2s_vs_m2f_selection.json`), `m3`, `m3j`. Chọn giá trị cao
  nhất; ghi quyết định vào `results/ilids_m3_selection.json` (có `recorded_before_test: true` và thời điểm)
  **trước khi** chạy `m3_test`.
- **PRID ablation**: với mỗi thành phần X, tính `full − no_X` trên val (3 seed). Nếu một thành phần vẫn không
  dương trên val: **không chấm test**; báo người dùng để quyết định (sửa tiếp thành phần đó trên val, hoặc bỏ nó khỏi
  danh sách đóng góp của bài). Không thử nhiều biến thể rồi chỉ giữ biến thể đẹp trên test.
- Sau khi chốt mới chạy phase `m3_test` (một lần).

### Tổng hợp

- Sửa `experiments/bootstrap_m2s.py` để nhận tên recipe (thay tiền tố `m2s_`) và danh sách seed cho ablation
  (trung bình 3 seed, bootstrap cặp theo truy vấn), rồi chạy: `python experiments/bootstrap_m2s.py results/v7/results/runs results/bootstrap_ci_m3.json`.
- Báo cáo cho người dùng: bảng test iLIDs từng seed + trung bình ± ĐLC + KTC 95%, so với 30,0 / 71,7 / 91,7 / 4,7;
  bảng ablation PRID (trung bình 3 seed, chênh lệch + KTC). Ghi rõ cấu hình nào thắng trên val và mọi run đã chạy.

## 4. Việc phát sinh phải báo người dùng

- Ablation dùng `m3` nên hàng chính PRID của bài cũng phải là `m3` (3 seed đã có sau phiên 2). Để bài nhất quán,
  **Duke cũng cần train lại bằng `m3`** (3 seed × 20 epoch ≈ 9 h/seed → thêm 2–3 phiên). Nếu không train lại Duke,
  bài phải ghi rõ mỗi tập con dùng cấu hình nào.
- Nếu iLIDs chốt `m3j`: bài phải ghi mô hình được huấn luyện trước trên tập train của cả ba tập con (MFGF train
  riêng từng tập), và ba seed chỉ khác nhau ở bước fine-tune.
- Mục 3–4 của bài (mô tả DSCA, công thức MIM, LR, lật ngang, EMA) phải viết lại theo `m3`.
- Quota Kaggle 30 giờ/tuần: phiên 1 + phiên 2 ≈ 22 giờ.
