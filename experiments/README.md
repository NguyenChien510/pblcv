# Thực nghiệm bổ sung theo phiếu nhận xét DG25 (bản nộp lần 2)

Thư mục này chứa toàn bộ mã để tạo ra các số liệu mà hội đồng yêu cầu. Không có số liệu nào được điền sẵn: mọi ô
tô vàng trong `experiments/paper/PAPER_SPACETIME_DSCA_v3_template.docx` chỉ được điền từ kết quả chạy thật (`results/tokens.json`).

## Phát hiện quan trọng trước khi chạy

- **Tập đánh giá cũ là tập validation, không phải tập test.** `train.py` đánh giá trên `captions/val.csv`
  (71/37/302 video), đúng bằng tập *Validate* trong Bảng 2 của bài MFGF. Tập test của MFGF là 142/75/603 video
  khác nhau (không phải bản sao của 71/37/302). Tập test (`captions/test.csv`) chưa từng được dùng → có thể đánh
  giá đúng giao thức MFGF mà **không cần huấn luyện lại**, và việc chọn epoch/trọng số trên validation trở thành
  quy trình đúng chuẩn.
- **Bảng 4 cũ ghép số từ nhiều epoch** (ví dụ PRID: R@1 của epoch 23, R@5 của epoch 30, R@10 của epoch 20/29,
  R@50 của epoch 8/11/17/27 — xem `reports/*/train_history.json`). Bản v3 chỉ báo cáo một checkpoint cho mỗi seed.
- **Bảng 6 (dòng 5: 40,14) và Bảng 7 (chỉ nhánh toàn thể: 38,03) mâu thuẫn nhau**: nếu dòng 6 chỉ đổi cách chấm
  điểm "không cần huấn luyện lại" thì dòng 5 phải bằng 38,03. Không tìm thấy log của các lần chạy ablation trong
  repo, nên toàn bộ ablation được chạy lại bằng các cờ mới.

## Ánh xạ yêu cầu → thực nghiệm

| # | Yêu cầu của hội đồng | Cách xử lý | Lệnh |
|---|---|---|---|
| 1 | So sánh cùng giao thức MFGF (1 câu/video, gallery 142/75/603) | Đánh giá tập test theo đúng giao thức; báo thêm giao thức 2 câu/video và câu thứ 2 | `--stage eval` |
| 2 | Chọn (w1, w2, w3) trên validation, test chỉ để báo cáo | Tìm kiếm lưới bước 0,05 trên val của từng lần chạy; test chạy một lần | `--stage eval` |
| 3 | ≥ 3 seed cho cả 3 tập con, mean ± std, bootstrap CI theo truy vấn | Seed 42/1234/2025; bootstrap 10.000 lần (theo truy vấn; theo video cho giao thức 2 câu) | `--stage train` + `aggregate` |
| 4 | Ablation độc lập + điền Bảng 5 | 4 lần huấn luyện bỏ từng thành phần + bỏ chấm điểm đa tầng; Bảng 5 lấy từ log | `--stage train` (PRID) |
| 5 | IDF1/MOTA hoặc ghi rõ demo; truy vấn tiếng Việt | Mục 6 ghi rõ là demo; hệ thống chỉ dùng câu tiếng Anh (ghi trong Hạn chế) | – |
| – | Tiêu chí 3: chia train/test theo ID có khớp giao thức gốc? | Đếm video/ID mỗi tập, ID dùng chung, so với Bảng 2 MFGF | `--stage split` |
| – | Tuyên bố "ít hơn 40% bộ nhớ" chưa có bằng chứng | Đo bộ nhớ đỉnh và độ trễ SpaceTimeViT vs ViT-B/16+S3D | `--stage bench` |

## Chạy trên Colab (GPU)

**Cách dễ nhất:** tải `revision_code.zip` (thư mục gốc `pblcv`) lên `MyDrive/pbl6remake/`, rồi mở `paper.ipynb` trên Colab và chạy từ trên xuống – notebook hướng dẫn từng bước (Phần A không cần train, B/C/D cần train, E điền bài). Tạo lại gói mã sau khi sửa script: `python experiments/pack_for_colab.py`.

Hoặc chạy tay lần lượt (thư mục repo trên Drive, dữ liệu ở `/content/dataset`):

```bash
python experiments/run_revision.py --stage split  --data_root /content/dataset   # đọc kỹ dòng "example ids"
python experiments/run_revision.py --stage train  --data_root /content/dataset   # lâu nhất, chạy lại được sau khi mất kết nối
python experiments/run_revision.py --stage encode --data_root /content/dataset
python experiments/run_revision.py --stage eval
python experiments/run_revision.py --stage bench
pip install sentencepiece sacremoses && python experiments/run_revision.py --stage mt --data_root /content/dataset
python experiments/run_revision.py --stage aggregate
python experiments/run_revision.py --stage figures
python experiments/fill_paper.py     # experiments/paper/PAPER_SPACETIME_DSCA_v3_template.docx -> results/PAPER_SPACETIME_DSCA_v3.docx
```

- **Số lần huấn luyện**: bắt buộc 10 lần – 2 seed mới × 3 tập con (seed 42 dùng lại `checkpoints/<tập con>/best.pth`
  của lần nộp trước nếu có, tắt bằng `--no_reuse_legacy`) + 4 lần ablation độc lập trên PRID
  (`--ablation_runs no_init no_temporal no_dsca no_mim`). Thêm 3 lần `cum1 cum2 cum3` nếu muốn đo lại Bảng 6
  (khuyến nghị). `--ablation_seeds 42 1234 2025` nếu muốn ablation cũng có 3 seed.
- **Thời gian**: log của `train.py` in `Elapsed` mỗi epoch; tổng ≈ thời gian/epoch × số epoch (PRID 30, iLIDs 50,
  Duke 30) × số lần chạy. Duke lớn nhất (3.920 video train), nên chạy PRID và iLIDs trước. Xem trước các lệnh bằng
  `--dry_run`; chỉ chạy một phần bằng `--subs TVPReid-PRID` hoặc `--seeds 1234`.
- **Dung lượng**: mỗi lần chạy cần ~4 GB khi đang huấn luyện. Mặc định, ngay khi một lần chạy xong, script mã hóa
  val/test rồi xóa `latest.pth`, bỏ trạng thái optimizer khỏi `best.pth` (~0,66 GB) và xóa checkpoint của các lần
  ablation (embedding đã lưu trong `results/emb`). Dùng `--no_slim` nếu muốn giữ tất cả.
- **Kết quả**: `results/tables.md` (đọc bằng mắt), `results/summary.json` (mọi số liệu), `results/tokens.json`
  (giá trị điền vào bài), `results/figures/*.png` (Hình 1, 3, 5).

## Thay đổi trong mã nguồn

- `train.py`: thêm `--seed`, `--tag` (thư mục riêng cho từng lần chạy, không bao giờ rơi về checkpoint khác),
  `--lambda_app/--lambda_mot/--lambda_mim`, `--no_temporal_attn`, `--no_dsca`, `--eval_weights`; seed cố định cả
  module `random`. Không truyền cờ nào thì huấn luyện giữ nguyên như trước.
- `models/spacetime_vit.py`, `models/damr_main.py`: tham số `use_temporal_attn` (mặc định `True`).
- Mã đánh giá mới khớp theo **video** (đúng giao thức text-to-video của MFGF); khớp theo person ID được lưu thêm
  (`ranks_pid`) để đối chiếu với log cũ.

## Kiểm tra cục bộ đã làm (CPU, dữ liệu giả)

Không có GPU và dữ liệu TVPReid trên máy này, nên chỉ kiểm tra đường chạy của mã: `train.py` 1 epoch với các cờ
ablation, `encode → eval → aggregate → figures → fill_paper` trên chỉ mục giả mà `TVPRDataset` tự sinh khi không có
chú thích. Các con số từ lần kiểm tra đó vô nghĩa và đã bị xóa.
