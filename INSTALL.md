# Cài đặt

Tài liệu gồm 4 phần:
1. Cài môi trường (Linux hoặc Windows).
2. Tải checkpoint.
3. Chạy web demo, huấn luyện hoặc đánh giá.
4. Deploy lên VPS bằng Docker.

Kiến trúc mô hình và kết quả xem [README.md](README.md).

Yêu cầu chung:
- **Python 3.12**, **Git**;
- khoảng 6 GB đĩa trống (thêm khoảng 4 GB cho mỗi lần huấn luyện);
- có mạng ở lần chạy đầu, vì code sẽ tải BERT, CLIP, YOLO và các model re-ID/thuộc tính.

Không có GPU vẫn chạy được web demo trên CPU. Huấn luyện thì cần GPU (các run trong bài dùng Kaggle 2×T4).

---

## 1. Cài môi trường

**Bắt buộc cài `torch` trước** từ index của PyTorch, nếu không pip sẽ kéo bản CUDA nặng vài GB. Máy không có GPU NVIDIA dùng `whl/cpu`; có GPU thì thay bằng `whl/cu126`.

### Linux / macOS
```bash
git clone https://github.com/NguyenChien510/pblcv.git
cd pblcv

python3.12 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Nếu Python hệ thống không có `pip`/`venv` (một số bản Ubuntu), dùng [uv](https://docs.astral.sh/uv/):
```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt \
    --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match
source .venv/bin/activate
```

### Windows (PowerShell)
```powershell
git clone https://github.com/NguyenChien510/pblcv.git
cd pblcv

py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

## 2. Checkpoint

Checkpoint (`*.pth`) không nằm trong git; tải từ GitHub release vào `checkpoints/`.

| File | Dung lượng | Release | Ghi chú |
| :--- | :---: | :--- | :--- |
| `best.pth` | ~941 MB | `v2.0` | **Khuyến nghị.** Recipe m3c (backbone `mfgf`, train chung 3 tập con, có pre-train CUHK-PEDES). Cùng file với `m3c_joint_seed42.pth` |
| `best_weights.pth` | ~659 MB | `v1.0` | Bản cũ, backbone `spacetime`, train trên Duke; yếu hơn nhiều trên PRID/iLIDs |

```bash
mkdir -p checkpoints
curl -L -o checkpoints/best.pth https://github.com/NguyenChien510/pblcv/releases/download/v2.0/best.pth
```
Trên Windows: `mkdir checkpoints`, rồi `curl.exe -L -o checkpoints\best.pth https://github.com/NguyenChien510/pblcv/releases/download/v2.0/best.pth`.

Web demo chọn checkpoint mặc định theo thứ tự:
1. `m3c_joint_seed42.pth`;
2. `best.pth`;
3. `best_weights.pth`;
4. file `best*` bất kỳ.

Có thể đổi checkpoint ngay trong giao diện.

Các model phụ của web demo **tự tải** ở lần chạy đầu, không cần làm gì thêm:

| Model | Vị trí lưu |
| :--- | :--- |
| `yolov8n.pt` | thư mục gốc repo |
| `yolov8n-pose.pt` | `checkpoints/` |
| re-ID OpenVINO | `checkpoints/reid/` |
| thuộc tính người OpenVINO | `checkpoints/attributes/` |
| BERT, CLIP | cache của Hugging Face |

---

## 3. Chạy

### Web demo
```bash
python web_demo.py                                   # tự mở http://127.0.0.1:8000
python web_demo.py --port 8080 --no-browser          # đổi cổng, không mở trình duyệt
python web_demo.py --samples_dir samples --no_clip   # thư mục video mẫu; tắt nhánh CLIP phụ
```
Dừng server bằng **Ctrl + C**.

Mỗi lần khởi động, server xóa `uploads/videos/`, vì index video chỉ nằm trong RAM. Trên CPU, mỗi người (track) mất khoảng 2 giây để mã hóa, nên video đông người sẽ phân tích lâu.

### Huấn luyện và đánh giá
Đặt dữ liệu TVPReid **cạnh** repo, tức `../dataset/TVPReid-PRID/captions/...`, hoặc truyền `--data_root`. Lệnh train recipe m3 và quy trình nhiều seed xem README mục 6.

```bash
PBL_NO_FROZEN=1 python train.py --sub_dataset TVPReid-PRID --backbone mfgf --tag m3_seed42 ...
python experiments/run_revision.py --stage eval --recipe m3 --grid_step 0.1
```

> `test.py` dựng cứng backbone `spacetime`, nên **không** dùng được với checkpoint `mfgf` (như `best.pth` của release v2.0). Với checkpoint `mfgf`, dùng `experiments/run_revision.py` hoặc web demo.

### Lỗi thường gặp
- **PowerShell chặn `Activate.ps1`:** chạy `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` một lần, rồi kích hoạt lại môi trường.
- **Cổng 8000 đang bị chiếm:** thêm `--port 8080`.
- **`No usable checkpoint` / `Architecture mismatch`:** file checkpoint tải chưa đủ dung lượng. Xóa file rồi tải lại.
- **Cảnh báo BERT không tải được:** khi đó encoder văn bản chạy với trọng số ngẫu nhiên và kết quả vô nghĩa. Hãy kiểm tra mạng hoặc cache Hugging Face.
- **Số liệu train/eval vô nghĩa:** kiểm tra `captions/` có được tìm thấy không. Nếu không, dataset âm thầm dùng dữ liệu giả (32 train / 16 val).
- **Log `[FrozenViT] timm ImageNet weights failed`:** chưa cài `timm`, nên ViT của backbone `mfgf` đang khởi tạo ngẫu nhiên. Chạy `pip install timm`.

---

## 4. Deploy lên VPS (Docker + Caddy, có mật khẩu)

Yêu cầu VPS:
- Ubuntu 22.04/24.04;
- **≥ 4 vCPU, 8 GB RAM, 25 GB đĩa**;
- mở cổng 80/443.

Có tên miền trỏ về IP VPS thì tốt; nếu không, dùng `<ip-vps>.sslip.io`.

Image chứa code, thư viện, BERT và CLIP. Image **không** chứa trọng số model: `checkpoints/` và `yolov8n.pt` nằm trên máy chủ và được mount vào container (xem `deploy/docker-compose.yml`).

```bash
# 1. Trên VPS: cài Docker và lấy code
curl -fsSL https://get.docker.com | sh
git clone https://github.com/NguyenChien510/pblcv.git && cd pblcv

# 2. Từ máy local: chép model sang VPS (~915 MB)
rsync -avP checkpoints/best.pth checkpoints/yolov8n-pose.pt checkpoints/reid checkpoints/attributes \
      user@VPS:~/pblcv/checkpoints/
rsync -avP yolov8n.pt user@VPS:~/pblcv/

# 3. Trên VPS: tạo mật khẩu và cấu hình
cd deploy
cp .env.example .env
docker run --rm caddy:2 caddy hash-password --plaintext 'mat-khau-cua-ban'   # dán kết quả vào AUTH_HASH
nano .env                                                                    # DOMAIN, AUTH_USER, AUTH_HASH

# 4. Chạy
docker compose up -d --build
docker compose logs -f app        # chờ dòng "SpaceTime-DSCA Person Search"
```

Mở `https://<DOMAIN>`; trình duyệt sẽ hỏi tài khoản `AUTH_USER` và mật khẩu đã tạo ở bước 3.

- **Cập nhật code:** `git pull && docker compose up -d --build`.
- **Dừng:** `docker compose down`.
- Mỗi lần container khởi động lại, video đã upload và kết quả phân tích bị xóa.
- Phân tích chạy tuần tự trên một worker: nếu hai người phân tích cùng lúc, người sau phải chờ.
- **Thử trên máy local không cần HTTPS:** trong `.env` đặt `DOMAIN=:80` và `HTTP_PORT=8080`, rồi mở `http://localhost:8080`.
