# Cài đặt

Tài liệu gồm 6 phần:
1. Cài môi trường (Linux hoặc Windows).
2. Tải checkpoint.
3. Chạy web demo, huấn luyện hoặc đánh giá.
4. Deploy lên VPS bằng Docker.
5. Deploy lên Google Cloud có GPU.
6. Deploy lên AWS EC2 với tên miền DuckDNS, tự cập nhật mỗi lần push lên GitHub.

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
python web_demo.py --no_clip                         # tắt nhánh CLIP phụ (đỡ ~600 MB RAM)
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

## 4. Deploy lên VPS (Docker + Caddy)

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

# 2. Model: checkpoint tải thẳng từ release v2.0 trên VPS; các model phụ (~25 MB) chép từ máy local.
#    Phải có đủ file TRƯỚC khi chạy: checkpoints/ được mount read-only, và nếu thiếu yolov8n.pt
#    thì Docker tạo một THƯ MỤC trùng tên làm app lỗi.
mkdir -p checkpoints
curl -L -o checkpoints/best.pth https://github.com/NguyenChien510/pblcv/releases/download/v2.0/best.pth
#    (chạy trên máy local, trong thư mục repo)
rsync -avP checkpoints/yolov8n-pose.pt checkpoints/reid checkpoints/attributes user@VPS:~/pblcv/checkpoints/
rsync -avP yolov8n.pt user@VPS:~/pblcv/

# 3. Trên VPS: cấu hình (tên miền; gallery Supabase và mật khẩu /admin, xem mục 6H)
cd deploy
cp .env.example .env
nano .env                         # DOMAIN
bash configure-app.sh             # hỏi Supabase URL + secret key (bỏ trống = không có gallery) và mật khẩu /admin

# 4. Chạy
docker compose up -d --build
docker compose logs -f app        # chờ dòng "SpaceTime-DSCA Person Search"
```

Mở `https://<DOMAIN>`: người dùng không cần đăng nhập. Trang quản trị ở `https://<DOMAIN>/admin`, mật khẩu đã đặt ở bước 3.

- **Cập nhật code:** `git pull && docker compose up -d --build`.
- **Dừng:** `docker compose down`.
- Có gallery Supabase thì video đã phân tích được giữ lại và mở lại ngay; không có thì mỗi lần container khởi động lại, video và kết quả phân tích bị xóa.
- Phân tích chạy tuần tự trên một worker: nếu hai người phân tích cùng lúc, người sau phải chờ.
- **Thử trên máy local không cần HTTPS:** trong `.env` đặt `DOMAIN=:80` và `HTTP_PORT=8080`, rồi mở `http://localhost:8080`.

---

## 5. Deploy lên Google Cloud có GPU (Compute Engine, NVIDIA L4)

Dùng credit $300 của Google Cloud. Máy: `g2-standard-4` (1× L4 24 GB, 4 vCPU, 16 GB RAM), giá khoảng **$0.71/giờ khi bật**, trừ vào credit. Cách dùng: bật trước buổi demo, tắt khi xong.

Các lệnh `gcloud` chạy trên **máy local** (đã cài Google Cloud SDK); các lệnh còn lại chạy **trên VM** sau khi `gcloud compute ssh`.

### A. Chuẩn bị tài khoản (một lần, trên console.cloud.google.com)

1. **Nâng cấp billing:** Billing → Account overview → **Upgrade**. Không mất phí; credit còn lại vẫn được trừ trước. Tài khoản trial không dùng được GPU.
2. **Cảnh báo chi tiêu:** Billing → Budgets & alerts → tạo budget $300, ngưỡng 25/50/75/90%. Budget chỉ gửi email, **không** tự dừng máy.
3. **Bật API:** APIs & Services → bật **Compute Engine API**.
4. **Xin hạn mức GPU:** IAM & Admin → Quotas & system limits, xin tăng lên **1** cho:
   - `GPUs (all regions)`;
   - `NVIDIA L4 GPUs`, region `asia-southeast1` (Singapore).

   Chờ email duyệt (vài giờ đến vài ngày) rồi mới làm phần B.

### B. Tạo VM (máy local)

```bash
gcloud auth login
gcloud config set project <PROJECT_ID>
gcloud config set compute/zone asia-southeast1-b      # L4 có ở asia-southeast1-a và -b

# IP tĩnh: tên miền <ip>.sslip.io không đổi sau mỗi lần tắt/bật VM
gcloud compute addresses create pblcv-ip --region=asia-southeast1
gcloud compute addresses describe pblcv-ip --region=asia-southeast1 --format='value(address)'   # ghi lại IP

# Mở cổng web 80/443
gcloud compute firewall-rules create pblcv-web --allow=tcp:80,tcp:443 --target-tags=pblcv-web

# VM: L4 đi kèm machine type g2-standard-4
gcloud compute instances create pblcv-gpu \
  --machine-type=g2-standard-4 \
  --maintenance-policy=TERMINATE \
  --image-family=ubuntu-2404-lts-amd64 --image-project=ubuntu-os-cloud \
  --boot-disk-size=60GB --boot-disk-type=pd-balanced \
  --address=pblcv-ip --tags=pblcv-web
```

Báo `ZONE_RESOURCE_POOL_EXHAUSTED` nghĩa là zone đang hết L4: thử `--zone=asia-southeast1-a`, hoặc một region khác có L4 (xem bảng *GPU regions and zones* của Google). Báo lỗi quota thì bước A4 chưa được duyệt.

### C. Cài driver NVIDIA, Docker và NVIDIA Container Toolkit (trên VM)

```bash
gcloud compute ssh pblcv-gpu        # từ máy local

# Driver (script chính thức của Google). VM sẽ tự khởi động lại: SSH lại và chạy lại đúng lệnh này đến khi báo đã cài xong.
sudo apt-get update && sudo apt-get install -y build-essential libvulkan1
curl -L https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz -o cuda_installer.pyz
sudo python3 cuda_installer.pyz install_driver
nvidia-smi                          # phải thấy "NVIDIA L4"

# Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER       # thoát SSH rồi vào lại để có quyền docker

# NVIDIA Container Toolkit: cho container dùng GPU
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
docker run --rm --gpus all ubuntu nvidia-smi    # container cũng phải thấy "NVIDIA L4"
```

### D. Code và model

Trên VM:

```bash
git clone https://github.com/NguyenChien510/pblcv.git && cd pblcv
mkdir -p checkpoints
curl -L -o checkpoints/best.pth https://github.com/NguyenChien510/pblcv/releases/download/v2.0/best.pth   # 941 MB
```

Trên máy local, trong thư mục repo (các model phụ, khoảng 25 MB):

```bash
gcloud compute scp --recurse checkpoints/yolov8n-pose.pt checkpoints/reid checkpoints/attributes pblcv-gpu:~/pblcv/checkpoints/
gcloud compute scp yolov8n.pt pblcv-gpu:~/pblcv/
```

Phải có đủ các file này **trước** bước E: `checkpoints/` được mount read-only, và nếu thiếu `yolov8n.pt` thì Docker tạo một thư mục trùng tên làm app lỗi.

### E. Cấu hình và chạy (trên VM)

```bash
cd ~/pblcv/deploy
cp .env.example .env
nano .env                  # DOMAIN=<IP-tĩnh>.sslip.io
bash configure-app.sh      # gallery Supabase (tuỳ chọn) và mật khẩu /admin, xem mục 6H

docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build   # lần đầu build khoảng 10-15 phút
docker compose logs -f app     # chờ "Device : CUDA" và "Checkpoint : best.pth"
```

Mở `https://<IP-tĩnh>.sslip.io` (không cần đăng nhập; quản trị ở `/admin`). Góc trên giao diện phải hiện **CUDA · YOLOv8n**. Nếu hiện CPU: thiếu `-f docker-compose.gpu.yml`, hoặc phần C chưa xong.

**Kiểm tra:** upload `TESTVIDEO.mp4`, phân tích rồi tìm vài câu mô tả. Trên CPU, video này cho 18 người và mất khoảng 3 phút; trên L4 sẽ nhanh hơn nhiều. Thứ hạng có thể lệch nhẹ so với CPU do khác phần cứng.

### F. Bật/tắt hằng ngày (máy local)

```bash
gcloud compute instances stop pblcv-gpu     # sau demo: hết tính tiền GPU/CPU
gcloud compute instances start pblcv-gpu    # trước demo: container tự chạy lại, chờ khoảng 1-2 phút
```

- Khi VM tắt vẫn tốn một khoản nhỏ cho ổ đĩa 60 GB và IP tĩnh (vài đô mỗi tháng, trừ vào credit).
- Mỗi lần VM hoặc container khởi động lại, video đã upload và kết quả phân tích bị xóa.
- **Cập nhật code:** trên VM chạy `cd ~/pblcv && git pull`, rồi lặp lại lệnh `docker compose ... up -d --build` ở bước E.
- **Tiết kiệm hơn:** máy Spot (thêm `--provisioning-model=SPOT --instance-termination-action=STOP` khi tạo VM; cần quota `Preemptible NVIDIA L4 GPUs`) rẻ hơn nhiều, nhưng Google có thể thu hồi bất cứ lúc nào, nên không dùng cho buổi demo chính.

### G. Dọn dẹp trước khi hết credit hoặc hết hạn 90 ngày (máy local)

Tài khoản đã nâng cấp sẽ trừ thẳng vào thẻ khi credit hết. Xóa hết tài nguyên:

```bash
gcloud compute instances delete pblcv-gpu
gcloud compute addresses delete pblcv-ip --region=asia-southeast1
gcloud compute firewall-rules delete pblcv-web
```

Kiểm tra lại trong Billing → Reports rằng chi phí đã về 0.

---

## 6. Deploy lên AWS EC2 (c7i.4xlarge) + tên miền DuckDNS + CI/CD từ GitHub

Web đang chạy: **https://findperson.duckdns.org**
- Người dùng không cần đăng nhập. Trang quản trị: **https://findperson.duckdns.org/admin**. Ở đó xem được CPU/RAM, job và IP đang dùng, hàng đợi, thư viện video và nhật ký. Mật khẩu admin đặt bằng `deploy/configure-app.sh` (mục H) và không ghi trong repo vì repo public.
- Video đã phân tích được lưu vào thư viện chung trên Supabase (mục H). Chọn lại trong thư viện là có kết quả ngay, không phải phân tích lại.
- Chỉ truy cập được khi máy EC2 đang bật.

### Truy cập EC2 bằng SSH

Cần file key `pblcv.pem`, tải về lúc tạo máy, và máy phải đang bật. Tên miền DuckDNS luôn trỏ về IP hiện tại của máy, nên không cần tra IP.

```bash
chmod 400 ~/Downloads/pblcv.pem                                    # một lần: SSH từ chối key ai cũng đọc được
ssh -i ~/Downloads/pblcv.pem ubuntu@findperson.duckdns.org
```

Trên Windows (PowerShell) dùng cùng lệnh, chỉ đổi đường dẫn:
```powershell
ssh -i $HOME\Downloads\pblcv.pem ubuntu@findperson.duckdns.org
```

Gõ ngắn hơn bằng cách thêm vào `~/.ssh/config` trên máy local:
```
Host findperson
    HostName findperson.duckdns.org
    User ubuntu
    IdentityFile ~/Downloads/pblcv.pem
```
Từ đó chỉ cần `ssh findperson`, `scp file.mp4 findperson:~` hoặc `ssh findperson 'lệnh'`.

Lệnh hay dùng trên EC2:

| Việc | Lệnh |
| :--- | :--- |
| Trạng thái container | `cd ~/pblcv/deploy && docker compose ps` |
| Xem log app (Ctrl+C để thoát) | `cd ~/pblcv/deploy && docker compose logs -f app` |
| Khởi động lại app | `cd ~/pblcv/deploy && docker compose restart app` |
| Image đang chạy | `grep APP_IMAGE ~/pblcv/deploy/.env` |
| Đĩa, RAM | `df -h / && free -h` |
| DuckDNS lần cập nhật gần nhất | `journalctl -u duckdns.service -n 3 --no-pager` |
| Tắt máy từ bên trong (như *Stop* trên console) | `sudo shutdown -h now` |

Lỗi thường gặp:
- **`Connection timed out`:** máy đang tắt, hoặc vừa bật và DuckDNS chưa kịp cập nhật IP. Chờ khoảng 30 giây rồi thử lại.
- **`REMOTE HOST IDENTIFICATION HAS CHANGED`:** chỉ xảy ra khi đã tạo máy EC2 mới mà vẫn dùng tên miền cũ. Chạy `ssh-keygen -R findperson.duckdns.org` rồi SSH lại.
- **`permission denied ... docker.sock`:** đang dùng chính phiên SSH vừa chạy `ec2-setup.sh`. Thoát ra SSH lại, hoặc thêm `sudo` trước lệnh `docker`.

Mỗi lần push lên `main` có sửa code app, GitHub tự build image và cập nhật server:

```
git push ─► GitHub Actions: build Docker image ─► ghcr.io/nguyenchien510/pblcv-webdemo:<commit>
                         └─► SSH vào EC2 ─► deploy/remote-deploy.sh: pull image, chạy lại,
                                            chờ app trả lời; lỗi thì tự quay về image trước
```

| File | Vai trò |
| :--- | :--- |
| [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml) | Workflow build và deploy |
| [`deploy/ec2-setup.sh`](deploy/ec2-setup.sh) | Cài máy EC2, chạy một lần |
| [`deploy/remote-deploy.sh`](deploy/remote-deploy.sh) | GitHub chạy script này trên server mỗi lần deploy |

Repo public nên GitHub Actions và GitHub Container Registry (GHCR) miễn phí. Image không chứa checkpoint; model nằm trên ổ đĩa EC2, `ec2-setup.sh` tải về.

### Máy và chi phí

**`c7i.4xlarge`** có 16 vCPU (8 nhân thật Intel Sapphire Rapids, có AVX-512/VNNI/AMX) và 32 GB RAM. Ba điểm cần biết trước:

- **Đắt.** Ở Singapore khoảng **$0.82/giờ khi bật**, theo các trang tổng hợp giá; số chính xác xem ở trang giá EC2 của AWS. Bật 24/7 thì khoảng $600/tháng, nên **chỉ bật khi demo**. Mục C có hướng dẫn cài tự tắt khi máy rảnh.
- **Phải nâng cấp lên Paid plan.** Máy này không nằm trong danh sách của Free plan. Nâng cấp thì credit vẫn còn và được trừ trước; khi hết credit mới trừ vào thẻ. $100 credit dùng được khoảng 115 giờ chạy máy.
- **Phải xin tăng hạn mức vCPU.** Tài khoản mới thường chỉ được chạy 5 vCPU, trong khi máy này cần 16.

| Khoản | Giá ước tính (Singapore) |
| :--- | :--- |
| `c7i.4xlarge` | ~$0.82/giờ khi bật |
| IPv4 công khai | ~$0.005/giờ khi bật |
| Ổ đĩa 30 GB gp3 | ~$3/tháng, tính cả khi máy tắt |
| Tên miền DuckDNS, GitHub Actions, GHCR | miễn phí |

Muốn thử miễn phí trước thì làm y hệt các bước dưới với `m7i-flex.large` (2 vCPU, 8 GB, nằm trong Free plan). Máy này không cần bước A2 và A4, nhưng phân tích sẽ chậm hơn nhiều.

### 0. Code phải có trên GitHub

Workflow và hai script phải nằm trên nhánh `main` (đã push). Lần push đầu chưa có cấu hình EC2 nên workflow chỉ build image; deploy sẽ chạy ở mục E.

### A. Tài khoản AWS (một lần)

1. **Đăng ký và chọn region.** Đăng ký tại aws.amazon.com (cần thẻ Visa/Mastercard). Góc phải trên console, chọn region **Asia Pacific (Singapore) `ap-southeast-1`**; mọi bước sau làm trong region này.
2. **Nâng cấp Paid plan:** bấm *Upgrade plan* trên banner Free plan, hoặc vào *Billing and Cost Management*.
   - Nâng cấp bình thường thì giữ nguyên credit.
   - Credit chỉ mất ngay nếu nâng cấp bằng cách tham gia AWS Organizations hoặc Control Tower.
3. **Cảnh báo chi tiêu:** *Billing and Cost Management → Budgets → Create budget* → mẫu *Monthly cost budget*, ví dụ $50, nhập email.
   - AWS gửi email khi chi tiêu tới 85% và 100%, và khi dự báo vượt ngân sách.
   - Budget **chỉ gửi email, không tự tắt máy**.
4. **Xin hạn mức vCPU:** *Service Quotas → AWS services → Amazon Elastic Compute Cloud (Amazon EC2)* → tìm **Running On-Demand Standard (A, C, D, H, I, M, R, T, Z) instances** (mã `L-1216C47A`).
   - Nếu *Applied quota value* nhỏ hơn 16, bấm *Request increase at account level* và nhập **16**.
   - Nếu AWS hỏi lý do, ghi: "1 c7i.4xlarge instance for a university web demo".
   - Chờ duyệt, từ vài giờ đến vài ngày; có email báo. Máy đang tắt không bị tính vào hạn mức.

### B. Tên miền miễn phí DuckDNS (một lần)

1. Vào https://www.duckdns.org, đăng nhập bằng GitHub hoặc Google.
2. Ô *sub domain* gõ một tên, ví dụ `findperson`, rồi bấm *add domain*. Bạn được `findperson.duckdns.org`. Cột IP để gì cũng được, script sẽ tự cập nhật.
3. Chép **token** ở đầu trang. Giữ bí mật: ai có token đều đổi được IP của tên miền này.

Ghi chú:
- Tên miền miễn phí chỉ có dạng `<tên>.duckdns.org`.
- `duckdns.org` nằm trong Public Suffix List, nên mỗi tên con có hạn mức chứng chỉ Let's Encrypt riêng; Caddy tự xin HTTPS bình thường.
- Không cần Elastic IP: mỗi lần máy bật, script trên máy tự trỏ tên miền về IP mới, nên tránh được phí IP tĩnh trong lúc máy tắt.

### C. Tạo máy EC2 (console)

1. **Tạo máy:** EC2 → **Launch instance** (chỉ làm được khi hạn mức ở bước A4 đã ≥ 16):

   | Mục | Chọn |
   | :--- | :--- |
   | Name | `pblcv-demo` |
   | AMI | **Ubuntu Server 24.04 LTS**, 64-bit (x86) |
   | Instance type | **`c7i.4xlarge`** |
   | Key pair | **Create new key pair**: ED25519, `.pem`; trình duyệt tải về `pblcv.pem`, giữ kỹ |
   | Network settings | Tạo security group mới: bật **Allow SSH**, **Allow HTTPS**, **Allow HTTP** từ *Anywhere* |
   | Storage | **30 GiB gp3** |

   SSH phải mở cho *Anywhere* vì máy chạy GitHub Actions không có IP cố định. Ubuntu trên EC2 chỉ cho đăng nhập bằng key, không dùng mật khẩu.

   Nếu Singapore không có `c7i.4xlarge`, chọn region khác và làm lại bước A4 ở region đó, vì hạn mức tính theo từng region.
2. **Tự tắt khi rảnh** (nên làm, tránh quên tắt máy): chọn máy → *Actions → Monitor and troubleshoot → Manage CloudWatch alarms* → *Create an alarm*:
   - *Alarm action*: **Stop**;
   - *Alarm thresholds*: **Average CPU utilization ≤ 2%**, **24 consecutive periods of 5 minutes**.

   Máy sẽ tự tắt sau 2 giờ liên tục không phân tích video. Lúc chỉ xem kết quả, CPU gần như 0%, nên nếu buổi demo dài thì nới thời gian ra.

### D. Cài máy (một lần)

Trên máy local, lấy *Public IPv4 address* của máy ở trang Instances:

```bash
chmod 400 ~/Downloads/pblcv.pem
ssh -i ~/Downloads/pblcv.pem ubuntu@<PUBLIC_IP>
```

Trên EC2:

```bash
curl -fsSL https://raw.githubusercontent.com/NguyenChien510/pblcv/main/deploy/ec2-setup.sh -o ec2-setup.sh
bash ec2-setup.sh findperson.duckdns.org
```

Script hỏi **token DuckDNS**, rồi ở cuối hỏi **Supabase URL + secret key** và **mật khẩu /admin** (`deploy/configure-app.sh`; key và mật khẩu không hiện lên màn hình). Làm mục H1 trước để có sẵn project Supabase, hoặc bỏ trống URL rồi cấu hình sau. Mất khoảng 5 phút, và làm các việc sau:
- cài Docker và thêm 4 GB swap;
- clone repo vào `~/pblcv`;
- tải model (~960 MB): checkpoint `best.pth` từ release v2.0, YOLO, pose, re-ID và thuộc tính;
- cài dịch vụ cập nhật DuckDNS: chạy lúc khởi động và mỗi 5 phút;
- ghi `deploy/.env` với tên miền, cấu hình gallery và hash mật khẩu admin;
- tạo SSH key riêng cho GitHub Actions.

Cuối cùng script in ra giá trị cần điền vào GitHub và **private key**, rồi xoá private key khỏi server. Lỡ mất key thì chạy lại script để tạo key mới.

### E. Nối GitHub với EC2 và deploy lần đầu

1. **Thêm cấu hình:** repo trên GitHub → **Settings → Secrets and variables → Actions**:

   | Tab | Tên | Giá trị |
   | :--- | :--- | :--- |
   | Variables | `EC2_HOST` | `findperson.duckdns.org` |
   | Variables | `EC2_USER` | chỉ cần nếu không phải `ubuntu` |
   | Secrets | `EC2_SSH_KEY` | toàn bộ private key script in ra, gồm cả dòng `BEGIN`/`END` |

2. **Chạy deploy:** tab **Actions** → *Build and deploy web demo* → **Run workflow**.
   - Job `build` dùng lại cache, nên nhanh.
   - Job `deploy` lần đầu phải kéo image vài GB về EC2, mất vài phút. Sau đó nó chờ app trả lời mới báo xanh.
3. **Mở web:** vào `https://findperson.duckdns.org` (không cần đăng nhập); trang quản trị ở `/admin`. Caddy tự xin chứng chỉ HTTPS ở lần truy cập đầu.

Server đăng nhập GHCR bằng token tạm của chính job, nên image để private cũng kéo được, không cần làm gì thêm.

### F. Dùng hằng ngày

- **Trước demo:** EC2 → Instances → *Instance state → Start*.
  - Khoảng 15 giây sau khi máy bật, DuckDNS trỏ tên miền về IP mới.
  - Container tự chạy; chờ 1–2 phút để app nạp model.
- **Sau demo:** *Instance state → Stop*, hoặc để alarm ở bước C2 tự tắt. Khi máy tắt, chỉ còn ổ đĩa tính tiền.
- **Cập nhật code:** push lên `main` là xong, xem tiến trình ở tab Actions.
  - Workflow chỉ chạy khi commit sửa `webapp/`, `models/`, `data/`, `utils/`, `configs/`, `web_demo.py`, `requirements.txt`, `Dockerfile`, `deploy/` hoặc chính workflow. Sửa README hay bài báo thì không deploy.
  - App không trả lời trong 5 phút thì workflow báo lỗi kèm log, và server tự chạy lại image trước đó.
  - Push lúc máy đang tắt thì job deploy báo lỗi; bật máy rồi bấm *Re-run failed jobs*.
- **Dữ liệu khi khởi động lại:** có gallery Supabase thì video đã lưu vào thư viện và kết quả phân tích vẫn còn, bản sao trên đĩa nằm trong volume `app_cache`. Không cấu hình Supabase thì mọi thứ bị xoá mỗi lần container khởi động lại.
- **Model mới thêm sau khi đã cài máy:** `checkpoints/` trên server chỉ đọc với app, nên model mới phải tải bằng tay. Model thuộc tính PA-100K (thêm ngày 28/11/2026; thiếu nó app vẫn chạy với model cũ):
  ```bash
  cd ~/pblcv/checkpoints/attributes && curl -fL -o par.tar https://paddleclas.bj.bcebos.com/models/PULC/inference/person_attribute_infer.tar && tar -xf par.tar && rm par.tar
  cd ~/pblcv/deploy && docker compose restart app
  ```
- **Xem log và các lệnh quản trị khác:** xem mục *Truy cập EC2 bằng SSH* ở đầu mục 6.
- **Đổi mật khẩu admin hoặc cấu hình Supabase:** SSH vào EC2 rồi chạy `bash ~/pblcv/deploy/configure-app.sh`. Script hỏi lại mọi thứ (mật khẩu không hiện lên màn hình) và khởi động lại app. Đổi mật khẩu cũng làm được ngay trong `/admin`, tab *Bảo mật*.

### G. Theo dõi chi phí và dọn dẹp

- **Xem chi phí:** *Billing and Cost Management → Free Tier* (credit còn lại) và *Bills* (chi phí theo ngày).
- **Ví dụ:** 20 buổi demo, mỗi buổi 2 giờ, tức 40 giờ, tốn khoảng $33 + $3 ổ đĩa.

Khi không dùng nữa:
1. EC2 → *Terminate instance*, rồi kiểm tra *Volumes* không còn ổ đĩa nào.
2. Xoá alarm trong CloudWatch.
3. Xoá biến `EC2_HOST` trên GitHub để workflow chỉ build, không deploy nữa.
4. Nếu muốn, xoá tên miền trên duckdns.org.

### H. Thư viện video (Supabase) và trang /admin

Gallery lưu proxy video và kết quả phân tích lên Supabase (gói Free: 1 GB storage, file tối đa 50 MB). App tự cắt file lớn và giới hạn dung lượng ở 950 MB. Không cấu hình thì app vẫn chạy, chỉ không có thư viện.

1. **Tạo project (một lần):** vào https://supabase.com → **New project**, region **Southeast Asia (Singapore)**.
   - **SQL Editor** → **New query** → dán toàn bộ `deploy/supabase/schema.sql` → **Run**.
   - Kiểm tra: **Table Editor** có 5 bảng, **Storage** có bucket `videos` ở chế độ Private.
2. **Lấy thông tin:** **Project Settings → API Keys**.
   - Project URL: `https://<ref>.supabase.co`. Dán thừa đuôi `/rest/v1/` cũng được, script tự bỏ.
   - **Secret key** `sb_secret_...`; project cũ thì dùng key `service_role`. Key này có toàn quyền: không dán vào chat, không commit.
3. **Điền vào server** (máy đã cài từ trước cũng làm bước này):
   ```bash
   bash ~/pblcv/deploy/configure-app.sh
   ```
   Script hỏi URL, secret key và mật khẩu `/admin`, ghi vào `deploy/.env` (quyền 600), kiểm tra kết nối Supabase rồi khởi động lại app. Dòng `Supabase: OK` nghĩa là đã nối được.
4. **Giữ project không bị tạm dừng:** Supabase gói Free tạm dừng project sau 1 tuần không có truy cập. Workflow `.github/workflows/supabase-keepalive.yml` gửi một truy vấn nhỏ mỗi ngày, kể cả khi EC2 tắt. Cần thêm trong GitHub → **Settings → Secrets and variables → Actions**:

   | Tab | Tên | Giá trị |
   | :--- | :--- | :--- |
   | Variables | `SUPABASE_URL` | `https://<ref>.supabase.co` |
   | Secrets | `SUPABASE_SECRET_KEY` | secret key ở bước 2 |

   Nếu project đã bị tạm dừng: vào dashboard Supabase bấm **Restore**. Trong lúc đó app vẫn chạy, chỉ thư viện không dùng được và `/admin` báo lỗi.

Máy đã cài trước ngày 28/11/2026: sau lần deploy có commit này, web hết hỏi mật khẩu Caddy ngay. Gallery và `/admin` chỉ bật sau khi chạy bước 3. Các dòng `AUTH_USER`/`AUTH_HASH` cũ trong `deploy/.env` không còn tác dụng, để nguyên cũng không sao.
