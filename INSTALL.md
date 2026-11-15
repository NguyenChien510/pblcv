# Cài đặt & chạy Web Demo (Windows)

Yêu cầu: **Python 3.12**, **Git**, khoảng 6 GB đĩa trống, có mạng ở lần cài đầu.

Mở **PowerShell** và chạy lần lượt:

```powershell
# 1. Clone
git clone https://github.com/NguyenChien510/pblcv.git
cd pblcv

# 2. Tạo môi trường ảo
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Cài thư viện (bắt buộc cài torch TRƯỚC)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 4. Tải checkpoint (~659 MB)
mkdir checkpoints
curl.exe -L -o checkpoints\best_weights.pth https://github.com/NguyenChien510/pblcv/releases/download/v1.0/best_weights.pth

# 5. Chạy
python web_demo.py
```

Trình duyệt sẽ tự mở **http://127.0.0.1:8000**. Lần chạy đầu sẽ tải thêm BERT, YOLO, YOLO-pose, model re-ID và model thuộc tính người (giới tính, balo, màu áo…) nên mất thêm 1–2 phút. Dừng server bằng **Ctrl + C**.

**Các lần sau** chỉ cần mở PowerShell và chạy:

```powershell
cd pblcv
.venv\Scripts\Activate.ps1
python web_demo.py
```

### Lỗi thường gặp
- **PowerShell chặn `Activate.ps1`:** chạy `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` một lần, sau đó kích hoạt lại môi trường.
- **Có GPU NVIDIA:** ở bước 3, thay `whl/cpu` bằng `whl/cu126`.
- **Cổng 8000 đang bị chiếm:** chạy `python web_demo.py --port 8080`.
- **Lỗi `No usable checkpoint` / `Architecture mismatch`:** file checkpoint chưa tải đủ 659 MB. Xóa file rồi tải lại ở bước 4.



.venv/bin/python web_demo.py --port 8000

---

# Deploy lên VPS (Docker + Caddy, có mật khẩu)

Yêu cầu VPS: Ubuntu 22.04/24.04, **≥ 4 vCPU, 8 GB RAM, 25 GB đĩa**, mở cổng 80/443. Có tên miền trỏ về IP thì tốt; không có thì dùng `<ip-vps>.sslip.io`.

Image chứa code, thư viện, BERT và CLIP. **Không** chứa trọng số model: `checkpoints/` và `yolov8n.pt` nằm trên máy chủ và được mount vào container (xem `deploy/docker-compose.yml`).

```bash
# 1. Trên VPS: cài Docker và lấy code
curl -fsSL https://get.docker.com | sh
git clone https://github.com/NguyenChien510/pblcv.git && cd pblcv

# 2. Từ máy local: chép model sang VPS (~915 MB)
rsync -avP checkpoints/m3c_joint_seed42.pth checkpoints/yolov8n-pose.pt checkpoints/reid checkpoints/attributes \
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

Mở `https://<DOMAIN>`, trình duyệt hỏi tài khoản `AUTH_USER` và mật khẩu ở bước 3.

- **Cập nhật code:** `git pull && docker compose up -d --build`.
- **Dừng:** `docker compose down`.
- Mỗi lần container khởi động lại, video đã upload và kết quả phân tích bị xóa (index nằm trong RAM).
- Phân tích chạy tuần tự trên một worker: hai người phân tích cùng lúc thì người sau phải chờ.
- **Thử trên máy local không cần HTTPS:** trong `.env` đặt `DOMAIN=:80` và `HTTP_PORT=8080`, rồi mở `http://localhost:8080`.
