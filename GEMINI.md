# 🧠 DAMR-LLM: Context & System Memory (GEMINI.md)

> **Mô hình**: DAMR-LLM (Disentangled Appearance-Motion Representation Learning for Text-to-Video Person Retrieval)  
> **Thư mục dự án**: `srccv/`  
> **Mục đích**: Lưu trữ ngữ cảnh, cấu trúc mã nguồn, công thức toán học và hướng dẫn chạy cho các phiên làm việc tiếp theo.

---

## 🏗️ 1. Cấu Trúc Mã Nguồn (`srccv/`)

```
srccv/
├── GEMINI.md                 # System memory & context snapshot (File này)
├── README.md                 # Tài liệu mô hình tổng quan
├── configs/
│   └── default.yaml          # Siêu tham số: embed_dim=512, K=4, L=16, lambda_mim=0.2, lr=1e-4
├── data/
│   ├── text_decomposer.py    # TextDecomposer: Tách Q -> (Q_app, Q_mot) bằng LLM/Rule-based
│   └── dataset.py            # TVPRDataset: Dataloader trích xuất K=4 keyframes & L=16 motion frames
├── models/
│   ├── text_encoder.py       # TextEncoder: Mã hóa Q_app -> T_app và Q_mot -> T_mot (512-D)
│   ├── visual_appearance.py  # VisualAppearanceEncoder: ViT/CLIP-ViT + Attention Pooling -> V_app
│   ├── motion_difference.py  # MotionDifferenceEncoder: Phép trừ ΔF_t + Diff-Trans -> V_mot
│   ├── mim_loss.py           # MutualInformationMinimizationLoss: Ép V_app ⊥ V_mot và T_app ⊥ T_mot
│   └── damr_main.py          # DAMRModel: Mô hình tổng hợp đầy đủ các nhánh
├── loss/
│   └── criterion.py          # DAMRCriterion: L_DAMR = L_app + λ_mot*L_mot + λ_mim*L_MIM + λ_cross_neg*L_cross_neg
├── utils/
│   ├── metrics.py            # Thước đo Rank@1, 5, 10, 50, MdR, mAP & Ma trận tương đồng luồng đôi
│   └── paths.py              # PathManager: Quản lý đường dẫn tập trung
├── train.py                  # Script huấn luyện chính (PyTorch AMP FP16 + Warmup + Cosine Annealing)
└── test.py                   # Script đánh giá benchmark & Truy vấn thử nghiệm câu tự nhiên
```

---

## 📐 2. Phép Toán & Công Thức Cốt Lõi (Standard KaTeX)

### 2.1. Module phân tách văn bản bằng LLM (Text Decomposition)
Câu truy vấn $Q$ được phân tách thành 2 sub-prompts:
- $Q_{app}$: Mô tả ngoại hình tĩnh (áo, quần, balo, phụ kiện, màu sắc)
- $Q_{mot}$: Mô tả hành động & hướng di chuyển (đi bộ, quay đầu, xách đồ)

Đưa qua Text Encoder trích xuất vector đặc trưng:

$$T_{app} = \text{TextEncoder}(Q_{app}) \in \mathbb{R}^{B \times D}$$

$$T_{mot} = \text{TextEncoder}(Q_{mot}) \in \mathbb{R}^{B \times D}$$

---

### 2.2. Module phân tách video (Disentangled Video Encoder)

#### A. Luồng Diện mạo Tĩnh (Appearance Stream - $V_{app}$)
Trích xuất $K=4$ keyframes đại diện $F_{key} = \{f_1, f_2, \dots, f_K\}$, đưa qua ViT/CLIP-ViT và Attention Pooling:

$$V_{app, k} = \text{ViT}(f_k) \in \mathbb{R}^{D}$$

$$V_{app} = \sum_{k=1}^K w_k \cdot V_{app, k} \in \mathbb{R}^{B \times D}$$

#### B. Luồng Hành động Động (Motion Difference Stream - $V_{mot}$)
Tính hiệu số khung hình liên tiếp ($L=16$ frames) để triệt tiêu 100% cảnh nền tĩnh và màu áo quần cố định ($=0$):

$$\Delta F_t = |F_{t+1} - F_t|, \quad t = 1, 2, \dots, L-1$$

Đưa chuỗi bản đồ chuyển động $\Delta F$ qua Temporal Difference Transformer (Diff-Trans):

$$V_{mot} = \text{DiffTrans}(\Delta F_1, \Delta F_2, \dots, \Delta F_{L-1}) \in \mathbb{R}^{B \times D}$$

---

### 2.3. Hàm mất mát Giảm thông tin tương hỗ (Mutual Information Minimization - MIM)
Ép không gian đặc trưng diện mạo và hành động vuông góc/độc lập với nhau:

$$\mathcal{L}_{MIM} = \frac{1}{B} \sum_{i=1}^B \frac{|V_{app, i} \cdot V_{mot, i}|}{\|V_{app, i}\| \|V_{mot, i}\|} + \frac{1}{B} \sum_{i=1}^B \frac{|T_{app, i} \cdot T_{mot, i}|}{\|T_{app, i}\| \|T_{mot, i}\|}$$

---

### 2.4. Hàm mất mát Tương phản Luồng đôi (Dual-Stream Contrastive Loss)

$$\mathcal{L}_{app} = \text{InfoNCE}(T_{app}, V_{app}; \tau)$$

$$\mathcal{L}_{mot} = \text{InfoNCE}(T_{mot}, V_{mot}; \tau)$$

$$\mathcal{L}_{cross\_neg} = \text{max}(0, \text{CosSim}(T_{app}, V_{mot})) + \text{max}(0, \text{CosSim}(T_{mot}, V_{app}))$$

---

### 2.5. Hàm mất mát Tổng thể (Overall DAMR Loss)

$$\mathcal{L}_{DAMR} = \mathcal{L}_{app} + \lambda_1 \mathcal{L}_{mot} + \lambda_2 \mathcal{L}_{MIM} + \lambda_3 \mathcal{L}_{cross\_neg}$$

Trong đó mặc định: $\lambda_1 = 1.0, \lambda_2 = 0.2, \lambda_3 = 0.1, \tau = 0.05$.

---

### 2.6. Điểm tương đồng luồng đôi khi Retrieval (Dual-Stream Scoring)

$$\text{Score}(Q_i, V_j) = 0.5 \cdot \text{CosSim}(T_{app, i}, V_{app, j}) + 0.5 \cdot \text{CosSim}(T_{mot, i}, V_{mot, j})$$

---

## 📊 3. So Sánh Với Paper Gốc MFGF (`rootpaper.pdf`)

| Tiêu chí | MFGF Paper gốc (`rootpaper.pdf`) | **DAMR-LLM (`srccv`)** |
| :--- | :--- | :--- |
| **Xử lý Văn bản** | Luật cứng NLTK đếm từ ($W_h = c_n / C$) | Phân tách LLM ($Q \rightarrow Q_{app}, Q_{mot}$) |
| **Xử lý Chuyển động** | Mô hình 3D-CNN (S3D) trên 16 frame gốc | Phép trừ khung hình ($\Delta F_t = \|F_{t+1} - F_t\|$) + Diff-Trans |
| **Kết hợp đặc trưng** | Nối thô `concat(f_vis, f_mot)` thành 1 vector $f_{ME}$ | Giữ riêng $V_{app}$ và $V_{mot}$, ép vuông góc $\mathbf{V}_{app} \perp \mathbf{V}_{mot}$ |
| **Hàm mất mát** | Contrastive $L_{common}$ + Chưng cất $D^2$ Space | Dual-Stream InfoNCE + MIM Orthogonality Loss + Cross-Penalty |

---

## 🛠️ 4. Hướng Dẫn Lệnh Chạy (Quick Execution Commands)

- **Huấn luyện (Train)**:
  ```bash
  python srccv/train.py
  ```

- **Đánh giá Benchmark (Eval Rank@1, Rank@5, MdR)**:
  ```bash
  python srccv/test.py --checkpoint srccv/checkpoints/best.pth
  ```

- **Truy vấn thử nghiệm 1 câu tự nhiên (Inference)**:
  ```bash
  python srccv/test.py --query "Người phụ nữ mặc áo khoác màu đỏ đi bộ qua cổng"
  ```
