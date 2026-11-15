"""
Builds the revised paper (v4) from the v3 template for the evaluated model: MFGF backbones (Space-Time ViT-B/16 on
4 discrete frames + S3D on 16 consecutive frames + BERT) with DSCA, L_MIM and multi-level scoring.

  1. structural edits on the template (drop the cumulative ablation and the resource paragraph, new equations 5-7,
     new figures 2, 4, 6),
  2. experiments/fill_paper.py with results/tokens.json (numbers from experiments/paper_tokens_m2s.py),
  3. text edits so every sentence describes the model and the numbers actually obtained.

    python experiments/paper/build_paper_m2s.py   # -> results/PAPER_SPACETIME_DSCA_v4.docx
"""

import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

REPO = Path(__file__).resolve().parents[2]
TEMPLATE = REPO / "experiments" / "paper" / "PAPER_SPACETIME_DSCA_v3_template.docx"
FIG = REPO / "results" / "figures"
OUT = REPO / "results" / "PAPER_SPACETIME_DSCA_v4.docx"
PARA = re.compile(r"<w:p[ >].*?</w:p>", re.S)

SENTENCES = {
    "ABS_COMPARE": "cao hơn MFGF rõ rệt trên PRID (+13,7 điểm) và Duke (+12,4 điểm), với khoảng tin cậy bootstrap 95% nằm "
                   "hoàn toàn trên giá trị công bố của MFGF; trên iLIDs, tập con nhỏ nhất, Rank@1 không khác biệt có ý nghĩa "
                   "so với MFGF và Median Rank tốt hơn (3,67 so với 4,7)",
    "ABS_ABLATION": "khởi tạo tiền huấn luyện của hai bộ mã hóa video là thành phần đóng góp lớn nhất (bỏ đi làm Rank@1 giảm "
                    "20,4 điểm)",
    "CMP_SENTENCE": "Trên PRID và Duke, KTC 95% của SpaceTime-DSCA ([39,4; 52,6] và [44,2; 51,0]) nằm hoàn toàn trên Rank@1 "
                    "công bố của MFGF; trên iLIDs, KTC [19,1; 35,6] chứa giá trị 30,0% nên hai phương pháp không phân biệt "
                    "được ở Rank@1.",
    "CMP_OTHER_SENTENCE": "SpaceTime-DSCA đạt Median Rank tốt hơn MFGF trên cả ba tập con (1,67; 3,67 và 2,00 so với 2,0; 4,7 "
                          "và 3,0), Rank@5 cao hơn trên PRID (+5,6) và Duke (+13,4), Rank@10 cao hơn trên PRID (+4,1) và tương "
                          "đương trên Duke (+0,5). Trên iLIDs, Rank@5 và Rank@10 thấp hơn MFGF (Hình 3).",
    "CONCL_COMPARE": "vượt MFGF rõ rệt trên PRID (+13,7 điểm) và Duke (+12,4 điểm), tương đương ở Rank@1 và tốt hơn ở Median "
                     "Rank trên iLIDs",
    "W_SENTENCE": "Trọng số chọn trên validation ưu tiên nhánh toàn thể trên PRID và Duke (w₁ từ 0,62 đến 0,77) và nhánh diện "
                  "mạo trên iLIDs (w₂ = 0,55). Trên tập test, Rank@1 của ba bộ trọng số chênh nhau không quá 2,8 điểm, nhỏ hơn "
                  "nhiều so với độ rộng KTC 95% (Bảng 4b), nên kết quả không nhạy với cách chọn trọng số; các số chính trong "
                  "Bảng 4 dùng bộ trọng số chọn trên validation.",
    "ABLL_SENTENCE": "Khởi tạo tiền huấn luyện là thành phần đóng góp lớn nhất: huấn luyện cả Space-Time ViT và S3D từ khởi "
                     "tạo ngẫu nhiên làm Rank@1 giảm 20,4 điểm (KTC 95% [−29,6; −11,3]) và Rank@5 giảm 32,4 điểm. Bỏ chú ý "
                     "thời gian (+2,8 [−4,2; 9,9]) hoặc chấm điểm đa tầng (−1,4 [−4,9; 2,1]) không làm Rank@1 thay đổi có ý "
                     "nghĩa. Bỏ phân rã DSCA hoặc ràng buộc ℒMIM làm Rank@1 tăng lần lượt 9,2 và 7,0 điểm, với KTC vừa loại "
                     "trừ 0, trong khi Rank@5 thay đổi không quá 2,8 điểm. Mô hình đầy đủ ở seed 42 đạt 42,3%, thấp hơn trung "
                     "bình ba seed (46,0%; Bảng 4b), nên với một seed duy nhất chúng tôi chỉ kết luận rằng trên PRID, DSCA và "
                     "ℒMIM không cải thiện độ chính xác ở hạng đầu. Một giải thích khả dĩ là khi hai bộ mã hóa đã được tiền "
                     "huấn luyện, chuỗi Qapp/Qmot sinh theo mẫu cố định mang ít thông tin hơn câu gốc, còn ràng buộc trực "
                     "giao hạn chế việc hai luồng cùng dùng tín hiệu màu sắc trang phục.",
    "SPLIT_SENTENCE": "Số video train/validation/test của cả ba tập con khớp chính xác Bảng 2 của [24] và không có video_id nào "
                      "xuất hiện ở hai phần. Bản phát hành gán cho mỗi video một mã riêng (person0001, …), nên số mã bằng số "
                      "video và dữ liệu không cho biết hai video có thuộc cùng một người hay không; vì vậy chúng tôi giữ nguyên "
                      "phép chia chính thức như [24] để kết quả so sánh được trực tiếp.",
    "SPLIT_LIMIT": "bản phát hành TVPReid không cung cấp mã danh tính, nên chưa kiểm chứng được phép chia có tách rời theo "
                   "người hay không",
    **{f"SPLIT_{s}_{k}": "–" for s in ("PRID", "ILIDS", "DUKE") for k in ("NID", "IDOVL")},
    "SPLIT_TOTAL_NID": "–",
}

# (paragraph text that must contain `old`, old, new); `old` may span runs (it is matched on the paragraph text)
EDITS = [
    ("song mã hóa diện mạo", "song mã hóa diện mạo và chuyển động bằng hai mạng tách rời (ViT và mạng tích chập 3D S3D), "
     "trích xuất", "song trích xuất"),
    ("Bài báo đề xuất kiến trúc", "(1) bộ mã hóa video SpaceTimeViT dựa trên cơ chế chú ý không–thời gian phân chia (Divided "
     "Space-Time Attention) với 6 khối Transformer, tách đặc trưng thành ba luồng diện mạo ",
     "(1) nhánh video hai luồng giữ các bộ mã hóa của MFGF – Vision Transformer với cơ chế chú ý không–thời gian phân chia "
     "(Divided Space-Time Attention) trên 4 khung hình rời rạc và S3D trên 16 khung hình liên tiếp – kèm ba đầu ra chuyên "
     "biệt cho diện mạo "),
    ("Hai bộ mã hóa tách rời", "Hai bộ mã hóa tách rời: ViT và S3D chạy song song trên hai tập khung hình khác nhau (4 khung "
     "rời rạc và 16 khung liên tiếp), nên diện mạo và chuyển động được mã hóa độc lập, không tương tác ở mức token, và hai "
     "mạng phải được huấn luyện, lưu trữ riêng. Chi phí bộ nhớ và thời gian của hai cách tiếp cận được đo trực tiếp ở mục 5.1.",
     "Hai luồng video chưa được căn chỉnh riêng: đặc trưng ViT (4 khung rời rạc) và S3D (16 khung liên tiếp) chỉ được ghép "
     "lại rồi căn chỉnh với câu truy vấn đầy đủ, nên không có tín hiệu giám sát nào buộc luồng diện mạo khớp với phần mô tả "
     "trang phục và luồng chuyển động khớp với phần mô tả hành động."),
    ("Bộ mã hóa video SpaceTimeViT:", "Bộ mã hóa video SpaceTimeViT: thay tổ hợp ViT + S3D bằng một Vision Transformer duy "
     "nhất với cơ chế chú ý không–thời gian phân chia [3] gồm 6 khối và 12 đầu chú ý. Phần không gian được khởi tạo từ "
     "ViT-B/16 tiền huấn luyện trên ImageNet-1K, phần thời gian được khởi tạo bằng 0 (zero-init); ba đầu ra chuyên biệt tạo "
     "vector diện mạo, chuyển động và toàn thể.",
     "Nhánh video hai luồng với ba đầu ra chuyên biệt: giữ nguyên hai bộ mã hóa của [24] – Space-Time ViT-B/16 (chú ý "
     "không–thời gian phân chia [3], 12 khối) trên 4 khung rời rạc và S3D [23] trên 16 khung liên tiếp – để so sánh công "
     "bằng, và bổ sung ba đầu ra tạo vector diện mạo, chuyển động và toàn thể phục vụ căn chỉnh phân tầng."),
    ("Giao thức đánh giá minh bạch", "ablation cộng dồn và ablation độc lập;", "ablation độc lập cho từng thành phần;"),
    ("SpaceTimeViT thống nhất (16 khung)", "SpaceTimeViT thống nhất (16 khung)",
     "Giữ ViT (4 khung) + S3D (16 khung) như [24]; thêm ba đầu ra chuyên biệt"),
    ("Chú ý thời gian + Bi-GRU", "Chú ý thời gian + Bi-GRU", "Như [24]"),
    ("Hình 2 trình bày kiến trúc", "(2) nhánh video với SpaceTimeViT và ba đầu ra chuyên biệt;",
     "(2) nhánh video hai luồng (Space-Time ViT và S3D) với ba đầu ra chuyên biệt;"),
    ("Hình 2: Kiến trúc tổng thể", "SpaceTimeViT xử lý 16 khung hình bằng chú ý thời gian và chú ý không gian xen kẽ, sau đó "
     "Attention Pooling, Bi-GRU và MLP Fusion tạo ba vector ",
     "Space-Time ViT-B/16 xử lý 4 khung rời rạc bằng chú ý thời gian và chú ý không gian xen kẽ, S3D xử lý 16 khung liên "
     "tiếp; hai đầu chiếu và MLP Fusion tạo ba vector "),
    ("Nhánh Video: SpaceTimeViT", "Nhánh Video: SpaceTimeViT", "Nhánh Video: Space-Time ViT và S3D"),
    ("3.2.1 Lấy mẫu khung hình", "Từ mỗi video có Nf khung hình, chúng tôi lấy mẫu T = 16 khung theo phân đoạn: video được "
     "chia thành 16 đoạn bằng nhau;",
     "Theo [24], mỗi video cung cấp hai tập khung hình. Luồng diện mạo nhận T = 4 khung rời rạc lấy theo phân đoạn: video "
     "được chia thành 4 đoạn bằng nhau;"),
    ("3.2.1 Lấy mẫu khung hình", "khung giữa đoạn được chọn để kết quả tất định.",
     "khung giữa đoạn được chọn để kết quả tất định. Luồng chuyển động nhận 16 khung liên tiếp, bắt đầu tại vị trí ngẫu "
     "nhiên khi huấn luyện và ở giữa video khi đánh giá."),
    ("3.2.2 Chú ý không–thời gian", "Mỗi khối DividedSpaceTimeBlock tách", "Mỗi khối trong 12 khối của Space-Time ViT [2] tách"),
    ("Ở bước chú ý thời gian", "qua 16 bước thời gian", "qua T = 4 bước thời gian"),
    ("Khởi tạo và tối ưu bộ nhớ", "MLP của 6 khối được nạp từ ViT-B/16 tiền huấn luyện trên ImageNet-1K [6].",
     "MLP của 12 khối được nạp từ ViT-B/16 tiền huấn luyện trên ImageNet-21k và tinh chỉnh trên ImageNet-1K [6], cũng là "
     "điểm khởi tạo của Frozen-in-Time [2]; S3D dùng trọng số tiền huấn luyện trên Kinetics-400, với BatchNorm giữ ở chế độ "
     "suy luận vì mỗi lô chỉ có 16 video."),
    ("Khởi tạo và tối ưu bộ nhớ", "Mỗi khối được bọc", "Mỗi khối của ViT và S3D được bọc"),
    ("3.2.3 Ba đầu ra chuyên biệt", "Từ chuỗi token đầu ra, chúng tôi lấy trung bình các patch trong mỗi khung để có đặc trưng "
     "khung ft ∈ ℝ768, tạo ma trận F = [f1, …, fT], rồi trích xuất ba vector.",
     "Từ đầu ra của hai bộ mã hóa, chúng tôi trích xuất ba vector."),
    ("(a) Vector diện mạo", "Attention Pooling động. Một mạng chú ý nhỏ (768 → 192 → 1) chấm điểm từng khung; khung rõ nét, "
     "ít bị che khuất nhận trọng số lớn hơn:",
     "token [CLS]. Ở mỗi khối, token [CLS] chú ý tới mọi patch của cả 4 khung, nên trạng thái cuối của nó tóm tắt diện mạo "
     "trên toàn đoạn video; trạng thái này được chiếu tuyến tính và chuẩn hóa:"),
    ("(b) Vector chuyển động", "Bi-GRU. Chuỗi đặc trưng khung được đưa qua mạng hồi quy hai chiều Bi-GRU [5] (768 → 2 × 384), "
     "thu được các trạng thái ẩn g1, …, gT. Vector chuyển động kết hợp độ biến thiên giữa trạng thái cuối và đầu với trạng "
     "thái trung bình, phản ánh sự thay đổi vận tốc và hướng bước đi:",
     "S3D. 16 khung liên tiếp được chuẩn hóa theo thống kê Kinetics và đưa qua phần trích đặc trưng của S3D [23]; bản đồ "
     "đặc trưng 1024 kênh được lấy trung bình theo thời gian và không gian rồi chiếu sang không gian 768 chiều:"),
    ("3.3.1 Căn chỉnh tương phản", "với nhiệt độ τ = 0,07:", "với nhiệt độ τ (khởi tạo 0,07 và được học cùng mô hình):"),
    ("với w1 + w2 + w3 = 1", "(0,50; 0,35; 0,15).",
     "(0,50; 0,35; 0,15). Ngoài ra, điểm của mỗi video trong gallery được trừ đi β lần độ “hub” của video đó (trung bình 10 "
     "độ tương đồng cao nhất với các câu mô tả của tập train, trừ đi giá trị trung bình trên gallery), với β ∈ {0; 0,1; 0,2; "
     "0,3; 0,5; 0,75; 1} chọn trên validation. Bước này không cần huấn luyện lại và hạn chế hiện tượng một số video xuất hiện "
     "ở đầu danh sách của rất nhiều truy vấn."),
    ("Vai trò của các tập", "Tập validation chỉ dùng cho hai quyết định: (i) chọn checkpoint là epoch có Rank@1 validation "
     "cao nhất", "Tập validation chỉ dùng cho ba quyết định: (i) chọn checkpoint là epoch có trung bình Rank@1, Rank@5 và "
     "Rank@10 validation cao nhất"),
    ("Vai trò của các tập", "khi dùng cả hai câu/video làm truy vấn.",
     "khi dùng cả hai câu/video làm truy vấn; (iii) chọn hệ số β giảm hub (mục 3.4) theo trung bình Rank@1, Rank@5 và Rank@10."),
    ("Đầu vào: T = 16 khung", "Đầu vào: T = 16 khung, 224 × 224,",
     "Đầu vào: 4 khung rời rạc cho Space-Time ViT và 16 khung liên tiếp cho S3D, 224 × 224,"),
    ("Mô hình: SpaceTimeViT 6 khối", "Mô hình: SpaceTimeViT 6 khối, 12 đầu chú ý, dropout 0,1;",
     "Mô hình: Space-Time ViT-B/16 12 khối, 12 đầu chú ý; S3D (Kinetics-400); dropout 0,1;"),
    ("Tối ưu hóa: AdamW", "weight decay 1 × 10−4;", "weight decay 0,05;"),
    ("Lô và số epoch", "huấn luyện 30 epoch trên PRID và Duke, 50 epoch trên iLIDs;",
     "mỗi lô không chứa hai câu của cùng một video; huấn luyện 30 epoch trên PRID và iLIDs, 20 epoch trên Duke;"),
    ("TB ± ĐLC: trung bình", "(xem summary.json)", "(results/bootstrap_ci.json)"),
    ("Hình 4: Động học huấn luyện", "vòng tròn đánh dấu giá trị cao nhất, tức epoch được chọn làm checkpoint",
     "vòng tròn đánh dấu epoch được chọn làm checkpoint (trung bình Rank@1/5/10 validation cao nhất)"),
    ("Hội tụ ổn định", "giảm từ 4,2–4,8 ở epoch đầu xuống dưới 1,0 ở epoch cuối",
     "giảm từ 3,3–4,4 ở epoch đầu xuống dưới 0,5 ở epoch cuối"),
    ("Luồng chuyển động hội tụ chậm hơn", "chỉ giảm từ 2,756 xuống 1,785, trong khi ℒapp giảm xuống 0,240.",
     "chỉ giảm từ 2,771 xuống 1,409, trong khi ℒapp giảm xuống 0,034."),
    ("Ràng buộc trực giao được duy trì", "luôn nhỏ hơn 0,05", "luôn nhỏ hơn 0,07"),
    ("Chỉ số xếp hạng (validation)", "Rank@1 trên PRID tăng từ 4,93% (epoch 1) lên 43,66% (lần đầu đạt ở epoch 23); iLIDs đạt "
     "cao nhất 45,95% ở epoch 43 và Duke 39,07% ở epoch 29; các epoch này được chọn làm checkpoint của seed 42. MdR giảm từ "
     "10,5–31 ở epoch đầu xuống 2,0 trên cả ba tập. Trong 10 epoch cuối, Rank@1 validation dao động trong khoảng "
     "30,99–43,66% (PRID), 36,49–45,95% (iLIDs) và 34,93–39,07% (Duke);",
     "Rank@1 trên PRID tăng từ 10,56% (epoch 1) lên 65,49% ở epoch 15; iLIDs đạt 47,30% và Duke 53,15%, cùng ở epoch 17; "
     "các epoch này được chọn làm checkpoint của seed 42. MdR giảm từ 7,5–10 ở epoch đầu xuống 1,0 (PRID, Duke) và 2,0 "
     "(iLIDs). Trong 10 epoch cuối, Rank@1 validation dao động trong khoảng 60,56–66,20% (PRID), 39,19–44,59% (iLIDs) và "
     "46,52–53,15% (Duke);"),
    ("Bảng 5: Giá trị các thành phần", "epoch được chọn theo Rank@1 validation là 23 (PRID), 43 (iLIDs) và 29 (Duke).",
     "epoch được chọn làm checkpoint là 15 (PRID), 17 (iLIDs) và 17 (Duke)."),
    ("Để bóc tách đóng góp", "chúng tôi thực hiện hai loại ablation", "chúng tôi thực hiện ablation độc lập"),
    ("Để bóc tách đóng góp", "(trung bình qua 1 seed): (i) ablation cộng dồn, lần lượt bổ sung từng khối lên baseline (Bảng 6, "
     "Hình 5); (ii) ablation độc lập, bỏ từng thành phần khỏi mô hình đầy đủ (Bảng 6b). Baseline gồm ViT không gian huấn "
     "luyện từ đầu, không có chú ý thời gian (các khung chỉ tương tác qua token [CLS] dùng chung và các đầu ra gộp thời "
     "gian), BERT mã hóa toàn câu, không dùng ℒapp, ℒmot, ℒMIM và chỉ dùng nhánh toàn thể khi chấm điểm. Ở cấu hình 1–3, "
     "checkpoint được chọn theo Rank@1 validation với chỉ nhánh toàn thể; ở các cấu hình còn lại, theo bộ trọng số mặc định.",
     "(seed 42): bỏ từng thành phần khỏi mô hình đầy đủ và giữ nguyên các thành phần còn lại (Bảng 6). Khác với ablation "
     "cộng dồn, cách này đo đóng góp riêng của từng thành phần mà không phụ thuộc vào thứ tự bổ sung. Hàng (a) huấn "
     "luyện cả Space-Time ViT và S3D từ khởi tạo ngẫu nhiên; (b) bỏ bước chú ý thời gian trong mọi khối của Space-Time ViT; "
     "(c) dùng câu đầy đủ cho cả ba nhánh văn bản; (d) bỏ số hạng ℒMIM khỏi hàm mất mát. Ngoài ra, chúng tôi chấm lại "
     "mô hình đầy đủ chỉ bằng nhánh toàn thể để đo riêng tác dụng của chấm điểm đa tầng (không cần huấn luyện lại)."),
    ("Bảng 6b: Ablation độc lập", "Bảng 6b:", "Bảng 6:"),
    ("− Khởi tạo ImageNet-1K", "− Khởi tạo ImageNet-1K", "− Khởi tạo tiền huấn luyện (ViT, S3D)"),
    ("Hạn chế thống kê", "kể cả giữa các cấu hình ablation liền kề,", "kể cả giữa các cấu hình ablation,"),
    ("Hình 6 trích ma trận", "Hình 6 trích", "Hình 5 trích"),
    ("Hình 5 trích ma trận", "(seed 42). Các ô trên đường chéo (cặp đúng) đạt 0,47–0,71, vượt trội so với các ô ngoài đường "
     "chéo, phần lớn nhỏ hơn 0,3 và nhiều giá trị âm.",
     "(seed 42; cos(Tfull, Vvideo)). Các ô trên đường chéo (cặp đúng) đạt 0,33–0,79 và là giá trị lớn nhất trên mỗi hàng; "
     "90% ô ngoài đường chéo nhỏ hơn 0,3 và hai phần ba là giá trị âm. Ô ngoài đường chéo cao nhất là cặp T2–V4 (0,59)."),
    ("Hình 6: Ma trận tương đồng", "Hình 6:", "Hình 5:"),
    ("Mã hóa vệt:", "cho SpaceTimeViT.", "cho bộ mã hóa video."),
    ("Ứng dụng cho phép chuyển đổi checkpoint", "phục vụ thử nghiệm trên nhiều bối cảnh camera khác nhau.",
     "phục vụ thử nghiệm trên nhiều bối cảnh camera khác nhau. Để chạy được trên CPU, bản demo dùng phiên bản bộ mã hóa video "
     "một luồng trước đây của nhóm (ViT 6 khối trên 16 khung); các kết quả định lượng ở mục 5 không dùng bản demo này."),
    ("Bài báo đã trình bày kiến trúc", "chúng tôi thay tổ hợp ViT + S3D bằng bộ mã hóa SpaceTimeViT dựa trên chú ý "
     "không–thời gian phân chia, đề xuất",
     "chúng tôi giữ hai bộ mã hóa video của MFGF và bổ sung ba đầu ra chuyên biệt, đề xuất"),
    ("Bài báo đã trình bày kiến trúc", "tốt hơn ở Median Rank trên iLIDs.",
     "tốt hơn ở Median Rank trên iLIDs. Ablation độc lập cho thấy khởi tạo tiền huấn luyện là yếu tố quyết định; đóng góp "
     "của DSCA và ℒMIM cần được đánh giá thêm với nhiều seed và trên tập dữ liệu lớn hơn."),
    ("Hạn chế. (1)", "(4) luồng chuyển động hội tụ chậm do mô tả hành động kém đa dạng; (5) bộ phân rã DSCA phụ thuộc vào từ "
     "điển định sẵn; (6) lấy mẫu 16 khung có thể bỏ sót sự kiện ngắn trong video dài; (7) phần ứng dụng mới dừng ở mức "
     "demo, chưa đo IDF1/MOTA; (8) hệ thống",
     "(4) ablation mới được thực hiện với một seed trên PRID; (5) bộ phân rã DSCA phụ thuộc vào từ điển định sẵn; (6) phần "
     "ứng dụng mới dừng ở mức demo, chưa đo IDF1/MOTA; (7) hệ thống"),
]

TABLE5 = {"4,774 → 0,710": "4,335 → 0,390", "2,768 → 0,240": "2,492 → 0,034", "2,756 → 1,785": "2,771 → 1,409",
          "0,009 → 0,020": "0,019 → 0,022", "4,711 → 0,535": "4,441 → 0,492", "2,691 → 0,318": "2,540 → 0,230",
          "2,701 → 0,635": "2,713 → 0,741", "0,026 → 0,046": "0,016 → 0,055", "4,203 → 0,262": "3,279 → 0,385",
          "2,353 → 0,070": "1,774 → 0,039", "2,673 → 0,797": "2,610 → 1,339", "0,011 → 0,025": "0,020 → 0,032"}


# ---------------------------------------------------------------------------------- OMML helpers
def mr(t, plain=False):
    return f'<m:r><m:rPr>{"<m:sty m:val=\"p\"/>" if plain else ""}</m:rPr><m:t>{escape(t)}</m:t></m:r>'


def msub(base, sub):
    return f"<m:sSub><m:sSubPr><m:ctrlPr/></m:sSubPr><m:e>{base}</m:e><m:sub>{sub}</m:sub></m:sSub>"


def msup(base, sup):
    return f"<m:sSup><m:sSupPr><m:ctrlPr/></m:sSupPr><m:e>{base}</m:e><m:sup>{sup}</m:sup></m:sSup>"


def md(inner):
    return f"<m:d><m:dPr><m:ctrlPr/></m:dPr><m:e>{inner}</m:e></m:d>"


H_CLS = msub(mr("h"), mr("CLS", True))
EQ5 = H_CLS + mr("=") + msub(mr("ViT", True), mr("ST", True)) + md(msub(mr("x"), mr("1")) + mr(",…,") + msub(mr("x"), mr("4"))) \
    + msub(mr(""), mr("[CLS]", True))
EQ6 = msub(mr("V"), mr("app", True)) + mr("=") + mr("Norm", True) + md(mr("LayerNorm", True) + md(msub(mr("W"), mr("a")) + H_CLS))
EQ7 = mr("m=") + mr("AvgPool", True) + md(mr("S3D", True) + md(msub(mr("x̃"), mr("1")) + mr(",…,") + msub(mr("x̃"), mr("16")))) \
    + mr("∈") + msup(mr("ℝ"), mr("1024"))


# ---------------------------------------------------------------------------------- text helpers
def ptext(p):
    return "".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", p))


def replace_in_para(p, old, new):
    """Replace `old` (matched on the paragraph's visible text, may span runs) by `new` in the first run it touches."""
    runs = [m for m in re.finditer(r"<w:r>(?:(?!</w:r>).)*?<w:t[^>]*>([^<]*)</w:t>(?:(?!</w:r>).)*?</w:r>", p, re.S)]
    texts = [m.group(1) for m in runs]
    full = "".join(texts)
    old_x, new_x = escape(old), escape(new)
    i = full.find(old_x)
    if i < 0:
        return None
    j = i + len(old_x)
    out, pos, acc = [], 0, 0
    for m, t in zip(runs, texts):
        a, b = acc, acc + len(t)
        acc = b
        if b <= i or a >= j:
            continue
        keep_l = t[:max(0, i - a)]
        keep_r = t[max(0, j - a):] if b > j else ""
        mid = new_x if a <= i < b or (i < a and not out) else ""
        nt = keep_l + mid + keep_r
        run = m.group(0)
        run2 = re.sub(r"<w:t[^>]*>[^<]*</w:t>", f'<w:t xml:space="preserve">{nt}</w:t>', run, count=1)
        out.append((m.start(), m.end(), run2))
    res = p
    for s, e, r in reversed(out):
        res = res[:s] + r + res[e:]
    return res


def edit_paras(xml, fn):
    return PARA.sub(lambda m: fn(m.group(0)), xml)


def structural(doc_xml: str) -> str:
    x = doc_xml
    # cumulative ablation: caption + table, figure 5 + caption, summary sentence; resource paragraph
    cap = x.find("Bảng 6: Ablation cộng dồn")
    p0 = x.rfind("<w:p>", 0, cap) if x.rfind("<w:p>", 0, cap) > x.rfind("<w:p ", 0, cap) else x.rfind("<w:p ", 0, cap)
    t1 = x.find("</w:tbl>", cap) + len("</w:tbl>")
    assert "ABLC_6_MDR" in x[p0:t1]
    x = x[:p0] + x[t1:]
    drop = ("rId16", "Hình 5: Rank@1 trên tập test TVPReid-PRID", 'w:val="ABLC_SENTENCE"', 'w:val="BENCH_GPU"')
    x = PARA.sub(lambda m: "" if any(d in m.group(0) for d in drop) else m.group(0), x)
    for d in drop:
        assert d not in x, d
    # equations (5), (6), (7)
    for num, eq in (("(5)", EQ5), ("(6)", EQ6), ("(7)", EQ7)):
        def fix(p, num=num, eq=eq):
            if "<m:oMath>" in p and f">{num}<" in p:
                return re.sub(r"<m:oMath>.*?</m:oMath>", lambda _: f"<m:oMath>{eq}</m:oMath>", p, count=1, flags=re.S)
            return p
        x = edit_paras(x, fix)
    return x


def text_edits(doc_xml: str) -> str:
    x = doc_xml
    for anchor, old, new in EDITS:
        hit = [False]

        def fix(p, anchor=anchor, old=old, new=new):
            if hit[0] or anchor not in ptext(p).replace("&amp;", "&"):
                return p
            r = replace_in_para(p, old, new)
            if r is None:
                return p
            hit[0] = True
            return r
        x = edit_paras(x, fix)
        if not hit[0]:
            sys.exit(f"[x] edit not applied: {anchor!r} / {old[:60]!r}")
    for old, new in TABLE5.items():
        assert x.count(f">{old}<") == 1, old
        x = x.replace(f">{old}<", f">{new}<")
    return layout(shorten(table6(x)))


# ---------------------------------------------------------------------------------- Table 6 as a component matrix
T6_CAPTION_OLD = ("(tập test, giao thức [24]): bỏ từng thành phần khỏi mô hình đầy đủ, giữ nguyên các thành phần còn lại. "
                  "Δ: thay đổi Rank@1 so với mô hình đầy đủ, kèm KTC 95% bootstrap cặp.")
T6_CAPTION_NEW = ("(tập test, giao thức [24], seed 42). Mỗi hàng (a)–(d) bỏ đúng một thành phần (✗) khỏi mô hình đầy đủ, "
                  "giữ nguyên các thành phần còn lại (✓) và được huấn luyện lại 30 epoch. Chênh lệch so với mô hình đầy đủ và KTC 95% bootstrap "
                  "cặp được nêu trong phần thảo luận bên dưới.")
T6_COLS = [1600, 650, 650, 650, 650, 700]                         # sum scaled to 4800 below
T6_HEAD = ["Cấu hình", "Khởi\ntạo", "Chú ý\nthời gian", "DSCA", "ℒMIM", "R@1 (%)"]
T6_ROWS = [("full", "Đầy đủ", None), ("no_init", "(a) Bỏ khởi tạo", 0), ("no_temporal", "(b) Bỏ chú ý thời gian", 1),
           ("no_dsca", "(c) Bỏ DSCA", 2), ("no_mim", "(d) Bỏ ℒMIM", 3)]


def _t6_cell(width, lines, top=None, bottom=None, bold=False, color=None, align="center", size=14):
    def border(side, v):
        return (f'<w:{side} w:val="single" w:color="000000" w:sz="{v}" w:space="0"/>' if v else
                f'<w:{side} w:val="none" w:color="ffffff" w:sz="0" w:space="0"/>')
    rpr = ('<w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="Times New Roman" '
           'w:cs="Times New Roman"/>' + ("<w:b/><w:bCs/>" if bold else "") + (f'<w:color w:val="{color}"/>' if color else "")
           + f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/></w:rPr>')
    paras = "".join(
        f'<w:p><w:pPr><w:keepNext w:val="true"/><w:spacing w:after="0" w:before="0" w:line="190" w:lineRule="exact"/>'
        f'<w:jc w:val="{align}"/></w:pPr><w:r>{rpr}<w:t xml:space="preserve">{escape(s)}</w:t></w:r></w:p>' for s in lines)
    return (f'<w:tc><w:tcPr><w:tcBorders>{border("top", top)}{border("left", 0)}{border("bottom", bottom)}{border("right", 0)}'
            f'</w:tcBorders><w:tcMar><w:left w:w="30" w:type="dxa"/><w:top w:w="20" w:type="dxa"/><w:right w:w="30" '
            f'w:type="dxa"/><w:bottom w:w="20" w:type="dxa"/></w:tcMar><w:tcW w:w="{width}" w:type="dxa"/>'
            f'<w:vAlign w:val="center"/></w:tcPr>{paras}</w:tc>')


def table6(x: str) -> str:
    tok = json.loads((REPO / "results" / "tokens.json").read_text(encoding="utf-8"))
    cap = x.find("Bảng 6: Ablation độc lập")
    assert cap > 0
    t0 = x.find("<w:tbl>", cap)
    t1 = x.find("</w:tbl>", t0) + len("</w:tbl>")
    old = x[t0:t1]
    tblpr = re.search(r"<w:tblPr>.*?</w:tblPr>", old, re.S).group(0)
    scale = 4800 / sum(T6_COLS)
    cols = [int(round(c * scale)) for c in T6_COLS]
    cols[0] += 4800 - sum(cols)
    grid = "<w:tblGrid>" + "".join(f'<w:gridCol w:w="{c}"/>' for c in cols) + "</w:tblGrid>"
    head = "".join(_t6_cell(w, h.split("\n"), top=12, bottom=6, bold=True, align="left" if i == 0 else "center", size=13)
                   for i, (w, h) in enumerate(zip(cols, T6_HEAD)))
    rows = ['<w:tr><w:trPr><w:cantSplit/><w:tblHeader/></w:trPr>' + head + "</w:tr>"]
    for k, (name, label, drop) in enumerate(T6_ROWS):
        last = k == len(T6_ROWS) - 1
        bot = 12 if last else (4 if k == 0 else 0)
        full = drop is None
        cells = [_t6_cell(cols[0], [label], bottom=bot, bold=full, align="left")]
        for j in range(4):
            mark, color = ("✗", "C8102E") if j == drop else ("✓", "2E7D32")
            cells.append(_t6_cell(cols[1 + j], [mark], bottom=bot, bold=True, color=color))
        cells.append(_t6_cell(cols[5], [tok[f"ABLL_{name}_R1"]], bottom=bot, bold=full))
        rows.append('<w:tr><w:trPr><w:cantSplit/></w:trPr>' + "".join(cells) + "</w:tr>")
    x = x[:t0] + "<w:tbl>" + tblpr + grid + "".join(rows) + "</w:tbl>" + x[t1:]

    def fix(p):
        if "Bảng 6: Ablation độc lập" not in ptext(p):
            return p
        r = replace_in_para(p, T6_CAPTION_OLD, T6_CAPTION_NEW)
        assert r is not None, "table 6 caption"
        return r
    return edit_paras(x, fix)


def main():
    tmp = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(TEMPLATE) as z:
        names = z.namelist()
        z.extractall(tmp / "a")
    doc = tmp / "a" / "word" / "document.xml"
    doc.write_text(structural(doc.read_text(encoding="utf-8")), encoding="utf-8")
    for media, fig in (("image2.png", "fig2_architecture_gen.png"), ("image4.png", "fig4_dynamics.png"),
                       ("image6.png", "fig6_similarity.png")):
        shutil.copyfile(FIG / fig, tmp / "a" / "word" / "media" / media)
    stage1 = tmp / "stage1.docx"
    with zipfile.ZipFile(stage1, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.write(tmp / "a" / n, n)

    tokens = json.loads((REPO / "results" / "tokens_numbers.json").read_text(encoding="utf-8"))
    tokens.update(SENTENCES)
    (REPO / "results" / "tokens.json").write_text(json.dumps(tokens, indent=1, ensure_ascii=False), encoding="utf-8")
    stage2 = tmp / "stage2.docx"
    subprocess.run([sys.executable, str(REPO / "experiments" / "fill_paper.py"), "--template", str(stage1),
                    "--tokens", str(REPO / "results" / "tokens.json"), "--figures", str(FIG), "--out", str(stage2)],
                   check=True)

    with zipfile.ZipFile(stage2) as z:
        names = z.namelist()
        z.extractall(tmp / "b")
    doc = tmp / "b" / "word" / "document.xml"
    doc.write_text(text_edits(doc.read_text(encoding="utf-8")), encoding="utf-8")
    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.write(tmp / "b" / n, n)
    shutil.rmtree(tmp, ignore_errors=True)
    print("[+] wrote", OUT)



# ---------------------------------------------------------------------------------- fit into 9 pages
DROP_PARAS = ("Hình 4: Động học huấn luyện", "Trang phục phổ biến giống nhau", "Ánh sáng ngược và bóng râm",
              "Mô tả hành động đơn điệu", "Về cơ chế, mỗi thành phần nhắm tới", "Hạn chế thống kê.")
SHORT_EDITS = [
    ("Giao thức hai câu/video (bổ sung)", "Nếu nạp dữ liệu theo cặp (video, câu) như khi huấn luyện, mỗi video xuất hiện "
     "hai lần trong gallery và chiếm hai vị trí trong danh sách xếp hạng, đẩy video đúng xuống thấp một cách giả tạo; trên "
     "tập validation PRID, với cùng một mô hình, việc khử trùng lặp nâng Rank@5 từ 48,0% lên 81,69%.",
     "Không khử trùng lặp thì mỗi video chiếm hai vị trí trong danh sách xếp hạng và đẩy video đúng xuống thấp một cách "
     "giả tạo."),
    ("Hình 4 thể hiện diễn biến", "Hình 4 thể hiện diễn biến Rank@1 và Median Rank trên tập validation theo epoch (seed 42); "
     "Bảng 5 so sánh", "Bảng 5 so sánh"),
    ("Hình 5 trích ma trận", "Hình 5 trích", "Hình 4 trích"),
    ("Hình 4 trích ma trận", " Phân tích các truy vấn bị xếp sai cho thấy ba nguyên nhân chính:",
     " Các lỗi còn lại chủ yếu là các cặp người có trang phục gần giống nhau."),
    ("Hình 5: Ma trận tương đồng", "Hình 5:", "Hình 4:"),
    ("Phát hiện và bám vết:", " Khi không dùng được YOLOv8, hệ thống chuyển sang Faster R-CNN của torchvision hoặc bộ "
     "phát hiện HOG của OpenCV, kèm bộ liên kết tham lam theo IoU và khoảng cách tâm.", ""),
    ("Xếp hạng và hiển thị:", " Demo có thể trừ thêm điểm phạt khi thuộc tính nêu trong câu (giới tính, màu áo/quần, ba "
     "lô…) mâu thuẫn với dự đoán của một mô hình thuộc tính người có sẵn, và tùy chọn cộng điểm tương đồng của CLIP [20]. "
     "Kết quả hiển thị hộp bao màu theo thứ hạng, ảnh đại diện, dải khung hình và thời điểm bắt đầu, kết thúc của từng vệt.",
     " Kết quả hiển thị hộp bao theo thứ hạng và thời điểm xuất hiện của từng vệt."),
    ("Nhánh video hai luồng với ba đầu ra", "để so sánh công bằng,", "để so sánh trên cùng kiến trúc bộ mã hóa,"),
    ("Môi trường: PyTorch", "GPU NVIDIA Tesla T4 / RTX (16 GB)", "GPU NVIDIA Tesla T4 (16 GB)"),
    ("Trong ví dụ ở Hình 2", "«A woman wearing green coat, black pants, brown shoes, backpack» và «A video of a woman "
     "walking, left, carrying»", "«A woman wearing white jacket, blue trousers, short black hair, black shoes, book, red "
     "backpack» và «A video of a woman walking, in her hand»"),
    ("Hạn chế. (1)", "; (5) bộ phân rã DSCA phụ thuộc vào từ điển định sẵn; (6) phần ứng dụng mới dừng ở mức demo, chưa đo "
     "IDF1/MOTA; (7) hệ thống chỉ nhận câu mô tả tiếng Anh như benchmark TVPReid.",
     "; (5) phần ứng dụng mới dừng ở mức demo, chưa đo IDF1/MOTA."),
]


def shorten(x: str) -> str:
    x = PARA.sub(lambda m: "" if ('r:embed="rId15"' in m.group(0)
                                 or any(ptext(m.group(0)).startswith(a) for a in DROP_PARAS)) else m.group(0), x)
    for a in DROP_PARAS:
        assert a not in "".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", x)), a
    for anchor, old, new in SHORT_EDITS:
        hit = [False]

        def fix(p, anchor=anchor, old=old, new=new):
            if hit[0] or anchor not in ptext(p):
                return p
            r = replace_in_para(p, old, new)
            if r is None:
                return p
            hit[0] = True
            return r
        x = edit_paras(x, fix)
        if not hit[0]:
            sys.exit(f"[x] shorten edit not applied: {anchor!r}")
    return x



# ---------------------------------------------------------------------------------- page layout fixes
def _para_bounds(x, needle):
    i = x.find(needle)
    assert i > 0, needle
    a = max(x.rfind("<w:p>", 0, i), x.rfind("<w:p ", 0, i))
    return a, x.find("</w:p>", i) + len("</w:p>")


def layout(x: str) -> str:
    # 1) Table 4b (caption + table + note) inside a borderless 1x1 table whose row cannot split -> never cut across pages
    a, _ = _para_bounds(x, "Bảng 4b: Rank@1 (%)")
    _, b = _para_bounds(x, "TB ± ĐLC: trung bình ± độ lệch chuẩn")
    inner = x[a:b]
    nb = '<w:{0} w:val="nil"/>'
    borders = "<w:tblBorders>" + "".join(nb.format(k) for k in ("top", "left", "bottom", "right", "insideH", "insideV")) \
        + "</w:tblBorders>"
    outer = ('<w:tbl><w:tblPr><w:tblW w:w="10080" w:type="dxa"/><w:jc w:val="center"/>' + borders +
             '<w:tblLayout w:type="fixed"/><w:tblCellMar><w:left w:w="0" w:type="dxa"/><w:right w:w="0" w:type="dxa"/>'
             '</w:tblCellMar></w:tblPr><w:tblGrid><w:gridCol w:w="10080"/></w:tblGrid><w:tr><w:trPr><w:cantSplit/>'
             '</w:trPr><w:tc><w:tcPr><w:tcW w:w="10080" w:type="dxa"/></w:tcPr>' + inner + '</w:tc></w:tr></w:tbl>')
    x = x[:a] + outer + x[b:]
    # 2) balance the two columns of the reference list: end the last two-column section with a continuous break
    body_end = x.rfind("<w:sectPr")
    last = x[body_end:x.find("</w:sectPr>", body_end) + len("</w:sectPr>")]
    assert 'w:num="2"' in last, "last section is not two-column"
    x = x[:body_end] + "<w:p><w:pPr>" + last + "</w:pPr></w:p>" + last.replace('w:num="2"', 'w:num="1"') \
        + x[body_end + len(last):]
    return x


if __name__ == "__main__":
    main()
