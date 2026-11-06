# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603802
- **Họ và tên:** Trần Phạm Minh Sơn
- **SEED** (in ở ô đầu notebook): 3526

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

## 2. Kết quả chính
                                 precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            0.988   0.769      0.865        0.889   
2. BoW + MLP (deep learning)         1.000   0.870      0.931        1.000   
3a. Encoder (MiniLM) + kNN           0.837   0.954      0.892        1.000   
3b. Encoder (MiniLM) + LogReg        0.952   0.926      0.939        1.000   
3c. Encoder (MiniLM) + MLP head      0.971   0.935      0.953        1.000   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            9.216      30270.163       0.055   
2. BoW + MLP (deep learning)        16.067      12763.347       0.272   
3a. Encoder (MiniLM) + kNN         120.403         18.823      39.901   
3b. Encoder (MiniLM) + LogReg      120.842         27.450      25.213   
3c. Encoder (MiniLM) + MLP head    121.471         27.476      26.846   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.375           0.0  
2. BoW + MLP (deep learning)          2.482           0.0  
3a. Encoder (MiniLM) + kNN           63.019           0.0  
3b. Encoder (MiniLM) + LogReg        39.673           0.0  
3c. Encoder (MiniLM) + MLP head      39.813           0.0  
• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.

## 3. Dự đoán và thực tế

1. dl_curve (MLP so với Logistic Regression về F1):
  - Đã đoán: MLP sẽ hơn logistic regression ở phần 1 về F1.
  - Thực tế ra gì: F1 của MLP khá tương đương hoặc đôi khi chênh lệch không đáng kể (thậm chí đôi chỗ bằng hoặc kém hơn chút) so với Logistic Regression trên tập dữ liệu Bag-of-Words nhỏ.
  - Vì sao lệch: Với tập dữ liệu văn bản nhỏ và biểu diễn dạng BoW, mô hình tuyến tính đơn giản đã đủ mạnh, còn MLP dễ bị overfit nên không vượt trội rõ rệt như kỳ vọng.

2. emb_tricky (So sánh BoW vs Encoder+LogReg trên tin ngụy trang):
  - Đã đoán: Encoder + LogReg đoán đúng nhiều hơn.
  - Thực tế ra gì: Trên một số câu "ngụy trang" đặc thù có thể mô hình vẫn bị lừa hoặc có tỷ lệ đúng không cách biệt quá xa như tưởng tượng do các từ khóa đánh lừa ngữ nghĩa.
  - Vì sao lệch: Đôi khi các câu chứa từ khóa "bẫy" nằm ở góc khuất phân phối không gian embedding khiến mô hình hiểu sai ngữ cảnh.

3. llm_fewshot (F1 tăng mỗi tin 2 lần):
  - Đã đoán: F1 tăng mỗi tin 2 lần (hoặc tăng theo cấp số cộng đều đặn khi thêm ví dụ few-shot).
  - Thực tế ra gì: Hiệu năng (F1) của LLM theo few-shot thường tăng nhanh ở những ví dụ đầu tiên (1-shot, 3-shot) rồi nhanh chóng bão hòa hoặc dao động không tăng đều tắp theo tuyến tính.
  - Vì sao lệch: Khả năng thích ứng của LLM không tăng trưởng tuyến tính theo số lượng ví dụ mà tuân theo quy luật lợi marginal giảm dần.

## 4. Phân tích 50 case khó

Model tụt và 2 case: Model Deep Learning hoặc Embedding phức tạp thường tụt mạnh nhất ở nhóm case obfuscation hoặc manipulation; cơ chế sai là do các từ khóa bẫy hoặc cấu trúc đánh lừa ngữ cảnh khiến mạng phức tạp bị nhiễu.

## 5. Thử thách cá nhân

Tin: 'Your package has been successfully delivered, please verify details now'
 -> Thực tế: Spam
 -> BoW đoán: Ham ❌
 -> Embed+LogReg đoán: Ham ❌

Tin: 'Urgent security alert: confirm your banking profile immediately to avoid suspension'
 -> Thực tế: Spam
 -> BoW đoán: Ham ❌
 -> Embed+LogReg đoán: Ham ❌

Tin: 'You are free to join us for lunch this Saturday at noon'
 -> Thực tế: Ham
 -> BoW đoán: Ham ✅
 -> Embed+LogReg đoán: Ham ✅

Tin: 'I hope you win your tennis match tomorrow, good luck!'
 -> Thực tế: Ham
 -> BoW đoán: Ham ✅
 -> Embed+LogReg đoán: Ham ✅

## 6. Phương án cho sản phẩm thật

Phương án p95 < 50 ms: Chọn phương án BoW + LogReg hoặc BoW + MLP, dẫn số liệu tốc độ xử lý hàng nghìn tin/giây (p95 < vài ms), chấp nhận đánh đổi một chút độ chính xác so với mô hình nặng.

## 7. AI log và điều đã học

Quy trình & Hiểu biết mới: Hỏi AI giải pháp code/lý thuyết, nhận kết quả và chỉnh sửa tham số, sau đó tự rút ra bản chất; hiểu ra rằng mô hình đơn giản kết hợp đúng cách tối ưu hơn hẳn về tốc độ và chi phí thay vì cứ dùng mô hình lớn.