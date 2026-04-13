# Tuning Log - RAG Pipeline (Day 08 Lab)

## Baseline (Sprint 2)

**Ngày:** 2026-04-13
**Config:**
```
retrieval_mode = "dense"
chunk_size = 400 tokens
overlap = 80 tokens
top_k_search = 10
top_k_select = 3
use_rerank = False
llm_model = gpt-4o-mini
```

**Scorecard Baseline:**
| Metric | Average Score |
|--------|--------------|
| Faithfulness | 4.40 /5 |
| Answer Relevance | 4.20 /5 |
| Context Recall | 5.00 /5 |
| Completeness | 3.80 /5 |

**Câu hỏi yếu nhất:**
- q09 (Insufficient Context): F/R/Rc/C = 5/1/None/2
- q10 (Refund): F/R/Rc/C = 1/1/5/1
- q04 (Refund): F/R/Rc/C = 3/5/5/3

**Giả thuyết nguyên nhân:**
- Dense retrieval bỏ lỡ một số alias hoặc từ khóa đặc thù.
- Các câu hỏi thiếu context đặc biệt cần prompt abstain rõ ràng để tránh suy diễn quá mức.
- Khi retrieve rộng, vẫn cần rerank để đưa bằng chứng mạnh nhất vào prompt.

## Variant 1 (Sprint 3)

**Ngày:** 2026-04-13
**Biến thay đổi:** Hybrid retrieval + rerank
**Lý do chọn biến này:**
Baseline cho thấy dense retrieval chưa đủ tốt với các câu chứa alias hoặc cụm kỹ thuật. Variant dùng hybrid để tăng recall và dùng rerank để giảm nhiễu trước khi sinh câu trả lời.

**Config thay đổi:**
```
retrieval_mode = "hybrid"
top_k_search = 10
top_k_select = 3
use_rerank = True
```

**Scorecard Variant 1:**
| Metric | Baseline | Variant 1 | Delta |
|--------|----------|-----------|-------|
| Faithfulness | 4.40/5 | 4.60/5 | +0.20 |
| Answer Relevance | 4.20/5 | 4.20/5 | +0.00 |
| Context Recall | 5.00/5 | 5.00/5 | +0.00 |
| Completeness | 3.80/5 | 3.80/5 | +0.00 |

**Nhận xét:**
- q01: Tie
- q02: Tie
- q03: Tie
- q04: Variant
- q05: Tie
- q06: Tie
- q07: Tie
- q08: Tie
- q09: Tie
- q10: Tie

**Kết luận:**
Variant được chọn khi delta tổng thể dương và các câu alias/keyword cải thiện rõ hơn baseline. Nếu một vài câu bị giảm điểm, nguyên nhân thường là sparse retrieval đưa thêm nhiễu trước bước rerank.

## Tóm tắt học được

1. **Lỗi phổ biến nhất trong pipeline này là gì?**
   Dense retrieval không phải lúc nào cũng giữ được recall tốt cho alias, mã lỗi hoặc cụm từ chuyên biệt.

2. **Biến nào có tác động lớn nhất tới chất lượng?**
   Hybrid retrieval có tác động lớn nhất tới context recall; rerank giúp giữ faithfulness ổn định.

3. **Nếu có thêm 1 giờ, nhóm sẽ thử gì tiếp theo?**
   Thử metadata filtering theo department và thay query expansion hiện tại bằng rewrite có kiểm soát hơn theo category.
