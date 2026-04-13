# Báo Cáo Cá Nhân - Lab Day 08: RAG Pipeline

**Họ và tên:** Bùi Đức Thắng-2A202600002
**Vai trò trong nhóm:** [Tech Lead / Retrieval Owner / Eval Owner / Documentation Owner]  
**Ngày nộp:** 2026-04-13  
**Lưu ý:** Chỉ giữ lại các ý đúng với phần bạn thực sự làm trong repo và commit. Rubric có phạt nặng nếu report không khớp với bằng chứng thực tế.

---

## 1. Tôi đã làm gì trong lab này? (100-150 từ)

Trong lab này tôi phụ trách chính phần `[điền đúng phần bạn làm: indexing / retrieval / evaluation / documentation]`. Công việc cụ thể của tôi là `[điền 2-3 việc thật bạn đã làm]`, ví dụ như chỉnh chunking theo paragraph để chunk tự nhiên hơn, rà lại retrieval baseline và variant, hoặc tổng hợp scorecard để so sánh baseline với hybrid + rerank. Phần tôi làm không đứng riêng lẻ mà nối trực tiếp với các phần còn lại của nhóm: nếu indexing hoặc metadata không ổn thì retrieval sẽ kéo sai evidence, còn nếu evaluation không rõ metric thì nhóm không biết variant có thật sự tốt hơn hay không. Sau khi hoàn tất phần việc của mình, tôi cùng nhóm kiểm tra lại output trên các câu hỏi mẫu như SLA P1, refund policy, access control và câu hỏi thiếu dữ liệu để chắc chắn pipeline vừa trả lời được vừa không bịa.

---

## 2. Điều tôi hiểu rõ hơn sau lab này (100-150 từ)

Điều tôi hiểu rõ hơn rõ rệt sau lab này là retrieval tốt không chỉ phụ thuộc vào embedding model mà còn phụ thuộc rất nhiều vào cách chunking và cách giữ metadata. Trước đây tôi nghĩ chỉ cần embed toàn bộ tài liệu rồi search cosine là đủ, nhưng khi làm lab tôi thấy nếu chunk bị cắt dở hoặc bỏ mất phần preamble quan trọng thì query alias vẫn fail dù embedding model không tệ. Tôi cũng hiểu rõ hơn sự khác nhau giữa `context recall` và `faithfulness`. Một pipeline có thể retrieve đúng tài liệu nên recall cao, nhưng câu trả lời cuối vẫn có thể thiếu ý hoặc thêm suy diễn nhỏ, khi đó faithfulness hoặc completeness sẽ bị trừ điểm. Vì vậy evaluation phải tách retrieval và generation thành các lỗi khác nhau, nếu không nhóm sẽ tune sai chỗ.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn (100-150 từ)

Điều làm tôi mất thời gian nhất là lỗi không nằm ở logic RAG như dự đoán ban đầu mà nằm ở hành vi của môi trường chạy. Trong lúc kiểm tra end-to-end, ChromaDB trên máy phát sinh `disk I/O error`, khiến pipeline không thể dựa hoàn toàn vào persistent store như kế hoạch ban đầu. Ngoài ra, khi xem kỹ output tôi phát hiện một lỗi tinh hơn: file `access_control_sop.txt` có ghi chú đổi tên tài liệu từ "Approval Matrix for System Access", nhưng phần này nằm trước heading đầu tiên nên bị preprocess bỏ mất. Hệ quả là query alias không ổn định dù dense retrieval nhìn bề ngoài vẫn chạy được. Điều này làm tôi thay đổi giả thuyết ban đầu. Vấn đề không phải chỉ ở retrieval mode, mà còn ở chất lượng dữ liệu đã được index. Sau khi giữ lại preamble đó, câu alias chạy đúng hơn và score ổn định hơn.

---

## 4. Phân tích một câu hỏi trong scorecard (150-200 từ)

**Câu hỏi:** `q07 - Approval Matrix để cấp quyền hệ thống là tài liệu nào?`

**Phân tích:**

Tôi chọn `q07` vì đây là câu dễ nhìn tưởng như đơn giản nhưng thực ra kiểm tra đúng vấn đề alias retrieval. Mục tiêu của câu hỏi không phải chỉ tìm tài liệu access control nói chung, mà phải nối được tên cũ "Approval Matrix" với tên tài liệu hiện tại là `Access Control SOP`. Ở giai đoạn đầu, pipeline có nguy cơ fail vì phần ghi chú đổi tên nằm trước heading đầu tiên và bị bỏ khỏi text đã index. Nếu điều đó xảy ra thì dense retrieval vẫn có thể tìm đúng domain "cấp quyền hệ thống", nhưng generation không có bằng chứng trực tiếp để kết luận rằng hai tên này là một tài liệu. Sau khi sửa preprocess để giữ lại preamble trước heading, baseline và variant đều retrieve được chunk chứa alias này, nên answer grounded hơn. Tuy vậy completeness của câu vẫn chỉ ở mức 2/5 vì câu trả lời mới chỉ chỉ ra tài liệu tương ứng chứ chưa diễn đạt đủ ý "tên cũ" và "tên mới" theo expected answer. Root cause chính nằm ở indexing và cách generation tóm tắt, không phải ở embedding model.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì? (50-100 từ)

Nếu có thêm thời gian, tôi sẽ làm hai việc cụ thể. Thứ nhất, thêm bước sinh `logs/grading_run.json` đúng format trong `SCORING.md` để repo khớp hoàn toàn với checklist nộp bài. Thứ hai, tôi sẽ sửa prompt và logic chấm cho các câu abstain để phân biệt tốt hơn giữa "không đủ dữ liệu nhưng nói rõ phạm vi thiếu" và "không đủ dữ liệu chung chung", vì scorecard hiện tại cho thấy các câu như `q09` và `q10` vẫn còn dư địa cải thiện ở relevance và completeness.

---

*Đổi tên file này thành `reports/individual/[ten_ban].md` trước khi nộp.*  
*Chỉ giữ lại các đoạn phản ánh đúng phần bạn thực sự làm và có thể giải thích khi bị hỏi lại.*
