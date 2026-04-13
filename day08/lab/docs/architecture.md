# Architecture - RAG Pipeline (Day 08 Lab)

## 1. Tổng quan kiến trúc

```
[Raw Docs]
    ->
[index.py: Preprocess -> Chunk -> Embed -> Store]
    ->
[ChromaDB Vector Store]
    ->
[rag_answer.py: Query -> Retrieve -> Rerank -> Generate]
    ->
[Grounded Answer + Citation]
```

**Mô tả ngắn gọn:**
Hệ thống RAG này phục vụ trợ lý nội bộ cho CS, IT Helpdesk và HR, trả lời câu hỏi chính sách bằng bằng chứng lấy từ tài liệu nội bộ.
Pipeline gồm index tài liệu thành chunk có metadata, truy hồi bằng dense hoặc hybrid, sau đó rerank và sinh câu trả lời có citation.

## 2. Indexing Pipeline (Sprint 1)

### Tài liệu được index
| File | Nguồn | Department | Số chunk |
|------|-------|-----------|---------|
| `access_control_sop.txt` | it/access-control-sop.md | IT Security | 8 |
| `hr_leave_policy.txt` | hr/leave-policy-2026.pdf | HR | 5 |
| `it_helpdesk_faq.txt` | support/helpdesk-faq.md | IT | 6 |
| `policy_refund_v4.txt` | policy/refund-v4.pdf | CS | 6 |
| `sla_p1_2026.txt` | support/sla-p1-2026.pdf | IT | 5 |

### Quyết định chunking
| Tham số | Giá trị | Lý do |
|---------|---------|-------|
| Chunk size | 400 tokens | Giữ mỗi chunk đủ ngắn để truy hồi chính xác nhưng vẫn chứa trọn ý chính |
| Overlap | 80 tokens | Giảm mất ngữ cảnh ở ranh giới giữa các đoạn |
| Chunking strategy | Heading-based + paragraph-based | Tách theo section trước, sau đó ghép paragraph để giữ cấu trúc tự nhiên |
| Metadata fields | source, section, effective_date, department, access | Phục vụ filter, freshness, citation |

### Embedding model
- **Provider**: openai
- **Model**: text-embedding-3-small
- **Vector store**: ChromaDB (PersistentClient)
- **Similarity metric**: Cosine

## 3. Retrieval Pipeline (Sprint 2 + 3)

### Baseline (Sprint 2)
| Tham số | Giá trị |
|---------|---------|
| Strategy | Dense (embedding similarity) |
| Top-k search | 10 |
| Top-k select | 3 |
| Rerank | Không |

### Variant (Sprint 3)
| Tham số | Giá trị | Thay đổi so với baseline |
|---------|---------|------------------------|
| Strategy | Hybrid dense + BM25 | Bổ sung sparse retrieval cho alias, tên cũ, keyword |
| Top-k search | 10 | Giữ nguyên |
| Top-k select | 3 | Giữ nguyên |
| Rerank | CrossEncoder | Có, để giảm noise sau bước retrieve |
| Query transform | Expansion cho hybrid | Có, để tăng recall cho alias và cách diễn đạt khác |

**Lý do chọn variant này:**
Hybrid + rerank phù hợp vì corpus có cả câu tự nhiên và các cụm đặc thù như P1, Level 3, Approval Matrix. Dense retrieval một mình dễ bỏ lỡ alias hoặc keyword chính xác, còn rerank giúp lọc lại các candidate gần đúng.

## 4. Generation (Sprint 2)

### Grounded Prompt Template
```
Answer only from the retrieved context below.
If the context is insufficient, say "Không đủ dữ liệu trong tài liệu hiện có."
Cite the source field when possible.
Keep your answer short, clear, and factual.

Question: {query}

Context:
[1] {source} | {section} | score={score}
{chunk_text}

Answer:
```

### LLM Configuration
| Tham số | Giá trị |
|---------|---------|
| Model | gpt-4o-mini |
| Temperature | 0 |
| Max tokens | 512 |

## 5. Failure Mode Checklist

| Failure Mode | Triệu chứng | Cách kiểm tra |
|-------------|-------------|---------------|
| Index lỗi | Retrieve về docs cũ / sai version | `inspect_metadata_coverage()` trong index.py |
| Chunking tệ | Chunk cắt giữa điều khoản | `list_chunks()` và đọc text preview |
| Retrieval lỗi | Không tìm được expected source | `score_context_recall()` trong eval.py |
| Generation lỗi | Answer không grounded / bịa | `score_faithfulness()` trong eval.py |
| Token overload | Context quá dài -> lost in the middle | Kiểm tra độ dài `context_block` |
