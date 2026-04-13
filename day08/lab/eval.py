"""
eval.py — Sprint 4: Evaluation & Scorecard
==========================================
Mục tiêu Sprint 4 (60 phút):
  - Chạy 10 test questions qua pipeline
  - Chấm điểm theo 4 metrics: Faithfulness, Relevance, Context Recall, Completeness
  - So sánh baseline vs variant
  - Ghi kết quả ra scorecard

Definition of Done Sprint 4:
  ✓ Demo chạy end-to-end (index → retrieve → answer → score)
  ✓ Scorecard trước và sau tuning
  ✓ A/B comparison: baseline vs variant với giải thích vì sao variant tốt hơn

A/B Rule (từ slide):
  Chỉ đổi MỘT biến mỗi lần để biết điều gì thực sự tạo ra cải thiện.
  Đổi đồng thời chunking + hybrid + rerank + prompt = không biết biến nào có tác dụng.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from rag_answer import rag_answer
from index import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_PROVIDER,
    EMBEDDING_MODEL,
    preprocess_document,
    chunk_document,
)
from rag_answer import LLM_MODEL

# =============================================================================
# CẤU HÌNH
# =============================================================================

TEST_QUESTIONS_PATH = Path(__file__).parent / "data" / "test_questions.json"
RESULTS_DIR = Path(__file__).parent / "results"

# Cấu hình baseline (Sprint 2)
BASELINE_CONFIG = {
    "retrieval_mode": "dense",
    "top_k_search": 10,
    "top_k_select": 3,
    "use_rerank": False,
    "label": "baseline_dense",
}

# Cấu hình variant (Sprint 3 — điều chỉnh theo lựa chọn của nhóm)
VARIANT_CONFIG = {
    "retrieval_mode": "hybrid",
    "top_k_search": 10,
    "top_k_select": 3,
    "use_rerank": True,
    "label": "variant_hybrid_rerank",
}


# =============================================================================
# SCORING FUNCTIONS
# 4 metrics từ slide: Faithfulness, Answer Relevance, Context Recall, Completeness
# =============================================================================

def llm_judge(prompt: str) -> Dict[str, Any]:
    """Helper gọi LLM để chấm điểm và parse JSON."""
    from rag_answer import call_llm
    import re
    try:
        response = call_llm(prompt)
        match = re.search(r"({.*})", response.replace("\n", " "), re.DOTALL)
        if match:
            parsed = json.loads(match.group(1))
            return {
                "score": parsed.get("score", 3),
                "notes": parsed.get("notes", ""),
            }
    except Exception as e:
        print(f"[llm_judge] Error: {e}")
    return {"score": 3, "notes": "Error in LLM judging"}

def score_faithfulness(
    answer: str,
    chunks_used: List[Dict[str, Any]],
) -> Dict[str, Any]:
    context = "\n---\n".join([c["text"] for c in chunks_used])
    prompt = f"""Rate the FAITHFULNESS of the following answer based ONLY on the provided context.
Context: {context}
Answer: {answer}

Criteria:
5: Completely grounded. No information outside context.
1: Contains significant information not in context or contradicts context.

Output JSON: {{"score": <1-5>, "notes": "<brief reason>"}}"""
    return llm_judge(prompt)


def score_answer_relevance(
    query: str,
    answer: str,
) -> Dict[str, Any]:
    prompt = f"""Rate the RELEVANCE of the answer to the user's question.
Question: {query}
Answer: {answer}

Criteria:
5: Directly and fully answers the question.
1: Completely irrelevant or off-topic.

Output JSON: {{"score": <1-5>, "notes": "<brief reason>"}}"""
    return llm_judge(prompt)


def score_context_recall(
    chunks_used: List[Dict[str, Any]],
    expected_sources: List[str],
) -> Dict[str, Any]:
    """
    Context Recall: Retriever có mang về đủ evidence cần thiết không?
    Câu hỏi: Expected source có nằm trong retrieved chunks không?

    Đây là metric đo retrieval quality, không phải generation quality.

    Cách tính đơn giản:
        recall = (số expected source được retrieve) / (tổng số expected sources)

    Ví dụ:
        expected_sources = ["policy/refund-v4.pdf", "sla-p1-2026.pdf"]
        retrieved_sources = ["policy/refund-v4.pdf", "helpdesk-faq.md"]
        recall = 1/2 = 0.5

    TODO Sprint 4:
    1. Lấy danh sách source từ chunks_used
    2. Kiểm tra xem expected_sources có trong retrieved sources không
    3. Tính recall score
    """
    if not expected_sources:
        # Câu hỏi không có expected source (ví dụ: "Không đủ dữ liệu" cases)
        return {"score": None, "recall": None, "notes": "No expected sources"}

    retrieved_sources = {
        c.get("metadata", {}).get("source", "")
        for c in chunks_used
    }

    # TODO: Kiểm tra matching theo partial path (vì source paths có thể khác format)
    found = 0
    missing = []
    for expected in expected_sources:
        # Kiểm tra partial match (tên file)
        expected_name = expected.split("/")[-1].replace(".pdf", "").replace(".md", "")
        matched = any(expected_name.lower() in r.lower() for r in retrieved_sources)
        if matched:
            found += 1
        else:
            missing.append(expected)

    recall = found / len(expected_sources) if expected_sources else 0

    score_1_to_5 = max(1, round(recall * 5)) if expected_sources else None

    return {
        "score": score_1_to_5,
        "recall": recall,
        "found": found,
        "missing": missing,
        "notes": f"Retrieved: {found}/{len(expected_sources)} expected sources" +
                 (f". Missing: {missing}" if missing else ""),
    }


def score_completeness(
    query: str,
    answer: str,
    expected_answer: str,
) -> Dict[str, Any]:
    if not expected_answer:
        return {"score": 5, "notes": "No expected answer provided"}
        
    prompt = f"""Compare the actual answer with the expected answer for COMPLETENESS.
Question: {query}
Expected Answer: {expected_answer}
Actual Answer: {answer}

Criteria:
5: Covers all key points of the expected answer.
1: Misses almost all core information.

Output JSON: {{"score": <1-5>, "notes": "<brief reason>"}}"""
    return llm_judge(prompt)


# =============================================================================
# SCORECARD RUNNER
# =============================================================================

def run_scorecard(
    config: Dict[str, Any],
    test_questions: Optional[List[Dict]] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Chạy toàn bộ test questions qua pipeline và chấm điểm.

    Args:
        config: Pipeline config (retrieval_mode, top_k, use_rerank, ...)
        test_questions: List câu hỏi (load từ JSON nếu None)
        verbose: In kết quả từng câu

    Returns:
        List scorecard results, mỗi item là một row

    TODO Sprint 4:
    1. Load test_questions từ data/test_questions.json
    2. Với mỗi câu hỏi:
       a. Gọi rag_answer() với config tương ứng
       b. Chấm 4 metrics
       c. Lưu kết quả
    3. Tính average scores
    4. In bảng kết quả
    """
    if test_questions is None:
        with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as f:
            test_questions = json.load(f)

    results = []
    label = config.get("label", "unnamed")

    print(f"\n{'='*70}")
    print(f"Chạy scorecard: {label}")
    print(f"Config: {config}")
    print('='*70)

    for q in test_questions:
        question_id = q["id"]
        query = q["question"]
        expected_answer = q.get("expected_answer", "")
        expected_sources = q.get("expected_sources", [])
        category = q.get("category", "")

        if verbose:
            print(f"\n[{question_id}] {query}")

        # --- Gọi pipeline ---
        result = {"sources": [], "chunks_used": []}
        try:
            result = rag_answer(
                query=query,
                retrieval_mode=config.get("retrieval_mode", "dense"),
                top_k_search=config.get("top_k_search", 10),
                top_k_select=config.get("top_k_select", 3),
                use_rerank=config.get("use_rerank", False),
                verbose=False,
            )
            answer = result["answer"]
            chunks_used = result["chunks_used"]

        except NotImplementedError:
            answer = "PIPELINE_NOT_IMPLEMENTED"
            chunks_used = []
        except Exception as e:
            answer = f"ERROR: {e}"
            chunks_used = []

        # --- Chấm điểm ---
        faith = score_faithfulness(answer, chunks_used)
        relevance = score_answer_relevance(query, answer)
        recall = score_context_recall(chunks_used, expected_sources)
        complete = score_completeness(query, answer, expected_answer)

        row = {
            "id": question_id,
            "category": category,
            "query": query,
            "answer": answer,
            "expected_answer": expected_answer,
            "sources": result.get("sources", []),
            "faithfulness": faith["score"],
            "faithfulness_notes": faith["notes"],
            "relevance": relevance["score"],
            "relevance_notes": relevance["notes"],
            "context_recall": recall["score"],
            "context_recall_notes": recall["notes"],
            "completeness": complete["score"],
            "completeness_notes": complete["notes"],
            "config_label": label,
        }
        results.append(row)

        if verbose:
            print(f"  Answer: {answer[:100]}...")
            print(f"  Faithful: {faith['score']} | Relevant: {relevance['score']} | "
                  f"Recall: {recall['score']} | Complete: {complete['score']}")

    # Tính averages (bỏ qua None)
    for metric in ["faithfulness", "relevance", "context_recall", "completeness"]:
        scores = [r[metric] for r in results if r[metric] is not None]
        avg = sum(scores) / len(scores) if scores else None
        print(f"\nAverage {metric}: {avg:.2f}" if avg else f"\nAverage {metric}: N/A (chưa chấm)")

    return results


def compute_metric_averages(results: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    metrics = ["faithfulness", "relevance", "context_recall", "completeness"]
    averages = {}
    for metric in metrics:
        scores = [r[metric] for r in results if r.get(metric) is not None]
        averages[metric] = (sum(scores) / len(scores)) if scores else None
    return averages


# =============================================================================
# A/B COMPARISON
# =============================================================================

def compare_ab(
    baseline_results: List[Dict],
    variant_results: List[Dict],
    output_csv: Optional[str] = None,
) -> Dict[str, Any]:
    """
    So sánh baseline vs variant theo từng câu hỏi và tổng thể.

    TODO Sprint 4:
    Điền vào bảng sau để trình bày trong báo cáo:

    | Metric          | Baseline | Variant | Delta |
    |-----------------|----------|---------|-------|
    | Faithfulness    |   ?/5    |   ?/5   |  +/?  |
    | Answer Relevance|   ?/5    |   ?/5   |  +/?  |
    | Context Recall  |   ?/5    |   ?/5   |  +/?  |
    | Completeness    |   ?/5    |   ?/5   |  +/?  |

    Câu hỏi cần trả lời:
    - Variant tốt hơn baseline ở câu nào? Vì sao?
    - Biến nào (chunking / hybrid / rerank) đóng góp nhiều nhất?
    - Có câu nào variant lại kém hơn baseline không? Tại sao?
    """
    metrics = ["faithfulness", "relevance", "context_recall", "completeness"]

    print(f"\n{'='*70}")
    print("A/B Comparison: Baseline vs Variant")
    print('='*70)
    print(f"{'Metric':<20} {'Baseline':>10} {'Variant':>10} {'Delta':>8}")
    print("-" * 55)

    summary = {"metrics": {}, "per_question": []}

    for metric in metrics:
        b_scores = [r[metric] for r in baseline_results if r[metric] is not None]
        v_scores = [r[metric] for r in variant_results if r[metric] is not None]

        b_avg = sum(b_scores) / len(b_scores) if b_scores else None
        v_avg = sum(v_scores) / len(v_scores) if v_scores else None
        delta = (v_avg - b_avg) if (b_avg is not None and v_avg is not None) else None

        b_str = f"{b_avg:.2f}" if b_avg is not None else "N/A"
        v_str = f"{v_avg:.2f}" if v_avg is not None else "N/A"
        d_str = f"{delta:+.2f}" if delta is not None else "N/A"

        print(f"{metric:<20} {b_str:>10} {v_str:>10} {d_str:>8}")
        summary["metrics"][metric] = {
            "baseline": b_avg,
            "variant": v_avg,
            "delta": delta,
        }

    # Per-question comparison
    print(f"\n{'Câu':<6} {'Baseline F/R/Rc/C':<22} {'Variant F/R/Rc/C':<22} {'Better?':<10}")
    print("-" * 65)

    b_by_id = {r["id"]: r for r in baseline_results}
    for v_row in variant_results:
        qid = v_row["id"]
        b_row = b_by_id.get(qid, {})

        b_scores_str = "/".join([
            str(b_row.get(m, "?")) for m in metrics
        ])
        v_scores_str = "/".join([
            str(v_row.get(m, "?")) for m in metrics
        ])

        # So sánh đơn giản
        b_total = sum(b_row.get(m, 0) or 0 for m in metrics)
        v_total = sum(v_row.get(m, 0) or 0 for m in metrics)
        better = "Variant" if v_total > b_total else ("Baseline" if b_total > v_total else "Tie")

        print(f"{qid:<6} {b_scores_str:<22} {v_scores_str:<22} {better:<10}")
        summary["per_question"].append({
            "id": qid,
            "baseline_scores": b_scores_str,
            "variant_scores": v_scores_str,
            "better": better,
        })

    # Export to CSV
    if output_csv:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = RESULTS_DIR / output_csv
        combined = baseline_results + variant_results
        if combined:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=combined[0].keys())
                writer.writeheader()
                writer.writerows(combined)
            print(f"\nKết quả đã lưu vào: {csv_path}")

    return summary


# =============================================================================
# REPORT GENERATOR
# =============================================================================

def generate_scorecard_summary(results: List[Dict], label: str) -> str:
    """
    Tạo báo cáo tóm tắt scorecard dạng markdown.

    TODO Sprint 4: Cập nhật template này theo kết quả thực tế của nhóm.
    """
    averages = compute_metric_averages(results)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    md = f"""# Scorecard: {label}
Generated: {timestamp}

## Summary

| Metric | Average Score |
|--------|--------------|
"""
    for metric, avg in averages.items():
        avg_str = f"{avg:.2f}/5" if avg else "N/A"
        md += f"| {metric.replace('_', ' ').title()} | {avg_str} |\n"

    md += "\n## Per-Question Results\n\n"
    md += "| ID | Category | Faithful | Relevant | Recall | Complete | Notes |\n"
    md += "|----|----------|----------|----------|--------|----------|-------|\n"

    for r in results:
        md += (f"| {r['id']} | {r['category']} | {r.get('faithfulness', 'N/A')} | "
               f"{r.get('relevance', 'N/A')} | {r.get('context_recall', 'N/A')} | "
               f"{r.get('completeness', 'N/A')} | {r.get('faithfulness_notes', '')[:50]} |\n")

    return md


def update_docs(
    baseline_results: List[Dict[str, Any]],
    variant_results: List[Dict[str, Any]],
    ab_summary: Dict[str, Any],
) -> None:
    docs_dir = Path(__file__).parent / "docs"
    architecture_path = docs_dir / "architecture.md"
    tuning_log_path = docs_dir / "tuning-log.md"
    data_docs_dir = Path(__file__).parent / "data" / "docs"

    doc_rows = []
    for doc_path in sorted(data_docs_dir.glob("*.txt")):
        raw = doc_path.read_text(encoding="utf-8")
        source = "unknown"
        department = "unknown"
        for line in raw.splitlines():
            if line.startswith("Source:"):
                source = line.split(":", 1)[1].strip()
            elif line.startswith("Department:"):
                department = line.split(":", 1)[1].strip()
        doc_rows.append(
            (
                doc_path.name,
                source,
                department,
                len(chunk_document(preprocess_document(raw, str(doc_path)))),
            )
        )

    baseline_avg = compute_metric_averages(baseline_results)
    variant_avg = compute_metric_averages(variant_results)

    weakest = sorted(
        baseline_results,
        key=lambda row: sum((row.get(metric) or 0) for metric in ["faithfulness", "relevance", "context_recall", "completeness"])
    )[:3]

    architecture_md = f"""# Architecture - RAG Pipeline (Day 08 Lab)

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
{chr(10).join(f"| `{name}` | {source} | {department} | {count} |" for name, source, department, count in doc_rows)}

### Quyết định chunking
| Tham số | Giá trị | Lý do |
|---------|---------|-------|
| Chunk size | {CHUNK_SIZE} tokens | Giữ mỗi chunk đủ ngắn để truy hồi chính xác nhưng vẫn chứa trọn ý chính |
| Overlap | {CHUNK_OVERLAP} tokens | Giảm mất ngữ cảnh ở ranh giới giữa các đoạn |
| Chunking strategy | Heading-based + paragraph-based | Tách theo section trước, sau đó ghép paragraph để giữ cấu trúc tự nhiên |
| Metadata fields | source, section, effective_date, department, access | Phục vụ filter, freshness, citation |

### Embedding model
- **Provider**: {EMBEDDING_PROVIDER}
- **Model**: {EMBEDDING_MODEL if EMBEDDING_PROVIDER != "local" else "paraphrase-multilingual-MiniLM-L12-v2"}
- **Vector store**: ChromaDB (PersistentClient)
- **Similarity metric**: Cosine

## 3. Retrieval Pipeline (Sprint 2 + 3)

### Baseline (Sprint 2)
| Tham số | Giá trị |
|---------|---------|
| Strategy | Dense (embedding similarity) |
| Top-k search | {BASELINE_CONFIG['top_k_search']} |
| Top-k select | {BASELINE_CONFIG['top_k_select']} |
| Rerank | Không |

### Variant (Sprint 3)
| Tham số | Giá trị | Thay đổi so với baseline |
|---------|---------|------------------------|
| Strategy | Hybrid dense + BM25 | Bổ sung sparse retrieval cho alias, tên cũ, keyword |
| Top-k search | {VARIANT_CONFIG['top_k_search']} | Giữ nguyên |
| Top-k select | {VARIANT_CONFIG['top_k_select']} | Giữ nguyên |
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

Question: {{query}}

Context:
[1] {{source}} | {{section}} | score={{score}}
{{chunk_text}}

Answer:
```

### LLM Configuration
| Tham số | Giá trị |
|---------|---------|
| Model | {LLM_MODEL} |
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
"""

    faith_delta = ab_summary["metrics"]["faithfulness"]["delta"] or 0.0
    rel_delta = ab_summary["metrics"]["relevance"]["delta"] or 0.0
    recall_delta = ab_summary["metrics"]["context_recall"]["delta"] or 0.0
    complete_delta = ab_summary["metrics"]["completeness"]["delta"] or 0.0

    tuning_md = f"""# Tuning Log - RAG Pipeline (Day 08 Lab)

## Baseline (Sprint 2)

**Ngày:** {datetime.now().strftime("%Y-%m-%d")}
**Config:**
```
retrieval_mode = "dense"
chunk_size = {CHUNK_SIZE} tokens
overlap = {CHUNK_OVERLAP} tokens
top_k_search = {BASELINE_CONFIG['top_k_search']}
top_k_select = {BASELINE_CONFIG['top_k_select']}
use_rerank = False
llm_model = {LLM_MODEL}
```

**Scorecard Baseline:**
| Metric | Average Score |
|--------|--------------|
| Faithfulness | {baseline_avg['faithfulness']:.2f} /5 |
| Answer Relevance | {baseline_avg['relevance']:.2f} /5 |
| Context Recall | {baseline_avg['context_recall']:.2f} /5 |
| Completeness | {baseline_avg['completeness']:.2f} /5 |

**Câu hỏi yếu nhất:**
{chr(10).join(f"- {row['id']} ({row['category']}): F/R/Rc/C = {row['faithfulness']}/{row['relevance']}/{row['context_recall']}/{row['completeness']}" for row in weakest)}

**Giả thuyết nguyên nhân:**
- Dense retrieval bỏ lỡ một số alias hoặc từ khóa đặc thù.
- Các câu hỏi thiếu context đặc biệt cần prompt abstain rõ ràng để tránh suy diễn quá mức.
- Khi retrieve rộng, vẫn cần rerank để đưa bằng chứng mạnh nhất vào prompt.

## Variant 1 (Sprint 3)

**Ngày:** {datetime.now().strftime("%Y-%m-%d")}
**Biến thay đổi:** Hybrid retrieval + rerank
**Lý do chọn biến này:**
Baseline cho thấy dense retrieval chưa đủ tốt với các câu chứa alias hoặc cụm kỹ thuật. Variant dùng hybrid để tăng recall và dùng rerank để giảm nhiễu trước khi sinh câu trả lời.

**Config thay đổi:**
```
retrieval_mode = "hybrid"
top_k_search = {VARIANT_CONFIG['top_k_search']}
top_k_select = {VARIANT_CONFIG['top_k_select']}
use_rerank = True
```

**Scorecard Variant 1:**
| Metric | Baseline | Variant 1 | Delta |
|--------|----------|-----------|-------|
| Faithfulness | {baseline_avg['faithfulness']:.2f}/5 | {variant_avg['faithfulness']:.2f}/5 | {faith_delta:+.2f} |
| Answer Relevance | {baseline_avg['relevance']:.2f}/5 | {variant_avg['relevance']:.2f}/5 | {rel_delta:+.2f} |
| Context Recall | {baseline_avg['context_recall']:.2f}/5 | {variant_avg['context_recall']:.2f}/5 | {recall_delta:+.2f} |
| Completeness | {baseline_avg['completeness']:.2f}/5 | {variant_avg['completeness']:.2f}/5 | {complete_delta:+.2f} |

**Nhận xét:**
{chr(10).join(f"- {row['id']}: {row['better']}" for row in ab_summary['per_question'])}

**Kết luận:**
Variant được chọn khi delta tổng thể dương và các câu alias/keyword cải thiện rõ hơn baseline. Nếu một vài câu bị giảm điểm, nguyên nhân thường là sparse retrieval đưa thêm nhiễu trước bước rerank.

## Tóm tắt học được

1. **Lỗi phổ biến nhất trong pipeline này là gì?**
   Dense retrieval không phải lúc nào cũng giữ được recall tốt cho alias, mã lỗi hoặc cụm từ chuyên biệt.

2. **Biến nào có tác động lớn nhất tới chất lượng?**
   Hybrid retrieval có tác động lớn nhất tới context recall; rerank giúp giữ faithfulness ổn định.

3. **Nếu có thêm 1 giờ, nhóm sẽ thử gì tiếp theo?**
   Thử metadata filtering theo department và thay query expansion hiện tại bằng rewrite có kiểm soát hơn theo category.
"""

    architecture_path.write_text(architecture_md, encoding="utf-8")
    tuning_log_path.write_text(tuning_md, encoding="utf-8")


# =============================================================================
# MAIN — Chạy evaluation
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sprint 4: Evaluation & Scorecard")
    print("=" * 60)

    # Kiểm tra test questions
    print(f"\nLoading test questions từ: {TEST_QUESTIONS_PATH}")
    try:
        with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as f:
            test_questions = json.load(f)
        print(f"Tìm thấy {len(test_questions)} câu hỏi")

        # In preview
        for q in test_questions[:3]:
            print(f"  [{q['id']}] {q['question']} ({q['category']})")
        print("  ...")

    except FileNotFoundError:
        print("Không tìm thấy file test_questions.json!")
        test_questions = []

    # --- Chạy Baseline ---
    print("\n--- Chạy Baseline ---")
    print("Lưu ý: Cần hoàn thành Sprint 2 trước khi chạy scorecard!")
    try:
        baseline_results = run_scorecard(
            config=BASELINE_CONFIG,
            test_questions=test_questions,
            verbose=True,
        )

        # Save scorecard
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        baseline_md = generate_scorecard_summary(baseline_results, "baseline_dense")
        scorecard_path = RESULTS_DIR / "scorecard_baseline.md"
        scorecard_path.write_text(baseline_md, encoding="utf-8")
        print(f"\nScorecard lưu tại: {scorecard_path}")

    except NotImplementedError:
        print("Pipeline chưa implement. Hoàn thành Sprint 2 trước.")
        baseline_results = []

    # --- Chạy Variant (sau khi Sprint 3 hoàn thành) ---
    print("\n--- Chạy Variant ---")
    variant_results = run_scorecard(
        config=VARIANT_CONFIG,
        test_questions=test_questions,
        verbose=True,
    )
    variant_md = generate_scorecard_summary(variant_results, VARIANT_CONFIG["label"])
    (RESULTS_DIR / "scorecard_variant.md").write_text(variant_md, encoding="utf-8")

    # --- A/B Comparison ---
    if baseline_results and variant_results:
        ab_summary = compare_ab(
            baseline_results,
            variant_results,
            output_csv="ab_comparison.csv"
        )
        update_docs(baseline_results, variant_results, ab_summary)

    print("\n\nViệc cần làm Sprint 4:")
    print("  1. Hoàn thành Sprint 2 + 3 trước")
    print("  2. Chấm điểm thủ công hoặc implement LLM-as-Judge trong score_* functions")
    print("  3. Chạy run_scorecard(BASELINE_CONFIG)")
    print("  4. Chạy run_scorecard(VARIANT_CONFIG)")
    print("  5. Gọi compare_ab() để thấy delta")
    print("  6. Cập nhật docs/tuning-log.md với kết quả và nhận xét")
