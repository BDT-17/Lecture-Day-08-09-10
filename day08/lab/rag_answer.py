"""
rag_answer.py — Sprint 2 + Sprint 3: Retrieval & Grounded Answer
================================================================
Sprint 2 (60 phút): Baseline RAG
  - Dense retrieval từ ChromaDB
  - Grounded answer function với prompt ép citation
  - Trả lời được ít nhất 3 câu hỏi mẫu, output có source

Sprint 3 (60 phút): Tuning tối thiểu
  - Thêm hybrid retrieval (dense + sparse/BM25)
  - Hoặc thêm rerank (cross-encoder)
  - Hoặc thử query transformation (expansion, decomposition, HyDE)
  - Tạo bảng so sánh baseline vs variant

Definition of Done Sprint 2:
  ✓ rag_answer("SLA ticket P1?") trả về câu trả lời có citation
  ✓ rag_answer("Câu hỏi không có trong docs") trả về "Không đủ dữ liệu"

Definition of Done Sprint 3:
  ✓ Có ít nhất 1 variant (hybrid / rerank / query transform) chạy được
  ✓ Giải thích được tại sao chọn biến đó để tune
"""

import json
import math
import os
import re
import sys
from typing import List, Dict, Any
from dotenv import load_dotenv

# Đảm bảo terminal Windows hiển thị được tiếng Việt
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

# =============================================================================
# CẤU HÌNH
# =============================================================================

TOP_K_SEARCH = 10    # Số chunk lấy từ vector store trước rerank (search rộng)
TOP_K_SELECT = 3     # Số chunk gửi vào prompt sau rerank/select (top-3 sweet spot)

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()


# =============================================================================
# RETRIEVAL — DENSE (Vector Search)
# =============================================================================

def retrieve_dense(query: str, top_k: int = TOP_K_SEARCH) -> List[Dict[str, Any]]:
    from index import get_embedding, CHROMA_DB_DIR, load_fallback_store
    query_embedding = get_embedding(query)

    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        collection = client.get_collection("rag_lab")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        candidates = []
        if results["documents"]:
            for i in range(len(results["documents"][0])):
                candidates.append({
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": 1 - results["distances"][0][i],
                })
        return candidates
    except Exception:
        fallback_rows = load_fallback_store(CHROMA_DB_DIR)
        if not fallback_rows:
            raise

        def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
            dot = sum(a * b for a, b in zip(vec_a, vec_b))
            norm_a = math.sqrt(sum(a * a for a in vec_a))
            norm_b = math.sqrt(sum(b * b for b in vec_b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

        scored = []
        for row in fallback_rows:
            score = cosine_similarity(query_embedding, row["embedding"])
            scored.append({
                "text": row["document"],
                "metadata": row["metadata"],
                "score": score,
            })

        return sorted(scored, key=lambda item: item["score"], reverse=True)[:top_k]


# =============================================================================
# RETRIEVAL — SPARSE / BM25 (Keyword Search)
# Dùng cho Sprint 3 Variant hoặc kết hợp Hybrid
# =============================================================================

def retrieve_sparse(query: str, top_k: int = TOP_K_SEARCH) -> List[Dict[str, Any]]:
    from rank_bm25 import BM25Okapi
    from index import CHROMA_DB_DIR, load_fallback_store

    all_data = None
    if not hasattr(retrieve_sparse, "cache"):
        try:
            import chromadb

            client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
            collection = client.get_collection("rag_lab")
            all_data = collection.get(include=["documents", "metadatas"])
            retrieve_sparse.cache = all_data
        except Exception:
            fallback_rows = load_fallback_store(CHROMA_DB_DIR)
            all_data = {
                "documents": [row["document"] for row in fallback_rows],
                "metadatas": [row["metadata"] for row in fallback_rows],
            }
            retrieve_sparse.cache = all_data
    else:
        all_data = retrieve_sparse.cache

    if not all_data.get("documents"):
        return []

    corpus = all_data["documents"]
    metadatas = all_data["metadatas"]
    
    # 2. Tokenize (đơn giản bằng cách lowercase và split)
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # 3. Query
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    
    # 4. Get top_k results
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    
    candidates = []
    for idx in top_indices:
        if scores[idx] > 0:  # Chỉ lấy kết quả có score > 0
            candidates.append({
                "text": corpus[idx],
                "metadata": metadatas[idx],
                "score": float(scores[idx])
            })
    
    return candidates


# =============================================================================
# RETRIEVAL — HYBRID (Dense + Sparse với Reciprocal Rank Fusion)
# =============================================================================

def retrieve_hybrid(
    query: str,
    top_k: int = TOP_K_SEARCH,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
) -> List[Dict[str, Any]]:
    # 1. Retrieve từ cả 2 nguồn
    dense_results = retrieve_dense(query, top_k=top_k*2) # Lấy nhiều hơn để rerank
    sparse_results = retrieve_sparse(query, top_k=top_k*2)
    
    # 2. Merge bằng Reciprocal Rank Fusion (RRF)
    all_chunks = {}
    rrf_k = 60

    for rank, doc in enumerate(dense_results):
        meta = doc.get("metadata", {})
        key = (meta.get("source", ""), meta.get("section", ""), doc["text"])
        if key not in all_chunks:
            all_chunks[key] = {"data": doc, "rrf_score": 0.0}
        all_chunks[key]["rrf_score"] += dense_weight * (1.0 / (rank + rrf_k))

    for rank, doc in enumerate(sparse_results):
        meta = doc.get("metadata", {})
        key = (meta.get("source", ""), meta.get("section", ""), doc["text"])
        if key not in all_chunks:
            all_chunks[key] = {"data": doc, "rrf_score": 0.0}
        all_chunks[key]["rrf_score"] += sparse_weight * (1.0 / (rank + rrf_k))

    sorted_chunks = sorted(all_chunks.values(), key=lambda x: x["rrf_score"], reverse=True)
    
    final_results = []
    for item in sorted_chunks[:top_k]:
        res = item["data"]
        res["score"] = item["rrf_score"] # Cập nhật score thành RRF score
        final_results.append(res)
        
    return final_results


# =============================================================================
# RERANK (Sprint 3 alternative)
# Cross-encoder để chấm lại relevance sau search rộng
# =============================================================================

def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = TOP_K_SELECT,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    
    from sentence_transformers import CrossEncoder
    
    # Sử dụng model reranker nhẹ và hiệu quả
    if not hasattr(rerank, "model"):
        rerank.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # Tạo các cặp (query, doc_text) để chấm điểm
    pairs = [[query, c["text"]] for c in candidates]
    scores = rerank.model.predict(pairs)
    
    # Gắn điểm mới và sort
    for i, c in enumerate(candidates):
        c["rerank_score"] = float(scores[i])
        
    ranked_candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    
    return ranked_candidates[:top_k]


# =============================================================================
# QUERY TRANSFORMATION (Sprint 3 alternative)
# =============================================================================

def transform_query(query: str, strategy: str = "expansion") -> List[str]:
    if strategy == "expansion":
        prompt = f"""Given the Vietnamese query: '{query}'
Generate 2-3 alternative phrasings or related terms in Vietnamese to improve search recall.
Keep them short and factual. Output only a JSON array of strings.
Example: ["SLA ticket P1", "thời gian xử lý sự cố P1", "cam kết dịch vụ P1"]"""
        
        try:
            response_text = call_llm(prompt)
            match = re.search(r"(\[.*\])", response_text.replace("\n", " "), re.DOTALL)
            if match:
                expanded_queries = json.loads(match.group(1))
                return [query] + expanded_queries
        except Exception as e:
            print(f"[transform_query] Error: {e}")
            
    return [query]


# =============================================================================
# GENERATION — GROUNDED ANSWER FUNCTION
# =============================================================================

def build_context_block(chunks: List[Dict[str, Any]]) -> str:
    """
    Đóng gói danh sách chunks thành context block để đưa vào prompt.

    Format: structured snippets với source, section, score (từ slide).
    Mỗi chunk có số thứ tự [1], [2], ... để model dễ trích dẫn.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", "unknown")
        section = meta.get("section", "")
        score = chunk.get("score", 0)
        text = chunk.get("text", "")

        # TODO: Tùy chỉnh format nếu muốn (thêm effective_date, department, ...)
        header = f"[{i}] {source}"
        if section:
            header += f" | {section}"
        if score > 0:
            header += f" | score={score:.2f}"

        context_parts.append(f"{header}\n{text}")

    return "\n\n".join(context_parts)


def build_grounded_prompt(query: str, context_block: str) -> str:
    """
    Xây dựng grounded prompt theo 4 quy tắc từ slide:
    1. Evidence-only: Chỉ trả lời từ retrieved context
    2. Abstain: Thiếu context thì nói không đủ dữ liệu
    3. Citation: Gắn source/section khi có thể
    4. Short, clear, stable: Output ngắn, rõ, nhất quán

    TODO Sprint 2:
    Đây là prompt baseline. Trong Sprint 3, bạn có thể:
    - Thêm hướng dẫn về format output (JSON, bullet points)
    - Thêm ngôn ngữ phản hồi (tiếng Việt vs tiếng Anh)
    - Điều chỉnh tone phù hợp với use case (CS helpdesk, IT support)
    """
    prompt = f"""Answer only from the retrieved context below.
If the context is insufficient to answer the question, explicitly say "Không đủ dữ liệu trong tài liệu hiện có." Do not make up information.
Cite the source field (in brackets like [1]) when possible.
Keep your answer short, clear, and factual.
Respond in the same language as the question.

Question: {query}

Context:
{context_block}

Answer:"""
    return prompt


def get_openai_client():
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = None
    if api_key and api_key.startswith("sk-or-v1-"):
        base_url = "https://openrouter.ai/api/v1"
    return OpenAI(api_key=api_key, base_url=base_url)

def call_llm(prompt: str) -> str:
    if LLM_PROVIDER != "openai":
        raise ValueError(f"Unsupported LLM_PROVIDER='{LLM_PROVIDER}'.")

    client = get_openai_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=512,
    )
    return response.choices[0].message.content


def rag_answer(
    query: str,
    retrieval_mode: str = "dense",
    top_k_search: int = TOP_K_SEARCH,
    top_k_select: int = TOP_K_SELECT,
    use_rerank: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Pipeline RAG hoàn chỉnh: query → retrieve → (rerank) → generate.

    Args:
        query: Câu hỏi
        retrieval_mode: "dense" | "sparse" | "hybrid"
        top_k_search: Số chunk lấy từ vector store (search rộng)
        top_k_select: Số chunk đưa vào prompt (sau rerank/select)
        use_rerank: Có dùng cross-encoder rerank không
        verbose: In thêm thông tin debug

    Returns:
        Dict với:
          - "answer": câu trả lời grounded
          - "sources": list source names trích dẫn
          - "chunks_used": list chunks đã dùng
          - "query": query gốc
          - "config": cấu hình pipeline đã dùng

    TODO Sprint 2 — Implement pipeline cơ bản:
    1. Chọn retrieval function dựa theo retrieval_mode
    2. Gọi rerank() nếu use_rerank=True
    3. Truncate về top_k_select chunks
    4. Build context block và grounded prompt
    5. Gọi call_llm() để sinh câu trả lời
    6. Trả về kết quả kèm metadata

    TODO Sprint 3 — Thử các variant:
    - Variant A: đổi retrieval_mode="hybrid"
    - Variant B: bật use_rerank=True
    - Variant C: thêm query transformation trước khi retrieve
    """
    config = {
        "retrieval_mode": retrieval_mode,
        "top_k_search": top_k_search,
        "top_k_select": top_k_select,
        "use_rerank": use_rerank,
    }

    query_variants = transform_query(query) if retrieval_mode == "hybrid" else [query]
    pooled_candidates: List[Dict[str, Any]] = []

    for query_variant in query_variants:
        if retrieval_mode == "dense":
            pooled_candidates.extend(retrieve_dense(query_variant, top_k=top_k_search))
        elif retrieval_mode == "sparse":
            pooled_candidates.extend(retrieve_sparse(query_variant, top_k=top_k_search))
        elif retrieval_mode == "hybrid":
            pooled_candidates.extend(retrieve_hybrid(query_variant, top_k=top_k_search))
        else:
            raise ValueError(f"retrieval_mode không hợp lệ: {retrieval_mode}")

    deduped_candidates = {}
    for candidate in pooled_candidates:
        meta = candidate.get("metadata", {})
        key = (meta.get("source", ""), meta.get("section", ""), candidate.get("text", ""))
        existing = deduped_candidates.get(key)
        if existing is None or candidate.get("score", 0) > existing.get("score", 0):
            deduped_candidates[key] = candidate

    candidates = sorted(
        deduped_candidates.values(),
        key=lambda item: item.get("score", 0),
        reverse=True,
    )

    if verbose:
        print(f"\n[RAG] Query: {query}")
        print(f"[RAG] Retrieved {len(candidates)} candidates (mode={retrieval_mode})")
        for i, c in enumerate(candidates[:3]):
            print(f"  [{i+1}] score={c.get('score', 0):.3f} | {c['metadata'].get('source', '?')}")

    # --- Bước 2: Rerank (optional) ---
    if use_rerank:
        candidates = rerank(query, candidates, top_k=top_k_select)
    else:
        candidates = candidates[:top_k_select]

    if verbose:
        print(f"[RAG] After select: {len(candidates)} chunks")

    if not candidates:
        return {
            "query": query,
            "answer": "Không đủ dữ liệu trong tài liệu hiện có.",
            "sources": [],
            "chunks_used": [],
            "config": config,
        }

    # --- Bước 3: Build context và prompt ---
    context_block = build_context_block(candidates)
    prompt = build_grounded_prompt(query, context_block)

    if verbose:
        print(f"\n[RAG] Prompt:\n{prompt[:500]}...\n")

    # --- Bước 4: Generate ---
    answer = call_llm(prompt)

    # --- Bước 5: Extract sources ---
    sources = list({
        c["metadata"].get("source", "unknown")
        for c in candidates
    })

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "chunks_used": candidates,
        "config": config,
    }


# =============================================================================
# SPRINT 3: SO SÁNH BASELINE VS VARIANT
# =============================================================================

def compare_retrieval_strategies(query: str) -> None:
    """
    So sánh các retrieval strategies với cùng một query.

    TODO Sprint 3:
    Chạy hàm này để thấy sự khác biệt giữa dense, sparse, hybrid.
    Dùng để justify tại sao chọn variant đó cho Sprint 3.

    A/B Rule (từ slide): Chỉ đổi MỘT biến mỗi lần.
    """
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print('='*60)

    strategies = [
        {"mode": "dense", "rerank": False},
        {"mode": "hybrid", "rerank": False},
        {"mode": "hybrid", "rerank": True}
    ]

    for strat in strategies:
        mode = strat["mode"]
        use_rr = strat["rerank"]
        print(f"\n--- Strategy: {mode} (Rerank={use_rr}) ---")
        try:
            result = rag_answer(query, retrieval_mode=mode, use_rerank=use_rr, verbose=False)
            print(f"Answer: {result['answer']}")
            print(f"Sources: {result['sources']}")
        except NotImplementedError as e:
            print(f"Chưa implement: {e}")
        except Exception as e:
            print(f"Lỗi: {e}")


# =============================================================================
# MAIN — Demo và Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sprint 2 + 3: RAG Answer Pipeline")
    print("=" * 60)

    # Test queries từ data/test_questions.json
    test_queries = [
        "SLA xử lý ticket P1 là bao lâu?",
        "Khách hàng có thể yêu cầu hoàn tiền trong bao nhiêu ngày?",
        "Ai phải phê duyệt để cấp quyền Level 3?",
        "ERR-403-AUTH là lỗi gì?",  # Query không có trong docs → kiểm tra abstain
    ]

    print("\n--- Sprint 2: Test Baseline (Dense) ---")
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            result = rag_answer(query, retrieval_mode="dense", verbose=True)
            print(f"Answer: {result['answer']}")
            print(f"Sources: {result['sources']}")
        except NotImplementedError:
            print("Chưa implement — hoàn thành TODO trong retrieve_dense() và call_llm() trước.")
        except Exception as e:
            print(f"Lỗi: {e}")

    # --- Sprint 3: So sánh strategies ---
    print("\n" + "="*60)
    print("--- Sprint 3: So sánh strategies ---")
    print("="*60)
    compare_retrieval_strategies("SLA xử lý ticket P1 là bao lâu?")
    compare_retrieval_strategies("Ai phê duyệt cấp quyền Level 3?")
    compare_retrieval_strategies("ERR-403-AUTH")

    print("\n\nViệc cần làm Sprint 2:")
    print("  1. Implement retrieve_dense() — query ChromaDB")
    print("  2. Implement call_llm() — gọi OpenAI hoặc Gemini")
    print("  3. Chạy rag_answer() với 3+ test queries")
    print("  4. Verify: output có citation không? Câu không có docs → abstain không?")

    print("\nViệc cần làm Sprint 3:")
    print("  1. Chọn 1 trong 3 variants: hybrid, rerank, hoặc query transformation")
    print("  2. Implement variant đó")
    print("  3. Chạy compare_retrieval_strategies() để thấy sự khác biệt")
    print("  4. Ghi lý do chọn biến đó vào docs/tuning-log.md")
