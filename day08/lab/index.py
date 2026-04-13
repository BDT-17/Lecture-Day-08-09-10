"""
index.py — Sprint 1: Build RAG Index
====================================
Mục tiêu Sprint 1 (60 phút):
  - Đọc và preprocess tài liệu từ data/docs/
  - Chunk tài liệu theo cấu trúc tự nhiên (heading/section)
  - Gắn metadata: source, section, department, effective_date, access
  - Embed và lưu vào vector store (ChromaDB)

Definition of Done Sprint 1:
  ✓ Script chạy được và index đủ docs
  ✓ Có ít nhất 3 metadata fields hữu ích cho retrieval
  ✓ Có thể kiểm tra chunk bằng list_chunks()
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any
import sys
from dotenv import load_dotenv

# Đảm bảo terminal Windows hiển thị được tiếng Việt
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

# =============================================================================
# CẤU HÌNH
# =============================================================================

DOCS_DIR = Path(__file__).parent / "data" / "docs"
CHROMA_DB_DIR = Path(
    os.getenv("CHROMA_DB_DIR", str(Path(__file__).parent / "chroma_db"))
)
FALLBACK_STORE_PATH = CHROMA_DB_DIR / "rag_store.json"
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LOCAL_EMBEDDING_MODEL = os.getenv(
    "LOCAL_EMBEDDING_MODEL",
    "paraphrase-multilingual-MiniLM-L12-v2",
)

# TODO Sprint 1: Điều chỉnh chunk size và overlap theo quyết định của nhóm
# Gợi ý từ slide: chunk 300-500 tokens, overlap 50-80 tokens
CHUNK_SIZE = 400       # tokens (ước lượng bằng số ký tự / 4)
CHUNK_OVERLAP = 80     # tokens overlap giữa các chunk


# =============================================================================
# STEP 1: PREPROCESS
# Làm sạch text trước khi chunk và embed
# =============================================================================

def preprocess_document(raw_text: str, filepath: str) -> Dict[str, Any]:
    """
    Preprocess một tài liệu: extract metadata từ header và làm sạch nội dung.

    Args:
        raw_text: Toàn bộ nội dung file text
        filepath: Đường dẫn file để làm source mặc định

    Returns:
        Dict chứa:
          - "text": nội dung đã clean
          - "metadata": dict với source, department, effective_date, access

    TODO Sprint 1:
    - Extract metadata từ dòng đầu file (Source, Department, Effective Date, Access)
    - Bỏ các dòng header metadata khỏi nội dung chính
    - Normalize khoảng trắng, xóa ký tự rác

    Gợi ý: dùng regex để parse dòng "Key: Value" ở đầu file.
    """
    lines = raw_text.strip().split("\n")
    metadata = {
        "source": filepath,
        "section": "",
        "department": "unknown",
        "effective_date": "unknown",
        "access": "internal",
    }
    content_lines = []
    header_done = False

    for line in lines:
        if not header_done:
            # TODO: Parse metadata từ các dòng "Key: Value"
            # Ví dụ: "Source: policy/refund-v4.pdf" → metadata["source"] = "policy/refund-v4.pdf"
            if line.startswith("Source:"):
                metadata["source"] = line.replace("Source:", "").strip()
            elif line.startswith("Department:"):
                metadata["department"] = line.replace("Department:", "").strip()
            elif line.startswith("Effective Date:"):
                metadata["effective_date"] = line.replace("Effective Date:", "").strip()
            elif line.startswith("Access:"):
                metadata["access"] = line.replace("Access:", "").strip()
            elif line.startswith("==="):
                # Gặp section heading đầu tiên → kết thúc header
                header_done = True
                content_lines.append(line)
            elif line.strip() == "" or line.isupper():
                # Dòng tên tài liệu (toàn chữ hoa) hoặc dòng trống
                continue
            else:
                # Giữ lại preamble hữu ích trước heading đầu tiên, ví dụ alias/tên cũ của tài liệu.
                content_lines.append(line)
        else:
            content_lines.append(line)

    cleaned_text = "\n".join(content_lines)
    cleaned_text = cleaned_text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    return {
        "text": cleaned_text,
        "metadata": metadata,
    }


# =============================================================================
# STEP 2: CHUNK
# Chia tài liệu thành các đoạn nhỏ theo cấu trúc tự nhiên
# =============================================================================

def chunk_document(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Chunk một tài liệu đã preprocess thành danh sách các chunk nhỏ.

    Args:
        doc: Dict với "text" và "metadata" (output của preprocess_document)

    Returns:
        List các Dict, mỗi dict là một chunk với:
          - "text": nội dung chunk
          - "metadata": metadata gốc + "section" của chunk đó

    TODO Sprint 1:
    1. Split theo heading "=== Section ... ===" hoặc "=== Phần ... ===" trước
    2. Nếu section quá dài (> CHUNK_SIZE * 4 ký tự), split tiếp theo paragraph
    3. Thêm overlap: lấy đoạn cuối của chunk trước vào đầu chunk tiếp theo
    4. Mỗi chunk PHẢI giữ metadata đầy đủ từ tài liệu gốc

    Gợi ý: Ưu tiên cắt tại ranh giới tự nhiên (section, paragraph)
    thay vì cắt theo token count cứng.
    """
    text = doc["text"]
    base_metadata = doc["metadata"].copy()
    chunks = []

    # TODO: Implement chunking theo section heading
    # Bước 1: Split theo heading pattern "=== ... ==="
    sections = re.split(r"(===.*?===)", text)

    current_section = "General"
    current_section_text = ""

    for part in sections:
        if re.match(r"===.*?===", part):
            # Lưu section trước (nếu có nội dung)
            if current_section_text.strip():
                section_chunks = _split_by_size(
                    current_section_text.strip(),
                    base_metadata=base_metadata,
                    section=current_section,
                )
                chunks.extend(section_chunks)
            # Bắt đầu section mới
            current_section = part.strip("= ").strip()
            current_section_text = ""
        else:
            current_section_text += part

    # Lưu section cuối cùng
    if current_section_text.strip():
        section_chunks = _split_by_size(
            current_section_text.strip(),
            base_metadata=base_metadata,
            section=current_section,
        )
        chunks.extend(section_chunks)

    return chunks


def _split_by_size(
    text: str,
    base_metadata: Dict,
    section: str,
    chunk_chars: int = CHUNK_SIZE * 4,
    overlap_chars: int = CHUNK_OVERLAP * 4,
) -> List[Dict[str, Any]]:
    """
    Helper: Split text dài thành chunks với overlap.

    TODO Sprint 1:
    Hiện tại dùng split đơn giản theo ký tự.
    Cải thiện: split theo paragraph (\n\n) trước, rồi mới ghép đến khi đủ size.
    """
    normalized_text = text.strip()
    if len(normalized_text) <= chunk_chars:
        # Toàn bộ section vừa một chunk
        return [{
            "text": normalized_text,
            "metadata": {**base_metadata, "section": section},
        }]

    def make_chunk(chunk_text: str) -> Dict[str, Any]:
        return {
            "text": chunk_text.strip(),
            "metadata": {**base_metadata, "section": section},
        }

    def split_long_paragraph(paragraph: str) -> List[str]:
        pieces = []
        remaining = paragraph.strip()
        while remaining:
            if len(remaining) <= chunk_chars:
                pieces.append(remaining)
                break

            window = remaining[:chunk_chars]
            split_at = max(
                window.rfind("\n"),
                window.rfind(". "),
                window.rfind("; "),
                window.rfind(", "),
                window.rfind(" "),
            )
            if split_at < max(100, chunk_chars // 3):
                split_at = chunk_chars

            pieces.append(remaining[:split_at].strip())
            remaining = remaining[split_at:].strip()
        return [piece for piece in pieces if piece]

    def build_overlap_seed(chunk_text: str) -> str:
        if overlap_chars <= 0:
            return ""
        seed = chunk_text[-overlap_chars:].strip()
        return seed if len(seed) >= min(80, overlap_chars) else ""

    paragraphs = []
    for paragraph in normalized_text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) > chunk_chars:
            paragraphs.extend(split_long_paragraph(paragraph))
        else:
            paragraphs.append(paragraph)

    chunks = []
    current_parts: List[str] = []
    current_len = 0

    for paragraph in paragraphs:
        addition_len = len(paragraph) + (2 if current_parts else 0)
        if current_parts and current_len + addition_len > chunk_chars:
            chunk_text = "\n\n".join(current_parts)
            chunks.append(make_chunk(chunk_text))

            overlap_seed = build_overlap_seed(chunk_text)
            current_parts = [overlap_seed] if overlap_seed else []
            current_len = len(overlap_seed)

            if current_parts and current_len + len(paragraph) + 2 > chunk_chars:
                chunks.append(make_chunk(paragraph))
                current_parts = []
                current_len = 0
                continue

        if current_parts:
            current_len += 2 + len(paragraph)
        else:
            current_len = len(paragraph)
        current_parts.append(paragraph)

    if current_parts:
        chunks.append(make_chunk("\n\n".join(current_parts)))

    return [chunk for chunk in chunks if chunk["text"].strip()]


# =============================================================================
# OPTIONAL: OPENAI / OPENROUTER CLIENT HELPER
# =============================================================================

def get_openai_client():
    """
    Khởi tạo OpenAI client, tự động cấu hình cho OpenRouter nếu phát hiện key.
    """
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = None
    
    if api_key and api_key.startswith("sk-or-v1-"):
        base_url = "https://openrouter.ai/api/v1"
        
    return OpenAI(api_key=api_key, base_url=base_url)


# =============================================================================
# STEP 3: EMBED + STORE
# Embed các chunk và lưu vào ChromaDB
# =============================================================================

def get_embedding(text: str) -> List[float]:
    """
    Tạo embedding vector cho một đoạn text.
    Hỗ trợ Local (Sentence Transformers) hoặc API (OpenAI/OpenRouter).
    """
    provider = EMBEDDING_PROVIDER

    if provider in {"openai", "openrouter"}:
        client = get_openai_client()
        response = client.embeddings.create(
            input=text,
            model=EMBEDDING_MODEL,
        )
        return response.data[0].embedding

    if provider != "local":
        raise ValueError(
            f"Unsupported EMBEDDING_PROVIDER='{provider}'. Use 'openai' or 'local'."
        )

    from sentence_transformers import SentenceTransformer

    if not hasattr(get_embedding, "model"):
        get_embedding.model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
    return get_embedding.model.encode(text).tolist()


def build_index(docs_dir: Path = DOCS_DIR, db_dir: Path = CHROMA_DB_DIR) -> None:
    """
    Pipeline hoàn chỉnh: đọc docs → preprocess → chunk → embed → store.

    TODO Sprint 1:
    1. Cài thư viện: pip install chromadb
    2. Khởi tạo ChromaDB client và collection
    3. Với mỗi file trong docs_dir:
       a. Đọc nội dung
       b. Gọi preprocess_document()
       c. Gọi chunk_document()
       d. Với mỗi chunk: gọi get_embedding() và upsert vào ChromaDB
    4. In số lượng chunk đã index

    Gợi ý khởi tạo ChromaDB:
        import chromadb
        client = chromadb.PersistentClient(path=str(db_dir))
        collection = client.get_or_create_collection(
            name="rag_lab",
            metadata={"hnsw:space": "cosine"}
        )
    """
    print(f"Đang build index từ: {docs_dir}")
    db_dir.mkdir(parents=True, exist_ok=True)
    collection = None
    use_fallback_store = False
    fallback_rows = []

    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(db_dir))
        try:
            client.delete_collection("rag_lab")
        except Exception:
            pass
        collection = client.get_or_create_collection(
            name="rag_lab",
            metadata={"hnsw:space": "cosine"}
        )
    except Exception as exc:
        use_fallback_store = True
        print(f"ChromaDB không khả dụng, dùng fallback JSON store. Lý do: {exc}")

    total_chunks = 0
    doc_files = list(docs_dir.glob("*.txt"))

    if not doc_files:
        print(f"Không tìm thấy file .txt trong {docs_dir}")
        return

    for filepath in doc_files:
        print(f"  Processing: {filepath.name}")
        raw_text = filepath.read_text(encoding="utf-8")

        doc = preprocess_document(raw_text, str(filepath))
        chunks = chunk_document(doc)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{filepath.stem}_{i}"
            embedding = get_embedding(chunk["text"])
            if use_fallback_store:
                fallback_rows.append({
                    "id": chunk_id,
                    "embedding": embedding,
                    "document": chunk["text"],
                    "metadata": chunk["metadata"],
                })
            else:
                collection.upsert(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    documents=[chunk["text"]],
                    metadatas=[chunk["metadata"]],
                )
        total_chunks += len(chunks)

    if use_fallback_store:
        FALLBACK_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        FALLBACK_STORE_PATH.write_text(
            json.dumps(fallback_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"\nHoàn thành! Tổng số chunks: {total_chunks}")


def load_fallback_store(db_dir: Path = CHROMA_DB_DIR) -> List[Dict[str, Any]]:
    store_path = Path(db_dir) / "rag_store.json"
    if not store_path.exists():
        return []
    return json.loads(store_path.read_text(encoding="utf-8"))


# =============================================================================
# STEP 4: INSPECT / KIỂM TRA
# Dùng để debug và kiểm tra chất lượng index
# =============================================================================

def list_chunks(db_dir: Path = CHROMA_DB_DIR, n: int = 5) -> None:
    """
    In ra n chunk đầu tiên trong ChromaDB để kiểm tra chất lượng index.

    TODO Sprint 1:
    Implement sau khi hoàn thành build_index().
    Kiểm tra:
    - Chunk có giữ đủ metadata không? (source, section, effective_date)
    - Chunk có bị cắt giữa điều khoản không?
    - Metadata effective_date có đúng không?
    """
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(db_dir))
        collection = client.get_collection("rag_lab")
        results = collection.get(limit=n, include=["documents", "metadatas"])

        print(f"\n=== Top {n} chunks trong index ===\n")
        for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
            print(f"[Chunk {i+1}]")
            print(f"  Source: {meta.get('source', 'N/A')}")
            print(f"  Section: {meta.get('section', 'N/A')}")
            print(f"  Effective Date: {meta.get('effective_date', 'N/A')}")
            print(f"  Text preview: {doc[:120]}...")
            print()
    except Exception as e:
        fallback_rows = load_fallback_store(db_dir)
        if not fallback_rows:
            print(f"Lỗi khi đọc index: {e}")
            print("Hãy chạy build_index() trước.")
            return

        print(f"\n=== Top {min(n, len(fallback_rows))} chunks trong fallback store ===\n")
        for i, row in enumerate(fallback_rows[:n], 1):
            meta = row.get("metadata", {})
            doc = row.get("document", "")
            print(f"[Chunk {i}]")
            print(f"  Source: {meta.get('source', 'N/A')}")
            print(f"  Section: {meta.get('section', 'N/A')}")
            print(f"  Effective Date: {meta.get('effective_date', 'N/A')}")
            print(f"  Text preview: {doc[:120]}...")
            print()


def inspect_metadata_coverage(db_dir: Path = CHROMA_DB_DIR) -> None:
    """
    Kiểm tra phân phối metadata trong toàn bộ index.

    Checklist Sprint 1:
    - Mọi chunk đều có source?
    - Có bao nhiêu chunk từ mỗi department?
    - Chunk nào thiếu effective_date?

    TODO: Implement sau khi build_index() hoàn thành.
    """
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(db_dir))
        collection = client.get_collection("rag_lab")
        results = collection.get(include=["metadatas"])

        print(f"\nTổng chunks: {len(results['metadatas'])}")

        # TODO: Phân tích metadata
        # Đếm theo department, kiểm tra effective_date missing, v.v.
        departments = {}
        missing_date = 0
        for meta in results["metadatas"]:
            dept = meta.get("department", "unknown")
            departments[dept] = departments.get(dept, 0) + 1
            if meta.get("effective_date") in ("unknown", "", None):
                missing_date += 1

        print("Phân bố theo department:")
        for dept, count in departments.items():
            print(f"  {dept}: {count} chunks")
        print(f"Chunks thiếu effective_date: {missing_date}")

    except Exception as e:
        fallback_rows = load_fallback_store(db_dir)
        if not fallback_rows:
            print(f"Lỗi: {e}. Hãy chạy build_index() trước.")
            return

        print(f"\nTổng chunks: {len(fallback_rows)}")
        departments = {}
        missing_date = 0
        for row in fallback_rows:
            meta = row.get("metadata", {})
            dept = meta.get("department", "unknown")
            departments[dept] = departments.get(dept, 0) + 1
            if meta.get("effective_date") in ("unknown", "", None):
                missing_date += 1

        print("Phân bố theo department:")
        for dept, count in departments.items():
            print(f"  {dept}: {count} chunks")
        print(f"Chunks thiếu effective_date: {missing_date}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sprint 1: Build RAG Index")
    print("=" * 60)

    # Bước 1: Kiểm tra docs
    doc_files = list(DOCS_DIR.glob("*.txt"))
    print(f"\nTìm thấy {len(doc_files)} tài liệu:")
    for f in doc_files:
        print(f"  - {f.name}")

    # Bước 2: Test preprocess và chunking (không cần API key)
    print("\n--- Test preprocess + chunking ---")
    for filepath in doc_files[:1]:  # Test với 1 file đầu
        raw = filepath.read_text(encoding="utf-8")
        doc = preprocess_document(raw, str(filepath))
        chunks = chunk_document(doc)
        print(f"\nFile: {filepath.name}")
        print(f"  Metadata: {doc['metadata']}")
        print(f"  Số chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n  [Chunk {i+1}] Section: {chunk['metadata']['section']}")
            print(f"  Text: {chunk['text'][:150]}...")

    # Bước 3: Build index
    print("\n--- Build Full Index ---")
    build_index()

    # Bước 4: Kiểm tra index
    list_chunks()
    inspect_metadata_coverage()

    print("\nSprint 1 setup hoàn thành!")
    print("Việc cần làm:")
    print("  1. Implement get_embedding() - chọn OpenAI hoặc Sentence Transformers")
    print("  2. Implement phần TODO trong build_index()")
    print("  3. Chạy build_index() và kiểm tra với list_chunks()")
    print("  4. Nếu chunking chưa tốt: cải thiện _split_by_size() để split theo paragraph")
