"""
Structure-aware, context-enhanced chunking for TelecomTS knowledge sources.

The implementation intentionally keeps dependencies light:
- Markdown/reStructuredText are split by heading hierarchy.
- Config-like files are split by logical blocks.
- PDF parsing is optional through pypdf.
- Semantic packing uses paragraph/block boundaries and lexical drift.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


TOKEN_RE = re.compile(
    r"[A-Za-z0-9_][A-Za-z0-9_./:+#-]*|[\u4e00-\u9fff]|[^\s]",
    re.UNICODE,
)

MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
RST_UNDERLINE_RE = re.compile(r"^([=\-~^\"'`:#*+])\1{2,}\s*$")

DOMAIN_KEYWORDS = [
    "5g",
    "ran",
    "o-ran",
    "oran",
    "nr",
    "gnb",
    "enb",
    "cu",
    "du",
    "cu-cp",
    "cu-up",
    "rrc",
    "mac",
    "phy",
    "f1ap",
    "e1ap",
    "nfapi",
    "a1",
    "e2",
    "ric",
    "xapp",
    "kpimon",
    "kpi",
    "prb",
    "bler",
    "rsrp",
    "rsrq",
    "sinr",
    "throughput",
    "latency",
    "handover",
    "slice",
    "qos",
    "open5gs",
    "telegraf",
    "grafana",
    "rag",
    "embedding",
    "bge",
    "bge-m3",
    "rerank",
    "reranker",
    "bm25",
    "chroma",
    "chunking",
    "late chunking",
    "contextual retrieval",
]


@dataclass(frozen=True)
class ChunkConfig:
    max_tokens: int = 900
    min_tokens: int = 160
    target_tokens: int = 700
    overlap_tokens: int = 120
    context_prefix_tokens: int = 100
    semantic_break_threshold: float = 0.10


@dataclass
class Section:
    text: str
    title: str
    section_path: List[str]
    page: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    content: str
    context_prefix: str
    token_count: int
    content_token_count: int
    metadata: Dict[str, Any]


def estimate_tokens(text: str) -> int:
    return len(TOKEN_RE.findall(text or ""))


def token_list(text: str) -> List[str]:
    return TOKEN_RE.findall(text or "")


def trim_to_tokens(text: str, max_tokens: int) -> str:
    tokens = token_list(text)
    if len(tokens) <= max_tokens:
        return text.strip()
    return " ".join(tokens[:max_tokens]).strip()


def tail_by_tokens(text: str, max_tokens: int) -> str:
    tokens = token_list(text)
    if len(tokens) <= max_tokens:
        return text.strip()
    return " ".join(tokens[-max_tokens:]).strip()


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def safe_read_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gbk", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def load_source_manifest(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    mapping: Dict[str, Dict[str, Any]] = {}
    for item in data.get("files", []):
        key = str(item.get("knowledge_path", "")).replace("\\", "/")
        if key:
            mapping[key] = item
    return mapping


def relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def infer_title(path: Path, rel_path: str) -> str:
    name = path.stem.replace("_", " ").replace("-", " ").strip()
    if name.lower() in {"readme", "index"}:
        parts = [p for p in Path(rel_path).parts if p not in {"README.md", "index.rst"}]
        if parts:
            return parts[-1].replace("_", " ").replace("-", " ")
    return name or path.name


def parse_markdown_sections(text: str, title: str) -> List[Section]:
    sections: List[Section] = []
    current_lines: List[str] = []
    heading_stack: List[str] = []
    in_fence = False

    def flush() -> None:
        content = normalize_text("\n".join(current_lines))
        if content:
            sections.append(
                Section(
                    text=content,
                    title=title,
                    section_path=heading_stack[:] or [title],
                )
            )
        current_lines.clear()

    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
        match = MARKDOWN_HEADING_RE.match(line) if not in_fence else None
        if match:
            flush()
            level = len(match.group(1))
            heading = clean_heading(match.group(2))
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(heading)
            current_lines.append(line)
        else:
            current_lines.append(line)
    flush()
    return sections or [Section(text=normalize_text(text), title=title, section_path=[title])]


def parse_rst_sections(text: str, title: str) -> List[Section]:
    lines = text.splitlines()
    heading_levels: Dict[str, int] = {}
    stack: List[str] = []
    sections: List[Section] = []
    current: List[str] = []
    i = 0

    def flush() -> None:
        content = normalize_text("\n".join(current))
        if content:
            sections.append(Section(text=content, title=title, section_path=stack[:] or [title]))
        current.clear()

    while i < len(lines):
        line = lines[i]
        next_line = lines[i + 1] if i + 1 < len(lines) else ""
        if line.strip() and RST_UNDERLINE_RE.match(next_line):
            marker = next_line.strip()[0]
            if marker not in heading_levels:
                heading_levels[marker] = len(heading_levels) + 1
            level = heading_levels[marker]
            flush()
            stack = stack[: level - 1]
            stack.append(clean_heading(line))
            current.extend([line, next_line])
            i += 2
            continue
        current.append(line)
        i += 1

    flush()
    return sections or [Section(text=normalize_text(text), title=title, section_path=[title])]


def parse_plain_sections(text: str, title: str, path: Path) -> List[Section]:
    ext = path.suffix.lower()
    section_path = [title]
    if ext in {".yaml", ".yml"}:
        section_path.append("YAML configuration")
    elif ext == ".json":
        section_path.append("JSON schema or dashboard")
    elif ext in {".conf", ".cfg", ".ini"}:
        section_path.append("runtime configuration")
    elif ext == ".toml":
        section_path.append("TOML configuration")
    return [Section(text=normalize_text(text), title=title, section_path=section_path)]


def parse_pdf_sections(path: Path, title: str) -> List[Section]:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PDF parsing requires pypdf. Install it with `pip install pypdf`.") from exc

    reader = PdfReader(str(path))
    sections: List[Section] = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = normalize_text(page.extract_text() or "")
        if not text:
            continue
        inferred = infer_pdf_section(text, title)
        sections.append(
            Section(
                text=text,
                title=title,
                section_path=[title, inferred] if inferred else [title],
                page=page_index,
            )
        )
    return sections


def infer_pdf_section(text: str, fallback: str) -> str:
    for line in text.splitlines()[:12]:
        clean = clean_heading(line)
        if 4 <= len(clean) <= 90 and not clean.endswith("."):
            return clean
    return fallback


def clean_heading(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[\*_`#]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_source_file(path: Path, rel_path: str, include_pdf: bool = True) -> List[Section]:
    title = infer_title(path, rel_path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        if not include_pdf:
            return []
        return parse_pdf_sections(path, title)

    text = normalize_text(safe_read_text(path))
    if not text:
        return []
    if ext == ".md":
        return parse_markdown_sections(text, title)
    if ext == ".rst":
        return parse_rst_sections(text, title)
    return parse_plain_sections(text, title, path)


def split_semantic_blocks(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []

    blocks: List[str] = []
    current: List[str] = []
    in_fence = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            current.append(line)
            continue
        if not in_fence and stripped == "":
            if current:
                blocks.append(normalize_text("\n".join(current)))
                current = []
            continue
        if not in_fence and is_config_boundary(stripped) and current:
            blocks.append(normalize_text("\n".join(current)))
            current = [line]
            continue
        current.append(line)
    if current:
        blocks.append(normalize_text("\n".join(current)))

    return [block for block in blocks if block]


def is_config_boundary(line: str) -> bool:
    if not line:
        return False
    if MARKDOWN_HEADING_RE.match(line):
        return True
    if re.match(r"^\[[A-Za-z0-9_.:-]+\]$", line):
        return True
    if re.match(r"^[A-Za-z0-9_.-]+:\s*$", line):
        return True
    if re.match(r"^#{1,3}\s+[A-Za-z0-9]", line):
        return True
    return False


def lexical_terms(text: str) -> set[str]:
    terms = set()
    for token in token_list(text.lower()):
        if len(token) < 2:
            continue
        if token in {"the", "and", "for", "with", "this", "that", "from", "into"}:
            continue
        terms.add(token)
    return terms


def lexical_similarity(a: str, b: str) -> float:
    left = lexical_terms(a)
    right = lexical_terms(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def split_large_block(block: str, max_tokens: int, overlap_tokens: int) -> List[str]:
    tokens = token_list(block)
    if len(tokens) <= max_tokens:
        return [block]
    pieces: List[str] = []
    stride = max(1, max_tokens - overlap_tokens)
    for start in range(0, len(tokens), stride):
        piece = " ".join(tokens[start : start + max_tokens]).strip()
        if piece:
            pieces.append(piece)
        if start + max_tokens >= len(tokens):
            break
    return pieces


def pack_blocks(blocks: Sequence[str], config: ChunkConfig) -> List[str]:
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        chunks.append(normalize_text("\n\n".join(current)))
        overlap = overlap_blocks(current, config.overlap_tokens)
        current = overlap
        current_tokens = estimate_tokens("\n\n".join(current))

    for block in blocks:
        sub_blocks = split_large_block(block, config.max_tokens, config.overlap_tokens)
        for sub_block in sub_blocks:
            block_tokens = estimate_tokens(sub_block)
            if current and current_tokens + block_tokens > config.max_tokens:
                if current_tokens >= config.min_tokens:
                    flush()
                else:
                    current = []
                    current_tokens = 0
                if current and current_tokens + block_tokens > config.max_tokens:
                    current = []
                    current_tokens = 0

            if current:
                drift = lexical_similarity(current[-1], sub_block)
                should_break = (
                    current_tokens >= config.target_tokens
                    and drift < config.semantic_break_threshold
                )
                if should_break:
                    flush()

            current.append(sub_block)
            current_tokens += block_tokens

            if current_tokens >= config.max_tokens:
                flush()

    if current:
        final = normalize_text("\n\n".join(current))
        if final and (estimate_tokens(final) >= config.min_tokens or not chunks) and (
            not chunks or final != chunks[-1]
        ):
            chunks.append(final)

    return dedupe_preserve_order(chunks)


def overlap_blocks(blocks: Sequence[str], overlap_tokens: int) -> List[str]:
    if overlap_tokens <= 0:
        return []
    selected: List[str] = []
    count = 0
    for block in reversed(blocks):
        block_tokens = estimate_tokens(block)
        if block_tokens > overlap_tokens and not selected:
            return [tail_by_tokens(block, overlap_tokens)]
        if count + block_tokens > overlap_tokens and selected:
            break
        selected.insert(0, block)
        count += block_tokens
    return selected


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        key = hashlib.sha1(item.encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def infer_topic(repo: str, group: str, source_path: str, section_path: Sequence[str]) -> str:
    text = " ".join([repo, group, source_path, *section_path]).lower()
    if group == "project_glossary" or repo == "TelecomTS_glossary":
        return "KPI and anomaly troubleshooting glossary"
    if repo == "FlagEmbedding":
        if "rerank" in text:
            return "RAG reranking and retrieval quality"
        if "bge_m3" in text or "bge-m3" in text:
            return "BGE-M3 multilingual embedding and hybrid retrieval"
        return "RAG embedding, indexing, and evaluation"
    if repo == "late-chunking":
        return "long-context embedding and late chunking"
    if "kpimon" in text:
        return "O-RAN KPI monitoring xApp"
    if "a1" in text or repo == "ric-plt-a1":
        return "O-RAN A1 policy interface"
    if "ric" in text:
        return "near-RT RIC integration"
    if "grafana" in text or "telegraf" in text:
        return "5G observability and metrics collection"
    if "gnb" in text or "cu" in text or "du" in text:
        return "5G RAN gNB/CU/DU configuration"
    return "5G network operations knowledge"


def extract_keywords(*texts: str, limit: int = 8) -> List[str]:
    joined = "\n".join(texts).lower()
    scored: List[Tuple[int, str]] = []
    for keyword in DOMAIN_KEYWORDS:
        count = joined.count(keyword.lower())
        if count:
            scored.append((count, keyword.upper() if keyword.islower() and len(keyword) <= 5 else keyword))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [keyword for _, keyword in scored[:limit]]


def build_context_prefix(
    repo: str,
    group: str,
    source_path: str,
    title: str,
    section_path: Sequence[str],
    content: str,
    config: ChunkConfig,
) -> str:
    topic = infer_topic(repo, group, source_path, section_path)
    section = " > ".join([part for part in section_path if part])
    keywords = extract_keywords(source_path, section, content)
    kw_text = ", ".join(keywords) if keywords else "5G/RAG"
    prefix = (
        f"上下文：来源 {repo}，类别 {group}，主题 {topic}，"
        f"章节 {section or title}，路径 {source_path}，关键词 {kw_text}。"
    )
    return trim_to_tokens(prefix, config.context_prefix_tokens)


def build_chunk_id(metadata: Dict[str, Any], index: int, content: str) -> str:
    base = "|".join(
        [
            str(metadata.get("repo", "")),
            str(metadata.get("source_path", "")),
            str(metadata.get("page", "")),
            str(index),
            hashlib.sha1(content.encode("utf-8")).hexdigest()[:12],
        ]
    )
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def chunks_for_section(
    section: Section,
    base_metadata: Dict[str, Any],
    config: ChunkConfig,
    start_index: int,
) -> Tuple[List[Chunk], int]:
    blocks = split_semantic_blocks(section.text)
    packed = pack_blocks(blocks, config)
    chunks: List[Chunk] = []
    index = start_index
    for content in packed:
        if estimate_tokens(content) < 20:
            continue
        metadata = dict(base_metadata)
        repo = str(metadata.get("repo", "project"))
        source_path = str(metadata.get("source_path", metadata.get("knowledge_path", "")))
        metadata.update(
            {
                "source": f"{repo}:{source_path}",
                "document_id": hashlib.sha1(f"{repo}|{source_path}".encode("utf-8")).hexdigest(),
                "title": section.title,
                "section": " > ".join(section.section_path),
                "section_path": list(section.section_path),
                "page": section.page,
                "chunk_index": index,
                "chunking_strategy": "structure_semantic_contextual_v1",
                "max_tokens": config.max_tokens,
                "overlap_tokens": config.overlap_tokens,
                "context_prefix_tokens": config.context_prefix_tokens,
            }
        )
        prefix = build_context_prefix(
            repo=str(metadata.get("repo", "project")),
            group=str(metadata.get("group", "project")),
            source_path=str(metadata.get("source_path", metadata.get("knowledge_path", ""))),
            title=section.title,
            section_path=section.section_path,
            content=content,
            config=config,
        )
        retrieval_text = f"{prefix}\n\n{content}".strip()
        chunk_id = build_chunk_id(metadata, index, content)
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=retrieval_text,
                content=content,
                context_prefix=prefix,
                token_count=estimate_tokens(retrieval_text),
                content_token_count=estimate_tokens(content),
                metadata=metadata,
            )
        )
        index += 1
    return chunks, index


def iter_source_files(source_dir: Path) -> Iterator[Path]:
    allowed = {".md", ".rst", ".txt", ".yaml", ".yml", ".json", ".toml", ".ini", ".conf", ".cfg"}
    for path in sorted(source_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in allowed:
            yield path


def extra_file_metadata(path: Path, project_root: Path) -> Dict[str, Any]:
    rel = relative_to_root(path, project_root)
    if path.suffix.lower() == ".pdf":
        repo = "TelecomTS_paper"
        group = "project_paper"
        source_path = rel
    elif "project_glossary" in rel.replace("\\", "/"):
        repo = "TelecomTS_glossary"
        group = "project_glossary"
        source_path = rel
    else:
        repo = "project"
        group = "project"
        source_path = rel
    return {
        "group": group,
        "repo": repo,
        "source_url": None,
        "commit": None,
        "source_path": source_path,
        "knowledge_path": rel,
        "bytes": path.stat().st_size if path.exists() else None,
    }


def build_chunks(
    source_dir: Path,
    manifest_path: Path,
    project_root: Path,
    config: ChunkConfig,
    extra_files: Optional[Sequence[Path]] = None,
    include_pdf: bool = True,
) -> List[Chunk]:
    manifest = load_source_manifest(manifest_path)
    chunks: List[Chunk] = []
    chunk_index = 0

    for path in iter_source_files(source_dir):
        rel = relative_to_root(path, project_root)
        metadata = dict(manifest.get(rel, {}))
        if not metadata:
            metadata = extra_file_metadata(path, project_root)
        metadata.setdefault("knowledge_path", rel)
        metadata["file_name"] = path.name
        metadata["file_ext"] = path.suffix.lower()
        sections = parse_source_file(path, rel, include_pdf=False)
        for section in sections:
            new_chunks, chunk_index = chunks_for_section(section, metadata, config, chunk_index)
            chunks.extend(new_chunks)

    for extra in extra_files or []:
        if not extra.exists():
            continue
        rel = relative_to_root(extra, project_root)
        metadata = extra_file_metadata(extra, project_root)
        metadata["file_name"] = extra.name
        metadata["file_ext"] = extra.suffix.lower()
        sections = parse_source_file(extra, rel, include_pdf=include_pdf)
        for section in sections:
            new_chunks, chunk_index = chunks_for_section(section, metadata, config, chunk_index)
            chunks.extend(new_chunks)

    return chunks


def write_chunks(chunks: Sequence[Chunk], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        for chunk in chunks:
            record = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "content": chunk.content,
                "context_prefix": chunk.context_prefix,
                "token_count": chunk.token_count,
                "content_token_count": chunk.content_token_count,
                "metadata": chunk.metadata,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize_chunks(chunks: Sequence[Chunk]) -> Dict[str, Any]:
    by_group: Dict[str, int] = {}
    by_repo: Dict[str, int] = {}
    token_counts = [chunk.content_token_count for chunk in chunks]
    for chunk in chunks:
        group = str(chunk.metadata.get("group", "unknown"))
        repo = str(chunk.metadata.get("repo", "unknown"))
        by_group[group] = by_group.get(group, 0) + 1
        by_repo[repo] = by_repo.get(repo, 0) + 1

    if token_counts:
        avg_tokens = round(sum(token_counts) / len(token_counts), 2)
        min_tokens = min(token_counts)
        max_tokens = max(token_counts)
    else:
        avg_tokens = min_tokens = max_tokens = 0

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "chunk_count": len(chunks),
        "by_group": dict(sorted(by_group.items())),
        "by_repo": dict(sorted(by_repo.items())),
        "content_tokens": {
            "min": min_tokens,
            "avg": avg_tokens,
            "max": max_tokens,
        },
        "strategy": "structure_semantic_contextual_v1",
    }


def write_chunk_manifest(chunks: Sequence[Chunk], output_path: Path, config: ChunkConfig) -> None:
    manifest = summarize_chunks(chunks)
    manifest["config"] = {
        "max_tokens": config.max_tokens,
        "min_tokens": config.min_tokens,
        "target_tokens": config.target_tokens,
        "overlap_tokens": config.overlap_tokens,
        "context_prefix_tokens": config.context_prefix_tokens,
        "semantic_break_threshold": config.semantic_break_threshold,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
