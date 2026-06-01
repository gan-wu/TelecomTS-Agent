# Knowledge Base

This directory contains the curated materials used by the RAG pipeline.

## Contents

- `source_docs/`: selected telecom and RAG method documents from srsRAN, OpenAirInterface, O-RAN SC, FlagEmbedding, late-chunking, and the project glossary.
- `chunks/knowledge_chunks.jsonl`: contextual chunks built from the source documents and TelecomTS paper.
- `chunks/chunk_manifest.json`: chunking metadata.
- `manifests/source_manifest.json`: source repository and file provenance.
- `manifests/extraction_summary.md`: extraction summary.
- `retrieval/`: saved smoke-test outputs for RAG and hybrid rerank.

## Excluded

`knowledge_base/chroma/` is excluded from the GitHub package because it is generated binary index data. Rebuild it with:

```cmd
.\.venv\Scripts\python.exe scripts\build_chroma_index.py ^
  --reset ^
  --batch-size 1 ^
  --max-length 1536 ^
  --device cuda:0 ^
  --use-fp16
```

