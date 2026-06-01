# Manifest

This repository is a cleaned GitHub display package derived from `TelecomTS_QA`.

## Included

- `src/`: core implementation for agents, LangGraph, model routing, token budget policy, Tool Calling, RAG, and batch evaluation.
- `scripts/`: smoke tests, benchmark construction, Chroma index build, routing checks, and experiment analysis.
- `data/benchmark.json`: 1000-sample benchmark used by the original workflow.
- `data/benchmark_main_agent_3000.json`: integrated 3000-sample benchmark for Tool/RAG/Agent routing display.
- `data/benchmark_main_agent_3000.summary.json`: sampling quotas and route distribution.
- `data/dataset_info.json`: dataset metadata.
- `knowledge_base/source_docs/`: curated telecom/RAG source documents.
- `knowledge_base/chunks/`: contextual RAG chunks and chunk metadata.
- `knowledge_base/manifests/`: source provenance and extraction metadata.
- `knowledge_base/retrieval/`: saved retrieval smoke outputs.
- `results/`: selected experiment outputs for comparison, including the latest 3000-sample run.
- `docs/`: paper PDF, historical Table 6 reference, QR/upload notes, and resume project summary.
- `requirements.txt`, `.env.example`, `Dockerfile`, `autodl_vllm_serve.sh`.

## Excluded

- `.venv/`
- `__pycache__/`
- raw Arrow dataset shards: `data/*.arrow`
- generated Chroma index: `knowledge_base/chroma/`
- temporary clone workspace: `external_sources/raw_repos/`
- local model weights: `*.gguf`, `*.safetensors`, `*.bin`
- real secrets: `.env`, API keys, private tokens.

## Rationale

The package is intended for GitHub and interview display. It keeps the code, benchmark JSON, curated knowledge sources, chunks, selected results, and reproducible commands while avoiding oversized generated artifacts and secrets.
