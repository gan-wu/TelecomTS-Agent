# Knowledge Base Extraction Summary

Generated at: 2026-05-31 16:39:34 +08:00

## Scope

- telecom_domain: 5G RAN, O-RAN, KPI monitoring, A1 policy, gNB/CU/DU config, observability dashboards.
- rag_methods: BGE embedding, BGE-M3, reranking, retrieval/indexing, late chunking.
- project_glossary: project-owned KPI, anomaly, O-RAN, and troubleshooting glossary.

## Repositories
- srsRAN_Project [telecom_domain]: 48 files, commit 4bf1543936d062686d64c10724d2f27a9854f065
- openairinterface5g [telecom_domain]: 121 files, commit cb0e501293a7a4664f09322136d7ff29a39343dc
- ric-app-kpimon-go [telecom_domain]: 7 files, commit 8bbbbbb90093db01f88820de755bce0ee2189c88
- ric-plt-a1 [telecom_domain]: 11 files, commit 09a757b4fd63198d8690d50b52bfd04552d47f1f
- oran-sc-ric [telecom_domain]: 2 files, commit 621ade26251f69a4ad079ba98bb708d8e5aeeb98
- FlagEmbedding [rag_methods]: 36 files, commit 7ed43d67ec03fbe5c31c0992dbfa941fb1860549
- late-chunking [rag_methods]: 2 files, commit 1d3bb02bf091becd0771455e4e7959463935e26c
- TelecomTS_glossary [project_glossary]: 1 file, project-maintained local source

## Next Step

Use source_manifest.json as metadata for external sources when chunking. Project-owned glossary files are assigned `TelecomTS_glossary / project_glossary` metadata by the chunking code.
