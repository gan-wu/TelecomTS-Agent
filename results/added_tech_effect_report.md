# Added Technology Effects Report

## Table 6 Style Metrics

| Run | Stat min | Stat max | Period min | Period max | Trend | Traffic | Mobility | Location | Congestion | Tool rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| previous | 0.000 | 136.868 | 30.277 | 30.277 | 96.094 | 95.536 | 98.980 | 97.531 | 45.946 | NA |
| current | 0.000 | 0.000 | 0.072 | 0.072 | 100.000 | 98.214 | 100.000 | 100.000 | 48.108 | 99.000 |

## LangGraph Branches

### numeric_tool

- trace: `load_case -> tool_router -> critic`
- final_answer: The mean of UL_BLER is 0.008
- rag_hit_count: 0
- model_calls: `[]`

### knowledge_rag

- trace: `load_case -> tool_router -> rag_tool -> solver -> critic`
- final_answer: High UL_BLER means unreliable uplink transmission and can be caused by poor radio quality, jamming, interference, insufficient uplink power, aggressive MCS, or resource congestion.
- rag_hit_count: 5
- model_calls: `[{"mode": "local", "task": "solver", "provider": "local", "model": "qwen3.5-9b-q4", "latency_ms": 2126.39, "fallback_used": false}]`

### normal_agent

- trace: `load_case -> tool_router -> analyst -> solver -> critic`
- final_answer: The session is healthy, characterized by stable signal quality, robust link reliability, and ample network resource headroom despite asymmetric video streaming activity.
- rag_hit_count: 0
- model_calls: `[{"mode": "local", "task": "analyst", "provider": "local", "model": "qwen3.5-9b-q4", "latency_ms": 3697.62, "fallback_used": false}, {"mode": "local", "task": "solver", "provider": "local", "model": "qwen3.5-9b-q4", "latency_ms": 1130.25, "fallback_used": false}]`

## Retrieval Experiments

### What does high UL_BLER mean for 5G uplink troubleshooting?

**Dense top5**

- #1 TelecomTS_glossary:knowledge_base/source_docs/project_glossary/kpi_anomaly_glossary.md | TelecomTS KPI And Anomaly Glossary > Resource And Traffic KPIs > EstimatedULBuffer | distance=0.3004
- #2 TelecomTS_glossary:knowledge_base/source_docs/project_glossary/kpi_anomaly_glossary.md | TelecomTS KPI And Anomaly Glossary > Radio Quality KPIs > DLBLER And ULBLER | distance=0.3119
- #3 TelecomTS_glossary:knowledge_base/source_docs/project_glossary/kpi_anomaly_glossary.md | TelecomTS KPI And Anomaly Glossary > Radio Quality KPIs > ULSNR | distance=0.3244
- #4 openairinterface5g:doc/UL_MIMO.md | UpLink Multiple Input Multiple Output (UL MIMO) > What is UL MIMO ? | distance=0.3368
- #5 openairinterface5g:doc/UL_MIMO.md | UpLink Multiple Input Multiple Output (UL MIMO) > Step 2: Use the sim > Option 2: launch UL sim | distance=0.3659

**Hybrid top5**

- #1 TelecomTS_glossary:knowledge_base/source_docs/project_glossary/kpi_anomaly_glossary.md | TelecomTS KPI And Anomaly Glossary > Radio Quality KPIs > DLBLER And ULBLER | rerank=0.9833 | dense=2 | bm25=1
- #2 TelecomTS_glossary:knowledge_base/source_docs/project_glossary/kpi_anomaly_glossary.md | TelecomTS KPI And Anomaly Glossary > Resource And Traffic KPIs > EstimatedULBuffer | rerank=0.9514 | dense=1 | bm25=4
- #3 TelecomTS_glossary:knowledge_base/source_docs/project_glossary/kpi_anomaly_glossary.md | TelecomTS KPI And Anomaly Glossary > Radio Quality KPIs > ULSNR | rerank=0.9161 | dense=3 | bm25=3
- #4 TelecomTS_paper:2510.06063.pdf | 2510.06063 > Algorithm 1End-to-End Data Collection Procedure | rerank=0.8790 | dense=24 | bm25=6
- #5 TelecomTS_glossary:knowledge_base/source_docs/project_glossary/kpi_anomaly_glossary.md | TelecomTS KPI And Anomaly Glossary > Resource And Traffic KPIs > PRBUtilizationDL And PRBUtilizationUL | rerank=0.7583 | dense=8 | bm25=5

### UL_BLER PRB_Utilization_DL gNB jamming KPI troubleshooting

**Dense top5**

- #1 TelecomTS_glossary:knowledge_base/source_docs/project_glossary/kpi_anomaly_glossary.md | TelecomTS KPI And Anomaly Glossary > Common Anomaly Patterns > Jamming Or Interference | distance=0.2894
- #2 TelecomTS_glossary:knowledge_base/source_docs/project_glossary/kpi_anomaly_glossary.md | TelecomTS KPI And Anomaly Glossary > Resource And Traffic KPIs > PRBUtilizationDL And PRBUtilizationUL | distance=0.3327
- #3 TelecomTS_glossary:knowledge_base/source_docs/project_glossary/kpi_anomaly_glossary.md | TelecomTS KPI And Anomaly Glossary > Common Anomaly Patterns > High Network Congestion | distance=0.3660
- #4 TelecomTS_glossary:knowledge_base/source_docs/project_glossary/kpi_anomaly_glossary.md | TelecomTS KPI And Anomaly Glossary > Resource And Traffic KPIs > ULNPRB, PRBsDLCurrent, And PRBsULCurrent | distance=0.3720
- #5 TelecomTS_glossary:knowledge_base/source_docs/project_glossary/kpi_anomaly_glossary.md | TelecomTS KPI And Anomaly Glossary > Radio Quality KPIs > DLBLER And ULBLER | distance=0.3750

**Hybrid top5**

- #1 TelecomTS_paper:2510.06063.pdf | 2510.06063 > ULNPRB (Allocated Uplink Physical Resource Blocks) | rerank=0.9474 | dense=7 | bm25=5
- #2 TelecomTS_paper:2510.06063.pdf | 2510.06063 > 2510.06063 | rerank=0.9439 | dense=16 | bm25=24
- #3 TelecomTS_glossary:knowledge_base/source_docs/project_glossary/kpi_anomaly_glossary.md | TelecomTS KPI And Anomaly Glossary > Radio Quality KPIs > DLBLER And ULBLER | rerank=0.8592 | dense=5 | bm25=3
- #4 TelecomTS_paper:2510.06063.pdf | 2510.06063 > Algorithm 1End-to-End Data Collection Procedure | rerank=0.8526 | dense=None | bm25=12
- #5 TelecomTS_paper:2510.06063.pdf | 2510.06063 > E TelecomTS: An Example | rerank=0.7893 | dense=None | bm25=9
