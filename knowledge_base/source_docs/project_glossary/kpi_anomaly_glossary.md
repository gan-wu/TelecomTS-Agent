# TelecomTS KPI And Anomaly Glossary

This glossary is a project-owned knowledge source for the RAG tool. It normalizes the KPI names, anomaly clues, and troubleshooting language used in TelecomTS questions, the dataset paper, srsRAN material, OpenAirInterface docs, and O-RAN SC docs.

## Radio Quality KPIs

### RSRP

Reference Signal Received Power. RSRP measures received reference-signal strength. Lower RSRP usually indicates weaker coverage, higher path loss, or a UE being farther from the radio unit. Very low RSRP can lead to lower MCS, lower throughput, higher BLER, and unstable connectivity.

### UL_SNR

Uplink Signal-to-Noise Ratio. UL_SNR describes uplink signal quality at the receiver. Low UL_SNR often points to weak uplink signal, interference, or poor radio conditions. Low UL_SNR and high UL_BLER together are strong evidence of uplink radio degradation.

### DL_BLER And UL_BLER

Block Error Rate for downlink and uplink transport blocks. BLER is the fraction of transport blocks that cannot be decoded correctly. High UL_BLER means unreliable uplink transmission and can be caused by poor radio quality, jamming, interference, insufficient uplink power, aggressive MCS, or resource congestion. High DL_BLER means unreliable downlink transmission and can be caused by weak RSRP, low SINR, interference, or scheduling issues.

### DL_MCS And UL_MCS

Modulation and Coding Scheme for downlink and uplink. Higher MCS means more aggressive modulation and coding, which can increase throughput when channel quality is good. If MCS stays high while BLER rises, the scheduler may be too aggressive for the current radio quality. If MCS drops while RSRP/SNR are poor, the network is adapting to a weaker channel.

## Resource And Traffic KPIs

### UL_NPRB, PRBs_DL_Current, And PRBs_UL_Current

Physical Resource Block allocation indicators. These KPIs describe how many PRBs are assigned to a UE for uplink or downlink transmission. High PRB allocation with low throughput can suggest poor spectral efficiency, retransmissions, congestion, or radio impairment.

### PRB_Utilization_DL And PRB_Utilization_UL

Downlink and uplink PRB utilization ratio. High PRB utilization means the cell or UE is consuming a large share of radio resources. High PRB utilization together with rising BLER or falling throughput may indicate resource pressure, congestion, or interference.

### Estimated_UL_Buffer

Estimated uplink buffer size at the UE. A high or growing uplink buffer suggests queued uplink traffic waiting for transmission. If Estimated_UL_Buffer increases while UL throughput is low, possible causes include uplink congestion, poor radio quality, insufficient scheduling grants, or high UL_BLER.

### TX_Bytes, RX_Bytes, UL_NumberOfPackets, And DL_NumberOfPackets

Traffic volume and packet-count KPIs. They help infer application behavior, throughput changes, and traffic bursts. Sudden drops in byte or packet counters can indicate link interruption, application inactivity, severe radio degradation, or congestion.

## Common Anomaly Patterns

### Jamming Or Interference

Jamming and interference often degrade radio quality. Typical clues include higher BLER, lower SNR, unstable throughput, retransmissions, and possible MCS reduction. Wideband noise jamming can affect many PRBs, while narrowband or pulsed jamming may create intermittent degradation.

### High Network Congestion

Congestion usually appears as high PRB utilization, growing buffers, reduced throughput, and sometimes higher latency or packet accumulation. Congestion can coexist with high BLER, but BLER-dominant failures should also be checked for radio impairment.

### Poor Coverage Or Weak Signal

Poor coverage usually appears as low RSRP, low SNR, lower MCS, and unstable throughput. If BLER is high while signal metrics are weak, the root cause is more likely radio coverage or interference than pure application traffic demand.

### Scheduler Or Resource Pressure

Scheduler/resource pressure can appear as high PRB utilization, high buffer, reduced throughput, and mismatch between allocated resources and delivered bytes. In O-RAN or RAN troubleshooting, this may involve gNB MAC scheduling behavior, CU/DU split behavior, or near-RT RIC policy effects.

## Architecture Terms

### gNB, CU, DU, RU, And UE

gNB is the 5G base station. CU handles higher-layer control/user-plane functions, DU handles lower-layer real-time radio functions, RU handles radio transmission and reception, and UE is the user equipment. KPI interpretation often depends on where a metric is reported: UE-side metrics reflect received signal and device state, while gNB-side metrics reflect scheduler, MAC, and radio resource behavior.

### O-RAN Near-RT RIC, E2, A1, And xApp

The near-real-time RIC hosts xApps for RAN monitoring and control. E2 connects the RIC to RAN nodes for telemetry and control. A1 is used for policy guidance. A KPI monitoring xApp can collect and expose radio KPIs for troubleshooting, anomaly detection, and policy optimization.

## Interview-Friendly RAG Answer Pattern

For a knowledge question, first retrieve definitions and evidence with `search_telecom_knowledge(query, top_k=5)`. Use dense retrieval for semantic matches, BM25 for exact KPI fields such as `UL_BLER` and `PRB_Utilization_DL`, RRF for fusion, and `bge-reranker-v2-m3` to select the most relevant evidence. The SolverAgent should then answer from the retrieved evidence and only use current case KPI data when the question asks about this specific session.
