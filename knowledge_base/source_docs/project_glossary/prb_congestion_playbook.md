# PRB And Congestion Playbook

This playbook is a project-owned source for interpreting PRB utilization, buffers, throughput, and congestion in 5G RAN troubleshooting. It is intended to support RAG answers and does not include sample-specific labels.

## PRB Utilization

Physical Resource Blocks are the radio resources scheduled for uplink or downlink transmission. High PRB utilization means a large share of radio resources is being consumed. Sustained high utilization is a common signal of capacity pressure, especially when throughput does not increase proportionally.

## Congestion Evidence

Congestion is more likely when several signals agree:

- High PRB_Utilization_DL or PRB_Utilization_UL.
- High UL_NPRB, PRBs_DL_Current, or PRBs_UL_Current.
- Growing Estimated_UL_Buffer.
- Throughput or packet delivery stalls despite continued demand.
- Increased latency, retransmissions, or packet accumulation.
- Congestion label, anomaly flag, or incident ticket mentions resource pressure.

## Non-Congestion Alternatives

High BLER with weak RSRP or low SNR can look like poor performance, but the root cause may be radio coverage or interference rather than congestion. Low PRB utilization with poor throughput often suggests link quality, scheduling policy, or application inactivity rather than capacity saturation.

## Practical Diagnosis Steps

1. Check whether PRB utilization is high enough to imply resource pressure.
2. Compare PRB utilization with throughput and packet counters.
3. Check BLER, RSRP, SNR, and MCS to distinguish radio impairment from congestion.
4. Check buffers to see whether traffic is queued.
5. Use labels and incident tickets as case-specific evidence, not as general knowledge.

## RAG Answer Pattern

For a PRB or congestion question, explain that PRB utilization is a resource-pressure indicator, then mention that congestion should be confirmed with buffers, throughput, BLER, and case labels. Avoid treating every degraded session as congested.
