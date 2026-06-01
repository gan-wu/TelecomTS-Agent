# Heavy Load Versus Performance Playbook

This playbook is a project-owned source for telecom AIOps questions that ask whether a session is under heavy load or performing well. It gives general diagnosis criteria and does not contain benchmark-specific answers.

## Decision Concepts

### Heavy Load

Heavy load means radio or transport resources are close to saturation. It is usually supported by high PRB utilization, growing uplink buffers, high packet or byte demand, and congestion labels or incident tickets. Heavy load is a capacity-pressure state, not just a low-quality radio state.

### Performing Well

Performing well means the session has enough radio resources and acceptable link quality for the current traffic. Typical evidence includes low BLER, stable or adequate RSRP/SNR, reasonable PRB utilization, and no congestion or anomaly indication.

### Degraded But Not Heavy

A session can be degraded without being heavily loaded. Poor coverage, low RSRP, low SNR, high BLER, jamming, or uplink impairment can reduce quality even when PRB utilization is low. In this case, the correct diagnosis should avoid calling it heavy load unless resource pressure is also visible.

## Evidence Checklist

- PRB utilization: high values support capacity pressure; low values argue against heavy load.
- Buffer growth: a high or increasing Estimated_UL_Buffer supports uplink resource pressure.
- BLER and retransmission clues: high UL_BLER or DL_BLER supports poor reliability, but it may be caused by radio impairment rather than load.
- Signal quality: low RSRP or low UL_SNR points to coverage or interference problems.
- Traffic counters: high TX_Bytes, RX_Bytes, and packet counts can indicate demand, but demand alone is not congestion.
- Labels and tickets: congestion labels, anomaly flags, affected KPIs, and incident tickets should be used as high-confidence case evidence.

## Answer Pattern

When asked "heavy load or performing well", first separate resource pressure from radio degradation. If PRB utilization and buffers are low but RSRP or BLER is poor, answer that the session is degraded or not performing well, but not necessarily under heavy load. If congestion evidence is explicit, answer that the session is under heavy load. If load and radio quality are both healthy, answer that it is performing well.
