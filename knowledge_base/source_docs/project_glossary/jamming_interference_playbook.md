# Jamming And Interference Playbook

This playbook is a project-owned source for distinguishing jamming or interference from normal traffic bursts in 5G observability data. It provides general troubleshooting knowledge, not benchmark-specific answers.

## Jamming And Interference

Jamming and interference degrade radio quality by increasing noise or disrupting useful signal reception. They often cause higher BLER, lower SNR, unstable MCS, throughput drops, retransmissions, and sometimes broad PRB impact.

## Normal Traffic Burst

A normal traffic burst is a demand-driven increase in packets or bytes. It can temporarily raise TX_Bytes, RX_Bytes, packet counts, or PRB usage without necessarily degrading radio quality. A burst is not an anomaly by itself if BLER, SNR, RSRP, and delivery remain healthy.

## Evidence That Supports Jamming Or Interference

- UL_SNR or downlink signal quality drops.
- UL_BLER or DL_BLER increases.
- MCS becomes unstable or drops in response to poor channel quality.
- Throughput falls despite continued packet demand.
- Affected KPIs include BLER, SNR, PRB, throughput, or buffer-related fields.
- Incident ticket or anomaly metadata explicitly mentions jamming, interference, or abnormal radio behavior.

## Evidence That Supports Normal Traffic

- Packet or byte counters increase while BLER and SNR stay stable.
- PRB utilization rises in line with throughput.
- No anomaly flag or incident ticket is present.
- Radio quality remains consistent and the application explains the traffic pattern.

## Diagnosis Guidance

Do not infer jamming from traffic volume alone. Treat jamming or interference as likely only when demand signals combine with radio-quality degradation or explicit anomaly evidence. When evidence is mixed, state the uncertainty and list the KPIs that should be checked next.
