# Table 6: 模型在问答推理任务上的对比性能表现 (Performance of the models on the QA task)

*(注：**粗体**表示该项指标排名第一，<u>下划线</u>表示排名第二。MAE 越小越好，Acc 越大越好)*

| 模型 | Statistics ($MAE_{min} \downarrow$) | Statistics ($MAE_{max} \downarrow$) | Periodicity ($MAE_{min} \downarrow$) | Periodicity ($MAE_{max} \downarrow$) | Trends (Acc % $\uparrow$) | Traffic (Acc % $\uparrow$) | Mobility (Acc % $\uparrow$) | Location (Acc % $\uparrow$) | Congestion (Acc % $\uparrow$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| GPT-4.1 | 0.163 | 1588.1 | 57.61 | 93.01 | <u>16.25</u> | <u>44.8</u> | 53.3 | 29.4 | **49.4** |
| Claude 3.7-Sonnet | 0.093 | 1315.8 | <u>32.04</u> | 64.04 | 10.92 | 41.4 | <u>95.0</u> | <u>42.8</u> | 46.1 |
| o4-mini | 0.027 | <u>247.1</u> | 37.21 | 63.15 | 13.37 | 43.3 | 76.7 | 36.7 | **49.4** |
| DeepSeek-R1 | <u>0.020</u> | 1542.6 | 50.33 | <u>61.73</u> | 13.39 | 35.7 | **98.3** | 33.9 | <u>48.3</u> |
| **TeleQnA-Agent (Ours, DeepSeek-V3.2)** | **0.000** | **136.87** | **30.28** | **30.28** | **96.09** | **96.43** | 77.55 | **97.53** | 47.57 |

> **说明：**
>
> 1. 旧版 TeleQnA-Agent 相比通用大模型基线，在 Statistics、Periodicity、Trends、Traffic 和 Location 等指标上表现更稳定。
> 2. 当前 GitHub 展示包中的 Tool Calling 版本进一步将结构化时序统计问题交由确定性工具执行，用于降低数值计算类问题的大模型生成误差。
> 3. 该表保留为历史对照。最新实验对比见 `results/added_tech_effect_report.md`。
