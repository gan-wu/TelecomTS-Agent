class ContextBuilder:
    """
    通信专域规则提示词转换器 (Domain Context Converter)。
    解决痛点：大模型长文本注意力衰减 (Attention Decay) 以及对 JSON 格式特征的数值误判。
    
    将 KPI statistics 转换为面向 LLM 的结构化上下文，并隐藏接近答案的 benchmark labels。
    """
    
    @staticmethod
    def build_system_prompt() -> str:
        """
        构建最底层的系统角色设定 (System Instruction)。极大地约束产生幻觉。
        """
        return """You are a senior 5G Telecommunications AIOps System Analyst.
Your core logic combines traditional telecom rule-based thresholds with advanced cognitive understanding.

Strict Execution Rules:
1. When asked about a metric's explicit value (e.g., Variance, Mean, Periodicity), read the exact value from the table.
2. If the user asks for the output format to be a specific string/number "Otherwise respond with 128", strictly follow it.
3. Be concise and precise. No conversational greetings or fluff in your final answer.
"""

    @staticmethod
    def json_to_markdown_report(context_dict: dict) -> str:
        """
        提取 JSON 中的离散张量特征，将其结构化为对大模型极度友好的 Markdown 报表。

        不输出 benchmark labels。labels 对应用、位置、移动性、拥塞、
        异常类问题接近目标变量，只应通过显式 Tool Calling 路径读取。
        """
        stats = context_dict.get("statistics", {})

        report = "### [1. Current Network Session]\n"
        report += "- Benchmark labels are intentionally hidden from the LLM prompt.\n"
        report += "- The report below contains only KPI statistics and derived rule warnings.\n\n"

        # Select the KPI fields most relevant to radio quality, BLER, PRB usage, and traffic.
        report += "### [2. Core KPI Statistics for the Window]\n"
        report += "| KPI Name | Mean Value | Variance | Extracted Trend (-1/0/1) | Period Length |\n"
        report += "| :--- | :--- | :--- | :--- | :--- |\n"
        
        # 挑选最影响网络性能的核心信号：信号强度 RSRP，误块率 BLER，调度 PRB 利用率，及收发流量
        critical_kpis = [
            "RSRP", "DL_BLER", "UL_BLER", "UL_SNR", "TX_Bytes", "RX_Bytes", 
            "PRB_Utilization_DL", "PRB_Utilization_UL", "DL_NumberOfPackets", "Estimated_UL_Buffer"
        ]

        for kpi in critical_kpis:
            if kpi in stats:
                val = stats[kpi]
                # Compact floating-point values to reduce prompt noise.
                mean_val = f"{val.get('mean', 'N/A'):.4f}" if isinstance(val.get('mean'), float) else val.get('mean', 'N/A')
                var_val = f"{val.get('variance', 'N/A'):.4f}" if isinstance(val.get('variance'), float) else val.get('variance', 'N/A')
                report += f"| {kpi} | {mean_val} | {var_val} | {val.get('trend', 'N/A')} | {val.get('periodicity', 'N/A')} |\n"
        
        # Add lightweight telecom rule hints derived from KPI statistics.
        report += "\n### [3. Domain Knowledge & Rule Warnings]\n"
        
        rsrp_stat = stats.get("RSRP", {})
        if rsrp_stat.get("mean", 0) < -105:
            report += "- [RULE] Note: The RSRP mean is below -105dBm. This definitively signifies **Poor Radio Coverage / Edge Cell condition**.\n"
            
        prb_dl = stats.get("PRB_Utilization_DL", {})
        if prb_dl.get("mean", 0) > 80:
            report += "- [RULE] Note: PRB Utilization DL is exceedingly high (>80%). The network is operating under **High Load / Congestion Risk**.\n"

        report += "\nNow, based solely on the data context above, please answer the user's specific query below.\n"
        
        return report

