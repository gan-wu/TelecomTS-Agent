"""
PromptLoader — 统一提示词管理工具
===================================
设计思路：
    - 所有 Prompt 从 src/prompts/*.yaml 中加载，与 Python 代码完全解耦
    - 支持 {variable} 占位符的 .format() 渲染
    - 单例缓存（避免重复磁盘 IO）
    - 提供直接可用的 messages 列表构造方法（符合 OpenAI API 格式）

用法示例：
    loader = PromptLoader()
    messages = loader.build_messages(
        agent="analyst",
        user_vars={"context_report": "..."}
    )
"""

import os
import yaml
from functools import lru_cache
from typing import Any


# prompts/ 文件夹与本文件同目录（src/tools/ → src/prompts/）
_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")


class PromptLoader:
    """
    统一的 YAML Prompt 读取与渲染工具。

    职责：
        1. 读取 src/prompts/<agent>.yaml 并缓存
        2. 渲染 user_template 中的占位符
        3. 组装为 OpenAI 风格的 messages 列表
        4. 暴露 regex_patterns（供 CriticAgent 直接使用）
    """

    _cache: dict[str, dict] = {}

    @classmethod
    def _load(cls, agent: str) -> dict:
        """带缓存的 YAML 文件读取。首次读取后常驻内存，无重复 IO。"""
        if agent not in cls._cache:
            path = os.path.join(_PROMPTS_DIR, f"{agent}.yaml")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"[PromptLoader] Prompt file not found: {path}\n"
                    f"Expected at: src/prompts/{agent}.yaml"
                )
            with open(path, "r", encoding="utf-8") as f:
                cls._cache[agent] = yaml.safe_load(f)
        return cls._cache[agent]

    @classmethod
    def get_system(cls, agent: str) -> str:
        """获取指定 Agent 的 system prompt 字符串。"""
        return cls._load(agent)["system"].strip()

    @classmethod
    def render_user(cls, agent: str, **kwargs: Any) -> str:
        """
        渲染 user_template，将 {variable} 占位符替换为真实值。

        Args:
            agent:   prompt 文件名（不含 .yaml），如 "analyst"
            **kwargs: 占位符对应的键值对

        Returns:
            渲染完成后的 user 消息字符串
        """
        template: str = cls._load(agent)["user_template"]
        return template.format(**kwargs).strip()

    @classmethod
    def build_messages(cls, agent: str, **kwargs: Any) -> list[dict]:
        """
        一步到位：构造符合 OpenAI API 格式的 messages 列表。

        Args:
            agent:   prompt 文件名（不含 .yaml），如 "solver"
            **kwargs: user_template 占位符对应的键值对

        Returns:
            [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
        """
        return [
            {"role": "system",  "content": cls.get_system(agent)},
            {"role": "user",    "content": cls.render_user(agent, **kwargs)},
        ]

    @classmethod
    def get_regex_patterns(cls, agent: str) -> dict[str, str]:
        """
        获取 YAML 中定义的 regex_patterns 字典。
        专供 CriticAgent 使用，避免正则硬散在代码里。

        Returns:
            {"stat": "^the...", "periodicity": "...", "trend": "..."}
        """
        data = cls._load(agent)
        return data.get("regex_patterns", {})
