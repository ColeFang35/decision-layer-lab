"""统一的 provider 接口。

三种做法要能公平对比，所以先定义清楚它们各自"能给出什么"：

- 规则：给标签，但**给不出有意义的置信度**（要么命中要么兜底）
- 生成式 LLM：给标签，也能让它自报一个置信度 —— 但那个数字可不可信正是本实验要查的
- System One 决策模型：给标签 + 校准过的概率分布

对比的关键不是谁准，而是**那个 confidence 能不能拿来设阈值**。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Prediction:
    label: str
    confidence: float                       # 0~1
    probs: dict[str, float] | None = None   # 完整分布，拿不到就是 None
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    raw: str | None = None                  # 原始响应，排查用

    @property
    def has_probs(self) -> bool:
        return bool(self.probs)


@dataclass
class Question:
    """一个原子问题。type 决定 criteria 的形态。

    choice -> criteria 是 {选项: 说明}
    noul   -> criteria 为空，retur 布尔（用 0/1 表示）
    score  -> criteria 是档位描述列表，return 档位下标
    """
    name: str
    type: str
    instructions: str
    criteria: dict[str, str] | list[str] | None = None

    def to_api(self) -> dict:
        q = {"type": self.type, "instructions": self.instructions}
        if self.criteria:
            q["criteria"] = self.criteria
        return q

    @property
    def options(self) -> list[str]:
        if self.type == "choice" and isinstance(self.criteria, dict):
            return list(self.criteria)
        if self.type == "score" and isinstance(self.criteria, list):
            return [str(i) for i in range(len(self.criteria))]
        return ["0", "1"]


@dataclass
class ProviderResult:
    provider: str
    answers: dict[str, Prediction] = field(default_factory=dict)
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    error: str | None = None


class Provider:
    name = "base"

    def run(self, state: str, questions: list[Question]) -> ProviderResult:
        raise NotImplementedError
