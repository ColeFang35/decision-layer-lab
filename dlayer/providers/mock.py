"""离线 mock —— 没有 API key 也能把全链路跑通。

**它产出的是假数据，不是实验结果。** run_*.py 会在报告里显式标注 mock，
避免任何一次误当成真数字引用。

存在的意义和客服项目里的 ScriptedMockModel 一样：
真实模型有随机性、要花钱、要网络，mock 是确定性的，能用来验证
「指标算得对不对、图画的通不通、报告生成有没有问题」这些代码层面的东西。
同一套代码路径，接上真 key 就跑真数据。
"""
from __future__ import annotations

import hashlib
import time

from .base import Prediction, Provider, ProviderResult, Question


def _h(*parts: str) -> float:
    """确定性伪随机：同样的输入永远得到同样的数。"""
    d = hashlib.sha256("||".join(parts).encode()).digest()
    return int.from_bytes(d[:4], "big") / 0xFFFFFFFF


class MockProvider(Provider):
    """profile 用来模拟两种典型行为，方便预览图表：

    calibrated  —— 置信度和正确率大致对得上（决策模型该有的样子）
    overconfident —— 置信度普遍很高，但错得不少（生成模型自我评估的常见毛病）
    """

    def __init__(self, profile: str = "calibrated", name: str | None = None,
                 truth: dict[tuple[str, str], str] | None = None, hit_rate: float = 0.78):
        self.profile = profile
        self.name = name or f"mock:{profile}"
        # 固定装置要"知道答案"才能模拟出合理的准确率，否则只能瞎猜（实测 4%）
        self.truth = truth or {}
        self.hit_rate = hit_rate

    def run(self, state: str, questions: list[Question]) -> ProviderResult:
        t0 = time.time()
        answers = {}
        for q in questions:
            if q.type != "choice":
                answers[q.name] = Prediction("0", 0.5)
                continue
            opts = q.options
            gold = self.truth.get((state, q.name))

            # 关键：先抽置信度，再**以该置信度的概率**决定答对与否。
            # 这样"calibrated"这个 profile 是**按构造就校准**的（ECE≈0），
            # 而不是靠调参凑出来的。
            r_conf, r_coin = _h(state, q.name, "c"), _h(state, q.name, "coin")
            if self.profile == "calibrated":
                conf = 0.55 + r_conf * 0.43          # 0.55 ~ 0.98
                correct = r_coin < conf              # 答对概率 = 自报置信度
            else:                                    # overconfident
                conf = 0.85 + r_conf * 0.14          # 0.85 ~ 0.99，永远很自信
                correct = r_coin < self.hit_rate     # 真实能力固定，和置信度无关

            if correct and gold in opts:
                label = gold
            else:                                    # 答错：挑一个别的选项
                others = [o for o in opts if o != gold] or opts
                label = others[int(_h(state, q.name, "w") * len(others)) % len(others)]
            answers[q.name] = Prediction(label, round(min(conf, 0.99), 3))
        time.sleep(0)                      # 保持调用形态一致
        return ProviderResult(self.name, answers, (time.time() - t0) * 1000, 0.0)
