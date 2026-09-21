"""TypeSafe / Jev（System One 决策模型）。

接口来自官方文档 https://docs.typesafe.ai/introduction/quickstart ：
    POST https://api.typesafe.ai/v1/systemone
    Authorization: Bearer <KEY>
    {"state": ..., "model": "jev-latest", "questions": {name: {type, instructions, criteria}}}

响应：
    {"model": "jev-1.13.0",
     "answers": {"intent": {"type":"choice","choice":"...","confidence":0.78,
                            "probabilities": {...}},
                 "urgency": {"type":"noul","noul":0.9,"confidence":...},
                 "complexity": {"type":"score","score":2,"confidence":...,"legend":{...}}}}

注意：一次调用里所有问题**并行评估、互不影响**，加问题几乎不增加耗时。
定价按输入 token 计、输出免费。
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from ..http import OPENER
from .base import Prediction, Provider, ProviderResult, Question

ENDPOINT = "https://api.typesafe.ai/v1/systemone"


class JevProvider(Provider):
    name = "jev"

    def __init__(self, api_key: str | None = None, model: str = "jev-latest",
                 price_per_mtok: float = 0.042):
        self.api_key = api_key or os.getenv("TYPESAFE_API_KEY", "")
        self.model = model
        self.price_per_mtok = price_per_mtok

    def run(self, state: str, questions: list[Question]) -> ProviderResult:
        if not self.api_key:
            return ProviderResult(self.name, error="TYPESAFE_API_KEY 未配置")

        payload = {
            "state": state,
            "model": self.model,
            "questions": {q.name: q.to_api() for q in questions},
        }
        req = urllib.request.Request(
            ENDPOINT,
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
        )

        t0 = time.time()
        try:
            with OPENER.open(req, timeout=60) as r:
                body = json.loads(r.read())
        except urllib.error.HTTPError as e:
            return ProviderResult(self.name, error=f"HTTP {e.code}: {e.read()[:200]!r}")
        except Exception as e:  # noqa: BLE001
            return ProviderResult(self.name, error=repr(e)[:200])
        latency = (time.time() - t0) * 1000

        answers = {}
        for name, a in (body.get("answers") or {}).items():
            answers[name] = _to_prediction(a)

        # 计费只算输入 token，输出免费。用量字段按文档取，拿不到就按字符粗估。
        usage = body.get("usage") or {}
        in_tok = usage.get("input_tokens") or usage.get("prompt_tokens") or len(json.dumps(payload)) // 4
        cost = in_tok / 1_000_000 * self.price_per_mtok
        return ProviderResult(self.name, answers, latency, cost)


def _to_prediction(a: dict) -> Prediction:
    t = a.get("type")
    if t == "choice":
        return Prediction(label=str(a.get("choice")), confidence=float(a.get("confidence") or 0.0),
                          probs=a.get("probabilities"), raw=json.dumps(a, ensure_ascii=False))
    if t == "noul":
        v = float(a.get("noul") or 0.0)
        return Prediction(label="1" if v >= 0.5 else "0", confidence=float(a.get("confidence") or 0.0),
                          probs={"1": v, "0": 1 - v}, raw=json.dumps(a, ensure_ascii=False))
    if t == "score":
        return Prediction(label=str(a.get("score")), confidence=float(a.get("confidence") or 0.0),
                          probs=a.get("probabilities"), raw=json.dumps(a, ensure_ascii=False))
    return Prediction(label="", confidence=0.0, raw=json.dumps(a, ensure_ascii=False))
