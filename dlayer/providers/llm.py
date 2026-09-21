"""生成式 LLM 对照（OpenAI 兼容协议，DashScope / pjlab / OpenAI 都能接）。

这是本实验的对照组，也是**大多数人现在的做法**：让生成模型输出一个 JSON，
里面带上它自己报的置信度。

要验证的正是这件事：那个自报的 confidence 能不能拿来设阈值。
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

from ..prices import price_of
from ..http import OPENER
from .base import Prediction, Provider, ProviderResult, Question

SYS = (
    "你是一个分类器。只输出一个 JSON 对象，不要任何解释、不要 markdown 代码块。\n"
    '格式：{"choice": "<选项名>", "confidence": <0 到 1 之间的小数>}\n'
    "confidence 表示你对这个判断的把握程度。"
)


class LLMProvider(Provider):
    name = "llm"

    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 model: str | None = None, thinking: str | None = None,
                 price_in: float | None = None, price_out: float | None = None,
                 name: str | None = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "")).rstrip("/")
        self.model = model or os.getenv("OPENAI_MODEL", "")
        # Thinking 模式默认是开的（DeepSeek），关掉能省大量输出 token 与时间。
        # 开着的时候 temperature 会被静默忽略 —— 官方文档明写的。
        self.thinking = thinking if thinking is not None else os.getenv("LLM_THINKING") or None
        pi, po = price_of(self.model)
        self.price_in = pi if price_in is None else price_in
        self.price_out = po if price_out is None else price_out
        self.name = name or f"llm:{self.model}"

    def run(self, state: str, questions: list[Question]) -> ProviderResult:
        if not (self.api_key and self.base_url and self.model):
            return ProviderResult(self.name, error="OPENAI_API_KEY / BASE_URL / MODEL 未配齐")

        answers, latency, cost = {}, 0.0, 0.0
        for q in questions:                      # 生成模型一次只问一个，逐题调用
            # 网络抖动会让单次调用返回空/超时（实测遇到过 10s + 空返回），
            # 重试一次即可，且不会污染统计口径（重试的耗时也算进去）。
            for attempt in range(3):
                pr, ms, c = self._ask_one(state, q)
                latency += ms
                cost += c
                if pr.label:
                    break
                if attempt < 2:
                    time.sleep(0.8)
            answers[q.name] = pr
        return ProviderResult(self.name, answers, latency, cost)

    def _ask_one(self, state: str, q: Question) -> tuple[Prediction, float, float]:
        opts = q.options
        if isinstance(q.criteria, dict):
            body = "\n".join(f"- {k}：{v}" for k, v in q.criteria.items())
        elif isinstance(q.criteria, list):
            body = "\n".join(f"- {i}：{v}" for i, v in enumerate(q.criteria))
        else:
            body = "- 0：否\n- 1：是"

        prompt = (f"待判断的内容：\n{state}\n\n"
                  f"问题：{q.instructions}\n可选：\n{body}\n\n"
                  f'请只输出：{{"choice": "<上面某个选项>", "confidence": <0~1>}}')

        payload = {"model": self.model, "temperature": 0,
                   "messages": [{"role": "system", "content": SYS},
                                {"role": "user", "content": prompt}]}
        if self.thinking in ("enabled", "disabled"):
            payload["thinking"] = {"type": self.thinking}
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
        )
        t0 = time.time()
        try:
            with OPENER.open(req, timeout=45) as r:
                body = json.loads(r.read())
        except urllib.error.HTTPError as e:
            return Prediction("", 0.0, raw=f"HTTP {e.code}: {e.read()[:150]!r}"), (time.time()-t0)*1000, 0.0
        except Exception as e:  # noqa: BLE001
            return Prediction("", 0.0, raw=repr(e)[:150]), (time.time()-t0)*1000, 0.0
        ms = (time.time() - t0) * 1000

        text = (body.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
        m = re.search(r"\{.*\}", text, re.S)
        choice, conf = "", 0.0
        if m:
            try:
                d = json.loads(m.group(0))
                choice = str(d.get("choice", "")).strip()
                conf = float(d.get("confidence", 0.0))
            except Exception:  # noqa: BLE001
                pass
        if choice not in opts:               # 生成模型常见的失手：编了个不存在的选项
            choice = _closest(choice, opts)

        u = body.get("usage") or {}
        cost = (u.get("prompt_tokens", 0) / 1e6 * self.price_in
                + u.get("completion_tokens", 0) / 1e6 * self.price_out)
        return Prediction(choice, max(0.0, min(1.0, conf)), raw=text[:300]), ms, cost


def _closest(s: str, opts: list[str]) -> str:
    """选项名没完全对上时，宽松匹配一次；再不行就返回空串（记为错）。"""
    s = s.strip()
    for o in opts:
        if o and (o in s or s in o):
            return o
    return s if s in opts else ""
