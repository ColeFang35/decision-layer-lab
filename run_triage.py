#!/usr/bin/env python3
"""任务 A：客服工单分诊。

复刻 TypeSafe 官方 patterns/intent-routing 的做法：
**一次调用并行问三个原子问题**（意图 / 紧急度 / 复杂度），
再按置信度决定这个工单走确定性代码、专家 LLM，还是转人工。

要验证的是：那个 confidence 到底能不能拿来分流。
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from dlayer.env import load_env  # noqa: E402

load_env()
from dlayer.providers import (JevProvider, LLMProvider, MockProvider,  # noqa: E402
                              RuleProvider, TRIAGE_KEYWORDS)
from dlayer.providers.base import Question  # noqa: E402
from dlayer.runner import build_report, run_provider  # noqa: E402

ROOT = pathlib.Path(__file__).parent
INTENTS = list(TRIAGE_KEYWORDS)


LEVELS = {"easy": "messages.jsonl", "hard": "messages_hard.jsonl"}


def load_samples(level: str) -> list[dict]:
    path = ROOT / "data" / "triage" / LEVELS[level]
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        r["state"] = r["text"]
        # 难档样本只标了意图（紧急度/复杂度本来就没有唯一答案）
        r["gold"] = {"intent": r["intent"],
                     "urgency": "1" if r.get("urgent") else "0",
                     "complexity": str(r.get("complexity", 1))}
    return rows


def questions() -> list[Question]:
    return [
        Question("intent", "choice", "这条客户消息属于下面哪一类问题",
                 {k: f"与「{k}」相关的问题" for k in INTENTS}),
        Question("urgency", "noul", "这条消息表达了紧迫性或时间敏感"),
        Question("complexity", "score", "这条消息的处理复杂度",
                 ["一句话就能答，查一下就行",
                  "需要查订单/账户等具体信息才能答",
                  "涉及多方、规则冲突或需要人工判断"]),
    ]


def build_providers(samples: list[dict]) -> list:
    out = []
    if os.getenv("TYPESAFE_API_KEY"):
        out.append(JevProvider())
    if os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_BASE_URL"):
        # 支持一次跑多个模型做对照：OPENAI_MODELS=deepseek-flash,deepseek-v4-pro
        models = [m.strip() for m in os.getenv("OPENAI_MODELS", "").split(",") if m.strip()]
        if not models and os.getenv("OPENAI_MODEL"):
            models = [os.getenv("OPENAI_MODEL")]
        for m in models:
            out.append(LLMProvider(model=m))
    out.append(RuleProvider("keyword", keyword_table=TRIAGE_KEYWORDS))
    if not any(isinstance(x, (JevProvider, LLMProvider)) for x in out):
        truth = {(s["state"], "intent"): s["gold"]["intent"] for s in samples}
        out.insert(0, MockProvider("calibrated", name="mock:决策模型(假数据)", truth=truth))
        out.insert(1, MockProvider("overconfident", name="mock:生成模型(假数据)", truth=truth))
    return out


def main() -> None:
    level = sys.argv[1] if len(sys.argv) > 1 else "easy"
    samples = load_samples(level)
    qs = questions()
    provs = build_providers(samples)
    mock = not any(isinstance(x, (JevProvider, LLMProvider)) for x in provs)
    print("  实际 provider:", ", ".join(p.name for p in provs), "\n")

    print(f"任务 A · 客服工单分诊［{level}］：{len(samples)} 条样本，{len(INTENTS)} 个意图\n")
    results = {}
    for p in provs:
        if isinstance(p, (JevProvider, LLMProvider)) and not _ready(p):
            print(f"  跳过 {p.name}（未配置 key）")
            continue
        r = run_provider(p, samples, qs, main="intent")
        results[p.name] = r
        c = r["calib"]
        print(f"  {p.name:<22} 准确率 {c.accuracy:6.1%}  平均置信度 {c.mean_confidence:.3f}  "
              f"ECE {c.ece:.3f}  AURC {c.aurc:.3f}")

    out = ROOT / "results" / f"triage-{level}"
    out.mkdir(parents=True, exist_ok=True)
    import dataclasses
    (out / "raw.json").write_text(json.dumps(          # ← 先存原始结果，报告出问题也不丢数据
        {n: {**r, "calib": dataclasses.asdict(r["calib"])} for n, r in results.items()},
        ensure_ascii=False, indent=1), encoding="utf-8")
    if level == "easy":
        level_note = "**易档**：消息里直接出现类名关键词，模型基本靠关键词就能分对。\n\n"
    else:
        level_note = ("**难档**：消息刻意改成间接表述，**类名关键词一个都不出现**"
                      "（已用脚本校验过），必须靠语义推断。\n\n")
    desc = (f"{len(samples)} 条人工构造的中文客服消息，10 个意图类别。\n\n"
            + level_note
            + "对应官方 `patterns/intent-routing`：一次调用并行问三个原子问题，"
              "再按 intent 的置信度分流 —— 低于阈值转人工、简单意图走确定性代码、"
              "复杂意图交给带领域上下文的专家 LLM。")

    md = build_report(f"任务 A · 客服工单分诊［{level}］", desc,
                      samples, results, qs[0], out, mock)
    print(f"\n报告：{md.relative_to(ROOT)}")
    if mock:
        print("⚠️ 当前是 mock 数据。配好 TYPESAFE_API_KEY / OPENAI_* 后重跑才是真结果。")


def _ready(p) -> bool:
    if isinstance(p, JevProvider):
        return bool(p.api_key)
    if isinstance(p, LLMProvider):
        return bool(p.api_key and p.base_url and p.model)
    return True


if __name__ == "__main__":
    main()
