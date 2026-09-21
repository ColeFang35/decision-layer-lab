#!/usr/bin/env python3
"""任务 B：知识路由。

把一句用户问法路由到 16 条知识里的正确一条。这是**高基数分类**（十几到几百个选项），
正是决策模型被设计出来处理的场景 —— 它的 Choice 原语支持到 255 个选项。

三档对照：
  规则/BM25   —— 就是原来项目里的关键词方案
  生成式 LLM  —— 让它输出 JSON
  决策模型    —— Choice 原语 + 校准概率

语料复用姊妹项目 rag-lab 的真实知识库与 query（同一套 gold），
所以这里的数字能和那边的检索实验直接对照。
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from dlayer.env import load_env  # noqa: E402

load_env()
from dlayer.providers import JevProvider, LLMProvider, MockProvider, RuleProvider  # noqa: E402
from dlayer.providers.base import Question  # noqa: E402
from dlayer.runner import build_report, run_provider  # noqa: E402

ROOT = pathlib.Path(__file__).parent
RAGLAB_DOCS = pathlib.Path("/Users/a1-6/Desktop/ffcc/rag-lab/data/docs")


def load_sections() -> dict[str, str]:
    """把 markdown 按二级标题拆成知识条目。和 rag-lab 的切法保持一致。"""
    sections: dict[str, str] = {}
    for md in sorted(RAGLAB_DOCS.glob("*.md")):
        for block in re.split(r"\n(?=## )", md.read_text(encoding="utf-8")):
            m = re.match(r"##\s+(.+)", block.strip())
            if m:
                sections[m.group(1).strip()] = block.strip()
    return sections


def load_samples(sections: dict[str, str]) -> list[dict]:
    qs = json.loads((ROOT / "data/routing/queries.json").read_text(encoding="utf-8"))["queries"]
    out = []
    for i, q in enumerate(qs, 1):
        if q["gold"] not in sections:
            continue
        out.append({"id": f"R{i:03d}", "state": q["query"], "gold": {"section": q["gold"]}})
    return out


def brief(text: str, n: int = 42) -> str:
    """给选项一句简短说明，太长会稀释模型的注意力。"""
    body = re.sub(r"^##.*\n|^关键词：.*\n|^\s*$", "", text, flags=re.M).strip()
    body = re.sub(r"\s+", " ", body)
    return (body[:n] + "…") if len(body) > n else body


def questions(sections: dict[str, str], bare: bool = False) -> list[Question]:
    """bare=True 时只给标题、不给描述。

    带描述那一档其实等于泄题 —— 描述是从知识正文里摘的，模型看到描述基本就知道选哪个。
    只给标题才是真实场景：用户问法千变万化，标题就那几个字。
    """
    criteria = ({k: "" for k in sections} if bare
                else {k: brief(v) for k, v in sections.items()})
    return [Question("section", "choice", "这句话应该在下面哪一条知识里找答案", criteria)]


def build_providers(samples, sections):
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
    out.append(RuleProvider("bm25", corpus=sections))
    if not any(isinstance(x, (JevProvider, LLMProvider)) for x in out):
        truth = {(s["state"], "section"): s["gold"]["section"] for s in samples}
        out.insert(0, MockProvider("calibrated", name="mock:决策模型(假数据)", truth=truth, hit_rate=0.72))
        out.insert(1, MockProvider("overconfident", name="mock:生成模型(假数据)", truth=truth, hit_rate=0.70))
    return out


def main() -> None:
    bare = len(sys.argv) > 1 and sys.argv[1] == "bare"
    level = "bare" if bare else "desc"
    sections = load_sections()
    samples = load_samples(sections)
    qs = questions(sections, bare=bare)
    provs = build_providers(samples, sections)
    mock = not any(isinstance(x, (JevProvider, LLMProvider)) for x in provs)
    print("  实际 provider:", ", ".join(p.name for p in provs), "\n")

    print(f"任务 B · 知识路由［{level}］：{len(sections)} 条知识，{len(samples)} 条 query\n")
    results = {}
    for p in provs:
        if isinstance(p, JevProvider) and not p.api_key:
            print(f"  跳过 {p.name}（未配 TYPESAFE_API_KEY）"); continue
        if isinstance(p, LLMProvider) and not (p.api_key and p.base_url and p.model):
            print(f"  跳过 {p.name}（未配 OPENAI_*）"); continue
        r = run_provider(p, samples, qs, main="section")
        results[p.name] = r
        c = r["calib"]
        flag = "" if r["conf_usable"] else "  ← 置信度无区分度"
        print(f"  {p.name:<24} 准确率 {c.accuracy:6.1%}  平均置信度 {c.mean_confidence:.3f}  "
              f"ECE {c.ece:.3f}  AURC {c.aurc:.3f}{flag}")

    md = build_report(
        f"任务 B · 知识路由［{level}］",
        f"把用户问法路由到 {len(sections)} 条知识中的正确一条。语料复用姊妹项目 `rag-lab` 的"
        "真实知识库（云雀客服 FAQ + 差旅报销政策）与 query，gold 是答案所在的小节。\n\n"
        "这是**高基数分类**场景 —— 候选多、且很多条知识语义相近（比如「退款到账时间」"
        "和「七天无理由退货规则」），靠关键词几乎分不开。\n\n"
        + ("**desc 档**：每个选项带一句从知识正文摘的说明 —— 这一档其实偏简单。\n"
           if not bare else
           "**bare 档**：只给标题、不给任何说明。这才是真实场景 —— 用户问法千变万化，"
           "标题就那几个字。\n"),
        samples, results, qs[0], ROOT / "results" / f"routing-{level}", mock)
    print(f"\n报告：{md.relative_to(ROOT)}")
    if mock:
        print("⚠️ 当前是 mock 数据。配好 key 重跑才是真结果。")


if __name__ == "__main__":
    main()
