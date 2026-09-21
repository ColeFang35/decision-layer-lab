"""任务执行与报告生成。两个任务共用这一套流程。"""
from __future__ import annotations

import dataclasses
import pathlib
import time

from .calibration import (Calibration, abstention_curve, confidence_usable, evaluate,
                          svg_abstention, svg_reliability, summary_table)
from .providers.base import Provider, Question


def run_provider(provider: Provider, samples: list[dict], questions: list[Question],
                 main: str) -> dict:
    """对一批样本跑一个 provider，收集主问题的 (置信度, 是否正确) 以及耗时成本。"""
    confs, corrects, rows = [], [], []
    lat = cost = 0.0
    errors = 0

    gold_of = {s["id"]: s["gold"][main] for s in samples}
    step = max(1, len(samples) // 5)
    for i, s in enumerate(samples):
        if i % step == 0:                    # 进度可见，别让长跑变成黑盒
            print(f"    [{provider.name}] {i}/{len(samples)}", flush=True)
        res = provider.run(s["state"], questions)
        if res.error:
            errors += 1
            rows.append({"id": s["id"], "pred": "", "gold": gold_of[s["id"]],
                         "conf": 0.0, "correct": 0, "error": res.error})
            confs.append(0.0)
            corrects.append(0)
            continue
        lat += res.latency_ms
        cost += res.cost_usd
        p = res.answers.get(main)
        if p is None:
            errors += 1
            confs.append(0.0); corrects.append(0)
            continue
        ok = int(p.label == gold_of[s["id"]])
        confs.append(p.confidence)
        corrects.append(ok)
        rows.append({"id": s["id"], "pred": p.label, "gold": gold_of[s["id"]],
                     "conf": p.confidence, "correct": ok, "error": None})

    n = len(samples)
    usable, why = confidence_usable(confs)
    return {
        "name": provider.name,
        "calib": evaluate(confs, corrects),
        "conf_usable": usable,
        "conf_why": why,
        "confs": confs,
        "curve": abstention_curve(confs, corrects),
        "rows": rows,
        "latency_ms_avg": lat / n if n else 0,
        "cost_usd_total": cost,
        "cost_usd_avg": cost / n if n else 0,
        "errors": errors,
    }


def build_report(task_title: str, task_desc: str, samples: list[dict],
                 results: dict[str, dict], main_q: Question, out_dir: pathlib.Path,
                 mock: bool) -> pathlib.Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    banner = ("> ⚠️ **本次是 mock 数据，不是实验结果。** 报告里的数字由 `MockProvider` 生成，\n"
              "> 只用于验证指标与图表是否正确。配好 API key 重跑才有真数字。\n\n") if mock else ""

    calibs: dict[str, Calibration] = {n: r["calib"] for n, r in results.items()}
    # 置信度无区分度的方案不画进弃权曲线：它的排序是任意的，画出来会误导
    curves = {n: r["curve"] for n, r in results.items() if r["conf_usable"]}
    skipped = [f"`{n}`（{r['conf_why']}）" for n, r in results.items() if not r["conf_usable"]]

    cost_rows = "\n".join(
        f"| {n} | {r['latency_ms_avg']:.0f} ms | {r['cost_usd_avg']*1000:.4f} 厘 | "
        f"{r['errors']} |" for n, r in results.items())

    body = f"""# {task_title}

{banner}## 任务

{task_desc}

- 样本数：**{len(samples)}**
- 主问题（做校准评测的那个）：`{main_q.name}` —— {main_q.instructions}
- 候选数：**{len(main_q.options)}**
- 评测时间：{time.strftime('%Y-%m-%d %H:%M')}

## 结果

{summary_table(calibs)}

置信度可用性：{("**下列方案的置信度没有区分度，已从弃权曲线中剔除**：" + "、".join(skipped)) if skipped else "各方案的置信度均有区分度。"}

### 延迟与成本

| 方案 | 单条延迟 | 单条成本 | 失败数 |
|---|---|---|---|
{cost_rows}

> 规则的延迟是纯本地字符串/BM25 匹配，所以接近 0；成本对比取决于你在 `.env` 里
> 填的单价（`OPENAI_PRICE_IN` / `OPENAI_PRICE_OUT`，单位是美元/百万 token），
> 没填就记 0，不要拿 0 当结论。

## 关键图：弃权曲线

允许「低置信度转人工」时，自动处理那部分的准确率会怎么变。
**这条线越陡，说明模型的置信度越可信、越适合拿来设阈值。**

![弃权曲线](abstention.svg)

> 横轴 = 自动处理的比例（其余转人工），纵轴 = 这部分样本的准确率。
> 一个只会硬答的模型，这条线是平的 —— 因为它"心虚"的样本和"有把握"的样本
> 在置信度上分不开。AURC 就是这条线下方风险面积的均值，**越小越好**。

## 可靠性图

模型说 0.9 的时候，真的有 90% 是对的吗？

![可靠性图](reliability.svg)

> 点越靠近对角线越准。**普遍落在对角线下方 = 过度自信** —— 这是生成式模型
> 自我评估的常见毛病，也是"拿它自报的置信度设阈值"这件事最大的坑。

## 明细

逐条预测见同目录 `rows.json`。
"""
    md = out_dir / "report.md"
    md.write_text(body, encoding="utf-8")
    (out_dir / "abstention.svg").write_text(svg_abstention(curves), encoding="utf-8")
    (out_dir / "reliability.svg").write_text(svg_reliability(calibs), encoding="utf-8")
    import json
    dump = {n: {**r, "calib": dataclasses.asdict(r["calib"])} for n, r in results.items()}
    (out_dir / "rows.json").write_text(json.dumps(dump, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    return md
