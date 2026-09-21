"""置信度校准评测。

比"谁准确率高"更能说明问题的一组指标：

  ECE / 可靠性图 —— 模型说 0.9 的时候，真的有 90% 是对的吗？
  Brier          —— 概率预测的均方误差，同时惩罚"错得自信"
  弃权曲线 / AURC —— **允许模型跳过自己最不确定的那批样本时，剩下的准确率怎么变**

最后一个是重点。真实业务里没人会让模型硬答所有题 —— 低置信度的应该转人工。
校准好的模型，这条曲线是陡的（弃权 20% 就能换来明显更高的准确率）；
过度自信的模型，这条曲线几乎是平的（因为它"不确定"的样本和"确定"的样本混在一起，
挑不出来）。**一个模型能不能用置信度做路由，看这一张图就够了。**
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Calibration:
    n: int
    accuracy: float
    mean_confidence: float
    ece: float
    brier: float
    aurc: float
    bins: list[tuple[float, float, int]]     # (平均置信度, 实际准确率, 样本数)


def evaluate(y_conf: list[float], y_correct: list[int], n_bins: int | None = None) -> Calibration:
    assert len(y_conf) == len(y_correct)
    n = len(y_conf)
    if n == 0:
        return Calibration(0, 0, 0, 0, 0, 0, [])
    # 每箱至少要有几个样本才有意义，否则分箱全是噪声
    if n_bins is None:
        n_bins = max(3, min(10, n // 8))
    acc = sum(y_correct) / n
    mean_conf = sum(y_conf) / n
    brier = sum((c - k) ** 2 for c, k in zip(y_conf, y_correct)) / n

    # --- ECE：按置信度分箱，看"箱内平均置信度"和"箱内实际准确率"差多少 ---
    edges = [i / n_bins for i in range(n_bins + 1)]
    bins, ece = [], 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        idx = [j for j, c in enumerate(y_conf)
               if (lo <= c < hi) or (i == n_bins - 1 and c == 1.0)]
        if not idx:
            continue
        bc = sum(y_conf[j] for j in idx) / len(idx)
        ba = sum(y_correct[j] for j in idx) / len(idx)
        bins.append((bc, ba, len(idx)))
        ece += len(idx) / n * abs(ba - bc)

    return Calibration(n, acc, mean_conf, ece, brier, _aurc(y_conf, y_correct), bins)


def abstention_curve(y_conf: list[float], y_correct: list[int],
                     steps: int = 20) -> list[tuple[float, float]]:
    """按置信度从高到低排序，取前 coverage 比例，算这批的准确率。

    这就是「低置信度转人工」策略的效果曲线：横轴是自动处理的比例，
    纵轴是自动处理那部分的准确率。
    """
    if not y_conf:
        return []
    order = sorted(range(len(y_conf)), key=lambda i: y_conf[i], reverse=True)
    out = []
    for s in range(1, steps + 1):
        cov = s / steps
        k = max(1, int(round(cov * len(order))))
        sel = order[:k]
        out.append((k / len(order), sum(y_correct[i] for i in sel) / k))
    return out


def _aurc(y_conf: list[float], y_correct: list[int]) -> float:
    """风险-覆盖率曲线下的面积（越小越好）。

    风险 = 1 - 准确率。随机弃权时 AURC ≈ 0.5*(1-acc) 这个量级；
    好的模型应该显著低于它。
    """
    curve = abstention_curve(y_conf, y_correct, steps=50)
    if not curve:
        return 0.0
    area = sum((1 - a) for _, a in curve) / len(curve)
    return area


def confidence_usable(y_conf: list[float], max_dominant: float = 0.7) -> tuple[bool, str]:
    """置信度能不能拿来排序/设阈值？

    如果绝大多数样本挤在同一个置信度上，排序就是任意的 —— 这时候弃权曲线
    画出来再好看也没有意义。规则的"命中就是 1.0、兜底就是 1/n"正是这种情况，
    必须显式标出来，否则图表会误导人。
    """
    if not y_conf:
        return False, "无数据"
    from collections import Counter
    top, cnt = Counter(round(c, 3) for c in y_conf).most_common(1)[0]
    frac = cnt / len(y_conf)
    if frac >= max_dominant:
        return False, f"{frac:.0%} 的样本置信度都是 {top}，排序无区分度"
    return True, ""


def summary_table(results: dict[str, Calibration]) -> str:
    """markdown 表格，方便直接贴进报告。"""
    head = ("| 方案 | 准确率 | 平均置信度 | ECE ↓ | Brier ↓ | AURC ↓ |\n"
            "|---|---|---|---|---|---|\n")
    rows = []
    for name, c in results.items():
        rows.append(f"| {name} | {c.accuracy:.1%} | {c.mean_confidence:.3f} | "
                    f"{c.ece:.3f} | {c.brier:.3f} | {c.aurc:.3f} |")
    return head + "\n".join(rows)


# ---------------- 纯 Python 画 SVG，不依赖 matplotlib ----------------
# 两条约定：图例一律放在绘图区外的底部（放图里会压住数据），
# 分箱数随样本量自适应（几十个样本分 10 箱，每箱一两个点，画出来全是噪声）。

_PALETTE = ["#2f5d8a", "#c0603a", "#4f81bd", "#8a8f98"]


def _legend(names: list[str], y: int, w: int, cols: int = 2) -> list[str]:
    out, cw = [], w // cols
    for i, n in enumerate(names):
        x = 20 + (i % cols) * cw
        r = y + (i // cols) * 16
        out.append(f'<line x1="{x}" y1="{r}" x2="{x+14}" y2="{r}" '
                   f'stroke="{_PALETTE[i%4]}" stroke-width="2.5"/>'
                   f'<text x="{x+19}" y="{r+4}" font-size="11" fill="#3a4356">{n}</text>')
    return out


def _axes(pad: int, w: int, h: int, xlab: str, ylab: str) -> list[str]:
    return [f'<text x="{(pad+w)/2:.0f}" y="{h-8}" font-size="11.5" fill="#5b6478" '
            f'text-anchor="middle">{xlab}</text>',
            f'<text x="14" y="{(pad+h)/2:.0f}" font-size="11.5" fill="#5b6478" '
            f'transform="rotate(-90 14 {(pad+h)/2:.0f})" text-anchor="middle">{ylab}</text>']


def svg_abstention(series: dict[str, list[tuple[float, float]]], w: int = 560, h: int = 400) -> str:
    pad, top = 52, 18
    legend_h = 22 + 16 * ((len(series) + 1) // 2)
    plot_h = h - pad - legend_h - top
    def X(c): return pad + c * (w - pad * 1.3)
    def Y(a): return top + (1 - a) * plot_h
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
         f'viewBox="0 0 {w} {h}" font-family="PingFang SC,Helvetica,Arial">',
         f'<rect width="{w}" height="{h}" fill="#fff"/>']
    for a in (0.4, 0.6, 0.8, 1.0):
        p.append(f'<line x1="{X(0):.0f}" y1="{Y(a):.0f}" x2="{X(1):.0f}" y2="{Y(a):.0f}" '
                 f'stroke="#eef2f8"/><text x="{pad-8}" y="{Y(a)+4:.0f}" font-size="10" '
                 f'fill="#7a8398" text-anchor="end">{a:.0%}</text>')
    for c in (0.25, 0.5, 0.75, 1.0):
        p.append(f'<text x="{X(c):.0f}" y="{top+plot_h+16}" font-size="10" fill="#7a8398" '
                 f'text-anchor="middle">{c:.0%}</text>')
    for i, (name, curve) in enumerate(series.items()):
        pts = " ".join(f"{X(c):.1f},{Y(a):.1f}" for c, a in curve)
        p.append(f'<polyline points="{pts}" fill="none" stroke="{_PALETTE[i%4]}" stroke-width="2.2"/>')
    p.append(f'<line x1="{X(0):.0f}" y1="{top:.0f}" x2="{X(0):.0f}" y2="{top+plot_h:.0f}" stroke="#c9d3e2"/>')
    p += _legend(list(series), top + plot_h + 34, w)
    p += _axes(pad, w, h, "自动处理比例（其余转人工）", "自动处理部分的准确率")
    p.append("</svg>")
    return "".join(p)


def svg_reliability(series: dict[str, Calibration], w: int = 460, h: int = 400) -> str:
    pad, top = 52, 18
    legend_h = 22 + 16 * ((len(series) + 1) // 2)
    plot_h = h - pad - legend_h - top
    def X(c): return pad + c * (w - pad * 1.3)
    def Y(a): return top + (1 - a) * plot_h
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
         f'viewBox="0 0 {w} {h}" font-family="PingFang SC,Helvetica,Arial">',
         f'<rect width="{w}" height="{h}" fill="#fff"/>']
    for a in (0.4, 0.6, 0.8, 1.0):
        p.append(f'<line x1="{X(0):.0f}" y1="{Y(a):.0f}" x2="{X(1):.0f}" y2="{Y(a):.0f}" '
                 f'stroke="#f2f5fa"/><text x="{pad-8}" y="{Y(a)+4:.0f}" font-size="10" '
                 f'fill="#7a8398" text-anchor="end">{a:.0%}</text>')
    p.append(f'<line x1="{X(0):.0f}" y1="{Y(0):.0f}" x2="{X(1):.0f}" y2="{Y(1):.0f}" '
             f'stroke="#c9d3e2" stroke-dasharray="5 4"/>')
    p.append(f'<text x="{X(0.80):.0f}" y="{Y(0.72):.0f}" font-size="10.5" fill="#9aa4b5">完美校准</text>')
    for i, (name, c) in enumerate(series.items()):
        if not c.bins:
            continue
        pts = " ".join(f"{X(bc):.1f},{Y(ba):.1f}" for bc, ba, _ in c.bins)
        p.append(f'<polyline points="{pts}" fill="none" stroke="{_PALETTE[i%4]}" '
                 f'stroke-width="1.8" opacity="0.85"/>')
        for bc, ba, k in c.bins:
            p.append(f'<circle cx="{X(bc):.1f}" cy="{Y(ba):.1f}" r="{2.2+math.sqrt(k)*1.1:.1f}" '
                     f'fill="{_PALETTE[i%4]}" opacity="0.5"/>')
    p += _legend(list(series), top + plot_h + 34, w)
    p += _axes(pad, w, h, "模型自报置信度", "实际准确率")
    p.append("</svg>")
    return "".join(p)
