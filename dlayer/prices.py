"""模型价目表（美元 / 百万 token）。

价格是会变的，**引用前先去官方页面核一遍**。这里记的是取数日期和口径，
免得哪天数字过期了还在被引用。

DeepSeek 官方：https://api-docs.deepseek.com/quick_start/pricing
- 取数日期 2026-09-21
- 这里记的是**非高峰（off-peak）**价，高峰时段翻倍
- 高峰 = 周一至周五 UTC 01:00-04:00 与 06:00-10:00，其余时段（含周末与中国法定节假日）都是非高峰
- 只区分「缓存未命中」的输入价（我们每次都是新 prompt，基本命中不了缓存）
"""
from __future__ import annotations

import os

FETCHED_ON = "2026-09-21"

# model -> (input_per_mtok, output_per_mtok)
TABLE: dict[str, tuple[float, float]] = {
    "deepseek-flash":   (0.15, 0.60),
    "deepseek-v4-pro":  (0.66, 1.98),
}


def price_of(model: str) -> tuple[float, float]:
    """查单价。查不到就返回 (0, 0) —— 报告里会显示 0，**别拿 0 当结论**。

    想用别的模型：在 .env 里写 MODEL_PRICES="模型名:输入价:输出价,另一个:..."
    """
    override = os.getenv("MODEL_PRICES", "")
    for item in filter(None, (s.strip() for s in override.split(","))):
        parts = item.split(":")
        if len(parts) == 3 and parts[0] == model:
            return float(parts[1]), float(parts[2])
    return TABLE.get(model, (0.0, 0.0))
