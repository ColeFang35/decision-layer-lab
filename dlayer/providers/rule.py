"""关键词规则 baseline —— 也就是"上模型之前"的做法。

两种模式：
  keyword  : 靠人工维护的关键词表命中（工单分诊用）
  bm25     : BM25 检索打分（知识路由用，等价于 rag-lab 里的关键词方案）

它给不出有意义的置信度，所以统一记 confidence = 1.0（命中）或 1/n（兜底）。
**这正是要对比的点**：规则方案要么确定要么瞎猜，中间没有过渡，
所以「低置信度转人工」这个策略在它身上根本用不了。
"""
from __future__ import annotations

import re
import time

import jieba
from rank_bm25 import BM25Okapi

from .base import Prediction, Provider, ProviderResult, Question


class RuleProvider(Provider):
    name = "rule"

    def __init__(self, mode: str = "keyword", keyword_table: dict[str, list[str]] | None = None,
                 corpus: dict[str, str] | None = None):
        self.mode = mode
        self.keyword_table = keyword_table or {}
        self.corpus = corpus or {}
        self._bm25 = None
        self._keys: list[str] = []
        if mode == "bm25" and self.corpus:
            self._keys = list(self.corpus)
            self._bm25 = BM25Okapi([list(jieba.cut(v)) for v in self.corpus.values()])

    def run(self, state: str, questions: list[Question]) -> ProviderResult:
        t0 = time.time()
        answers = {}
        for q in questions:
            if self.mode == "bm25":
                answers[q.name] = self._bm25_pick(state, q)
            else:
                answers[q.name] = self._keyword_pick(state, q)
        return ProviderResult(self.name, answers, (time.time() - t0) * 1000, 0.0)

    # ---------- keyword ----------
    def _keyword_pick(self, state: str, q: Question) -> Prediction:
        opts = q.options
        hits = {o: sum(1 for kw in self.keyword_table.get(o, []) if kw in state) for o in opts}
        best = max(hits, key=lambda o: hits[o]) if hits else ""
        if not best or hits[best] == 0:                    # 一条没命中 → 兜底
            return Prediction(opts[0] if opts else "", 1.0 / max(1, len(opts)), probs=None)
        # 命中即确定。规则方案给不出"有点不确定"这种状态。
        return Prediction(best, 1.0, probs=None)

    # ---------- bm25 ----------
    def _bm25_pick(self, state: str, q: Question) -> Prediction:
        if self._bm25 is None:
            return Prediction("", 0.0)
        scores = self._bm25.get_scores(list(jieba.cut(state)))
        order = sorted(range(len(self._keys)), key=lambda i: scores[i], reverse=True)
        top, second = order[0], (order[1] if len(order) > 1 else order[0])
        # 用第一名与第二名分差造一个很粗的置信度（BM25 分数本身无上界、不能当概率）
        margin = (scores[top] - scores[second]) / (scores[top] + 1e-9)
        return Prediction(self._keys[top], max(0.0, min(1.0, margin)), probs=None)


TRIAGE_KEYWORDS = {
    "发货时效": ["发货", "发出", "什么时候发", "多久发", "预售", "双十一"],
    "运费规则": ["运费", "包邮", "快递费", "邮费"],
    "退货退款": ["退货", "退了", "退换", "不想要", "七天无理由", "吊牌"],
    "退款到账": ["退款", "到账", "多久到", "退回", "退到"],
    "发票": ["发票", "开票", "抬头", "税号", "专票"],
    "会员积分": ["会员", "积分", "等级", "升级", "黄金"],
    "修改地址": ["地址", "改地址", "收货地址"],
    "物流快递": ["物流", "快递", "签收", "驿站", "顺丰"],
    "支付问题": ["支付", "付款", "扣了", "花呗", "扣款"],
    "投诉": ["投诉", "主管", "态度", "消协", "失望"],
}
