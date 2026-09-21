"""统一的 HTTP opener —— 默认**不走系统代理**。

macOS 上 Python 的 urllib 会自动读系统网络设置里的代理（走 _scproxy），
即使环境变量里 `HTTP_PROXY` 是空的，`urllib.request.getproxies()` 也会返回
系统代理。这个行为很容易踩：

- `curl` 只读环境变量，所以它走直连、很快；
- Python 走系统代理，同样的请求却可能超时或挂住。

实测过一次：同一台机器上 curl 到 DeepSeek 0.2 秒返回，Python 却因为走了
系统代理卡了十几分钟，且 CPU 累计只用了 1.18 秒 —— 全在等网络。

所以这里显式关掉代理。真要代理就设 `USE_SYSTEM_PROXY=1`。
"""
from __future__ import annotations

import os
import urllib.request


def build_opener() -> urllib.request.OpenerDirector:
    if os.getenv("USE_SYSTEM_PROXY") == "1":
        return urllib.request.build_opener()
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


OPENER = build_opener()
