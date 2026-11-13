"""联网检索和网页读取工具。"""

from __future__ import annotations

import urllib.parse
import urllib.request
from typing import Any


def web_fetch(args: dict[str, Any], workspace: str) -> dict[str, Any]:
    if urllib.parse.urlsplit(str(args["url"])).scheme.lower() not in {"http", "https"}:
        raise PermissionError("网页工具只允许 HTTP/HTTPS，不能读取本地文件或其他协议。")
    request = urllib.request.Request(str(args["url"]), headers={"User-Agent": "VirgoExperiment/0.1"})
    with urllib.request.urlopen(request, timeout=20) as response:
        content = response.read().decode("utf-8", errors="ignore")
    return {"url": str(args["url"]), "content": content}


def web_search(args: dict[str, Any], workspace: str) -> dict[str, Any]:
    query = urllib.parse.quote_plus(str(args["query"]))
    url = f"https://html.duckduckgo.com/html/?q={query}"
    request = urllib.request.Request(url, headers={"User-Agent": "VirgoExperiment/0.1"})
    with urllib.request.urlopen(request, timeout=20) as response:
        content = response.read().decode("utf-8", errors="ignore")
    return {"query": str(args["query"]), "raw_html": content[:12000]}
