"""评测联网边界：允许资料查询，阻止获取当前题目解答。"""

import re
from urllib.parse import unquote


def answer_lookup_request(arguments: dict, task: str) -> bool:
    """按请求内容识别题号、题面片段及明确求解搜索。"""
    text = unquote(str(arguments.get("query") or arguments.get("url") or "")).lower()
    normalized = re.sub(r"[^\w]+", " ", text)
    if re.search(r"\b(aime|bigcodebench|humaneval|mmlu|swe.?bench)\b", text):
        return True
    if re.search(r"题解|答案|原题|题号|解答|solutions?|answer[ _/-]?key|editorial|walkthrough", text):
        return True
    # 禁止复制题面片段搜索；通用 API 名称和短术语仍可使用。
    words = re.findall(r"\w+", task.lower())
    for size in (6,):
        if any(" ".join(words[index:index + size]) in normalized for index in range(max(0, len(words) - size + 1))):
            return True
    fragments = re.findall(r"[\u4e00-\u9fff]{10,}", task)
    return any(fragment[index:index + 10] in text for fragment in fragments for index in range(len(fragment) - 9))
