"""命令行入口。"""

from __future__ import annotations

import argparse
import json
import os
import sys

from virgo.system import VirgoSystem
from virgo.types import TaskRequest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Virgo 多智能体实验系统")
    parser.add_argument("task", nargs="?", help="待处理任务")
    parser.add_argument("--budget", default="low", choices=["xlow", "low", "medium", "high", "xhigh"])
    parser.add_argument("--budget-cost", type=float, default=None)
    parser.add_argument("--online", action="store_true", help="使用本地 OpenAI 兼容模型服务")
    parser.add_argument("--task-id", help="查看任务状态")
    parser.add_argument("--trace", action="store_true", help="查看任务决策轨迹")
    parser.add_argument("--serve", action="store_true", help="启动 HTTP API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    if args.online:
        os.environ["VIRGO_OFFLINE"] = "0"
    system = VirgoSystem()
    if args.serve:
        from virgo.server import serve
        serve(system, args.host, args.port)
        return 0
    if args.task_id:
        value = system.get_trace(args.task_id) if args.trace else system.get_task(args.task_id)
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return 0
    if not args.task:
        parser.error("请提供 task 或 --task-id")
    result = system.submit(TaskRequest(args.task, args.budget, args.budget_cost))
    print(json.dumps({"task_id": result.task_id, "status": result.status, "output": result.output, "budget": result.snapshot["budget"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
