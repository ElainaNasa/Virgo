"""标准库 HTTP API，便于实验脚本和数据集批量调用。"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from virgo.system import VirgoSystem
from virgo.types import TaskRequest


class VirgoRequestHandler(BaseHTTPRequestHandler):
    system: VirgoSystem

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.strip("/").split("/")
        try:
            if path == ["health"]:
                self._send(200, {"status": "ok", "version": "0.1.0"})
            elif len(path) == 2 and path[0] == "tasks":
                self._send(200, self.system.get_task(path[1]))
            elif len(path) == 3 and path[0] == "tasks" and path[2] == "trace":
                self._send(200, self.system.get_trace(path[1]))
            elif len(path) == 3 and path[0] == "tasks" and path[2] == "logs":
                self._send(200, {"task_id": path[1], "logs": self.system.get_runtime_log(path[1])})
            else:
                self._send(404, {"error": "资源不存在"})
        except FileNotFoundError:
            self._send(404, {"error": "任务不存在"})
        except Exception as exc:
            self.system.execution_log.emit("http_request_failed", level="ERROR", status="failed", method="GET", path=self.path, error_type=type(exc).__name__, error=str(exc))
            self._send(500, {"error": str(exc)})

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/tasks":
            self._send(404, {"error": "资源不存在"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            request = TaskRequest(str(data["task"]), str(data.get("budget_tier", "low")), data.get("budget_cost"), tuple(data.get("constraints", [])), tuple(data.get("acceptance_criteria", [])))
            result = self.system.submit(request)
            self._send(201, {"task_id": result.task_id, "status": result.status, "output": result.output, "budget": result.snapshot["budget"]})
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send(400, {"error": f"请求格式错误：{exc}"})
        except Exception as exc:
            self.system.execution_log.emit("http_request_failed", level="ERROR", status="failed", method="POST", path=self.path, error_type=type(exc).__name__, error=str(exc))
            self._send(500, {"error": str(exc)})

    def _send(self, status: int, data: Any) -> None:
        payload = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:
        return


def serve(system: VirgoSystem | None = None, host: str = "127.0.0.1", port: int = 8765) -> None:
    VirgoRequestHandler.system = system or VirgoSystem()
    server = ThreadingHTTPServer((host, port), VirgoRequestHandler)
    print(f"Virgo API listening on http://{host}:{port}")
    server.serve_forever()
